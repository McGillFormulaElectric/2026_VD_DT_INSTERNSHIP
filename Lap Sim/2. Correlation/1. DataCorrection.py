import sys
import copy
from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# Path to the Lap Sim folder
LAPSIM_ROOT = Path(__file__).resolve().parents[1]   # this file is in "2. Correlation"
DATA_DIR = LAPSIM_ROOT / "1. Data" / "Motec"
TIRE_DIR = LAPSIM_ROOT / "1. Data" / "Tire"
TRACK_DIR = LAPSIM_ROOT / "1. Data" / "Track"

# ── inputs, change these ──────────────────────────────────────
R_CG_TO_IMU = np.array([-0.255, 0.0, -0.027865])   # m, from the CG to the IMU, car axes: x forward, y left, z up. ENTER FROM CAD

# log: what it correlates. accel -> efficiency_scale, brake -> muxScale, skidpad -> muyScale, lap -> check only
LOGS = {
    "accel.mat":               "accel",
    "brake.mat":               "brake",
    "skidpad2.mat":            "skidpad",
    "autocross.mat":           "lap",
    "TorontoShootout2026.mat": "lap",
}
TARGET_LOGS = {"accel": "accel.mat", "brake": "brake.mat", "skidpad": "skidpad2.mat"}   # the run each scale was correlated on
GRADIENT_LOGS = ["autocross.mat", "TorontoShootout2026.mat"]   # turn both ways, so the mounting offset cancels
TILT_FIT_LOGS = ["accel.mat", "autocross.mat", "TorontoShootout2026.mat"]   # raw accel - GPS angle, brake.mat left out (GPS glitch)
STOP_LOG = "brake.mat"                                         # ends with the car stopped

AY_TARGET = 1.3      # g, roll is reported at this lateral acceleration
AX_TARGET = 1.0      # g, pitch is reported at this longitudinal acceleration
R_E = 0.1997         # m, effective rolling radius from RollingRadius.py
R_SKIDPAD = 9.67     # m, skidpad path radius, for the camber load case
GEAR_RATIO = 13.39   # -
CAMBER_GAIN_F = 0.0  # deg of wheel camber recovered per deg of body roll, front (0 = none, worst case), from kinematics
CAMBER_GAIN_R = 0.0  # same, rear
RUN_SIM = True       # False skips part D (no lap sim needed)
PLOT = True          # False skips the IMU tilt graph

CUTOFF = 2.0         # Hz, low pass on every channel
G = 9.81             # m/s^2
IMU_SIGN = -1        # IMU x axis points rearward: accX, accY, gyrX, gyrY, Roll, Pitch flip. Set +1 once fixed
WHEELS = ["FL", "FR", "RL", "RR"]


def lowpass(x, fs=100.0):
    b, a = butter(2, CUTOFF / (fs / 2))
    return filtfilt(b, a, x)


def load(log):
    # every signal in car axes (x forward, y left, z up), SI units
    mat = sio.loadmat(DATA_DIR / log, squeeze_me=True, struct_as_record=False)
    ch = lambda name: np.asarray(mat[name].Value, float)
    s = {}
    t = np.asarray(mat["velX"].Time, float)
    s["t"] = t - t[0]                                                          # s
    s["v"] = lowpass(np.hypot(ch("velX"), ch("velY")))                         # m/s, GPS speed
    s["course"] = np.degrees(np.arctan2(ch("velY"), ch("velX")))               # deg, direction of travel, 0 = East
    rpm = np.median([ch("SpeedActual" + c) for c in WHEELS], axis=0)           # rpm, median motor
    s["v_wheel"] = lowpass(rpm / GEAR_RATIO * 2 * np.pi / 60 * R_E)            # m/s
    s["throttle"] = ch("THROTTLE_PERCENT")                                     # %
    roll_imu, pitch_imu = np.deg2rad(ch("Roll")), np.deg2rad(ch("Pitch"))      # rad, IMU frame
    s["roll"] = np.rad2deg(lowpass(IMU_SIGN * roll_imu))                       # deg, car, + = rolls right, includes mounting offset
    s["pitch"] = np.rad2deg(lowpass(IMU_SIGN * pitch_imu))                     # deg, car, + = nose down, includes mounting offset
    s["accX_imu"], s["accZ_imu"] = ch("accX"), ch("accZ")                      # m/s^2, as logged, for the tilt check

    # 1. raw: what the IMU reads, in car axes
    s["ax_raw"] = lowpass(IMU_SIGN * ch("accX"))
    s["ay_raw"] = lowpass(IMU_SIGN * ch("accY"))
    # 2. gravity removed: the IMU is tilted by roll and pitch, so it feels part of g
    s["ax_g"] = lowpass(IMU_SIGN * (ch("accX") + G * np.sin(pitch_imu)))
    s["ay_g"] = lowpass(IMU_SIGN * (ch("accY") - G * np.sin(roll_imu)))
    # 3. at the CG: a_cg = a_imu - w_dot x r - w x (w x r), r from CG to IMU
    w = np.array([lowpass(IMU_SIGN * ch("gyrX_HR")), lowpass(IMU_SIGN * ch("gyrY_HR")), lowpass(ch("gyrZ_HR"))])   # rad/s
    w_dot = np.gradient(w, s["t"], axis=1)
    rr = np.repeat(R_CG_TO_IMU[:, None], len(t), axis=1)
    corr = np.cross(w_dot, rr, axis=0) + np.cross(w, np.cross(w, rr, axis=0), axis=0)
    s["ax_cg"] = s["ax_g"] - corr[0]
    s["ay_cg"] = s["ay_g"] - corr[1]
    # 4. from speed only, no IMU tilt or position
    s["ax_gps"] = np.gradient(s["v"], s["t"])                                  # m/s^2, d|v|/dt from GPS
    s["ax_whl"] = np.gradient(s["v_wheel"], s["t"])                            # m/s^2, from wheel speed
    s["ay_kin"] = s["v"] * w[2]                                                # m/s^2, v * yaw rate
    return s


def sections(kind, s):
    # (name, axis, mask, sign): the part of each log that sets a correlation target
    # the mask uses the average of the IMU and the speed based value, so it doesn't favour either one
    ax_mid = (s["ax_g"] + (s["ax_whl"] if kind == "brake" else s["ax_gps"])) / 2   # brake.mat: GPS glitch, use wheels
    ay_mid = (np.abs(s["ay_g"]) + np.abs(s["ay_kin"])) / 2
    out = []
    if kind == "accel":
        out.append(("accel, full throttle", "x", (s["throttle"] >= 95) & (s["v"] > 5), +1))
    if kind in ("brake", "lap"):
        out.append(("braking > 0.5 g", "x", (ax_mid < -0.5 * G) & (s["v"] > 3), -1))
    if kind == "skidpad":
        out.append(("skidpad, steady", "y", np.abs(s["v"] / np.median(s["v"]) - 1) < 0.05, +1))
    if kind == "lap":
        out.append(("cornering > 1 g", "y", ay_mid > 1.0 * G, +1))
    return out


data = {log: load(log) for log in LOGS}
summary = []   # (item, number, conclusion), printed at the end

# ── A. roll and pitch gradients (Q2) ──────────────────────────
def gradient(logs, angle, acc):
    # deg/g through zero, each log's own offset (mounting + track slope) removed first. Also returns the median offset
    xs, ys, offs = [], [], []
    for log in logs:
        s = data[log]
        m = s["v"] > 3
        x, y = s[acc][m] / G, s[angle][m]
        k, off = np.polyfit(x, y, 1)
        xs.append(x); ys.append(y - off); offs.append(off)
    x, y = np.concatenate(xs), np.concatenate(ys)
    return np.sum(x * y) / np.sum(x * x), np.median(offs)

k_roll, roll_mount = gradient(GRADIENT_LOGS, "roll", "ay_cg")
k_pitch, pitch_mount = gradient(GRADIENT_LOGS + ["accel.mat", "brake.mat"], "pitch", "ax_cg")
roll = abs(k_roll) * AY_TARGET                                                 # deg
pitch = abs(k_pitch) * AX_TARGET                                               # deg
leak_y = np.sin(np.deg2rad(roll)) / AY_TARGET * 100                            # %
leak_x = np.sin(np.deg2rad(pitch)) / AX_TARGET * 100                           # %
print("A. BODY ROLL AND PITCH")
print(f"   roll gradient  {abs(k_roll):.2f} deg/g  ->  {roll:.2f} deg at {AY_TARGET} g")
print(f"   pitch gradient {abs(k_pitch):.2f} deg/g  ->  {pitch:.2f} deg at {AX_TARGET} g")
print(f"   gravity leaking into the IMU from body motion: lateral {leak_y:.2f} % at {AY_TARGET} g, "
      f"longitudinal {leak_x:.2f} % at {AX_TARGET} g")
print(f"   IMU tilt at rest: roll {roll_mount:+.2f} deg, pitch {pitch_mount:+.2f} deg"
      f"  ->  constant {np.sin(np.deg2rad(abs(roll_mount))):.3f} g lateral, {np.sin(np.deg2rad(abs(pitch_mount))):.3f} g longitudinal")
print("   column 2 in part C removes both (body motion + tilt at rest)")
summary.append(("Body roll", f"{roll:.2f} deg at {AY_TARGET} g ({abs(k_roll):.2f} deg/g)", "small"))
summary.append(("Body pitch", f"{pitch:.2f} deg at {AX_TARGET} g ({abs(k_pitch):.2f} deg/g)", "small"))
summary.append(("Gravity leak from roll", f"{leak_y:.1f} % of gLat at {AY_TARGET} g", "remove it with the Roll channel"))

# ── B. IMU tilt at rest: where the ~4.3 deg comes from ────────
print("\nB. IMU TILT AT REST (deg, + = nose down)")
s = data[STOP_LOG]
stop = (s["t"] > s["t"][-1] - 0.5) & (s["v_wheel"] < 0.2)                     # last 0.5 s, wheels stopped (not still braking)
angle_stop = np.degrees(np.arctan2(IMU_SIGN * -s["accX_imu"][stop].mean(), s["accZ_imu"][stop].mean())) if stop.sum() > 20 else np.nan
print(f"   car stopped ({STOP_LOG}, {stop.sum() * 0.01:.1f} s): atan(accX / accZ) = {angle_stop:.2f}   <- accelerometer only feels gravity")
tilt_names, tilt_pitch, tilt_off = [], [], []
for log in TILT_FIT_LOGS:
    s = data[log]
    m = s["v"] > 5
    off = np.mean(s["ax_raw"][m] - s["ax_gps"][m])                             # m/s^2, constant part = -g sin(angle)
    tilt_names.append(log[:-4]); tilt_pitch.append(s["pitch"].mean()); tilt_off.append(np.degrees(np.arcsin(-off / G)))
    print(f"   {log:26s} Pitch channel mean {tilt_pitch[-1]:.2f}   from raw accel - GPS d|v|/dt {tilt_off[-1]:.2f}")
dir_means = []
for log in GRADIENT_LOGS:                                                      # pitch per 30 deg of travel direction
    s = data[log]
    m = s["v"] > 5
    idx = np.digitize(s["course"][m], np.arange(-180, 181, 30))
    dir_means += [s["pitch"][m][idx == i].mean() for i in np.unique(idx) if (idx == i).sum() > 20]
print(f"   pitch vs direction of travel: {min(dir_means):.2f} to {max(dir_means):.2f} in every direction "
      f"(a track slope would flip sign, this doesn't)")
all_pitch = np.mean([data[log]["pitch"].mean() for log in LOGS])
summary.append(("IMU tilt at rest", f"{all_pitch:.1f} deg nose down (stopped car {angle_stop:.1f})",
                f"same in every log and direction ({min(dir_means):.1f} to {max(dir_means):.1f}): IMU mount + rake, not slope"))
summary.append(("Gravity leak from tilt", f"{np.sin(np.deg2rad(all_pitch)):.3f} g on accX", "never use raw accX"))

# ── C. correlation targets, computed every way (Q2 + Q3) ──────
print("\nC. CORRELATION TARGETS (g)          1 raw 2 -grav  3 @CG   4 GPS 4 wheel  |  tilt+roll  IMU pos   3 vs 4")
print("                                                                           |   (1 vs 2) (2 vs 3)")
err = {}
for log, kind in LOGS.items():
    s = data[log]
    for name, axis, m, sign in sections(kind, s):
        if m.sum() < 20:
            continue
        val = lambda ch: np.mean(sign * s[ch][m]) / G if axis == "x" else np.mean(np.abs(s[ch][m])) / G
        raw, grav, cg = val(f"a{axis}_raw"), val(f"a{axis}_g"), val(f"a{axis}_cg")
        k1 = val("ax_gps") if axis == "x" else val("ay_kin")
        k2 = val("ax_whl") if axis == "x" else np.nan
        ref = k2 if kind == "brake" else k1                                    # brake.mat: GPS glitch, compare to wheels
        d_tilt, d_pos, d_ref = (raw / grav - 1) * 100, (cg / grav - 1) * 100, (cg / ref - 1) * 100   # %
        print(f"   {log[:-4][:12]:12s} {name:20s}{raw:7.3f}{grav:8.3f}{cg:8.3f}{k1:8.3f}{k2:8.3f}  |"
              f"{d_tilt:+9.2f} %{d_pos:+8.2f} %{d_ref:+8.2f} %")
        if TARGET_LOGS.get(kind) == log:
            err[kind] = raw / cg - 1                                           # -, error if the raw IMU were used
            ref_name = "wheels" if kind == "brake" else ("GPS" if axis == "x" else "v*r")
            verdict = "agree, use corrected IMU" if abs(d_ref) < 2 else "differ > 2 %, check which is right"
            summary.append((f"{kind.capitalize()} target", f"{cg:.3f} g (raw {raw:.3f}, {ref_name} {ref:.3f})",
                            f"raw off {(raw / cg - 1) * 100:+.1f} %, corrected vs {ref_name} {d_ref:+.1f} %: {verdict}"))
            if kind == "skidpad":
                summary.append(("IMU position (Q3)", f"{d_pos:+.2f} % on gLat",
                                "enter R_CG_TO_IMU from CAD" if not np.any(R_CG_TO_IMU) else f"applied, r = {R_CG_TO_IMU} m"))
print("   3 vs 4: two independent measurements (IMU vs speed). Close = trust column 3.")
print("   4 GPS = d|v|/dt or v * yaw rate, 4 wheel = d(wheel speed)/dt. brake.mat is compared to wheels (GPS glitch)")

# ── D. what it does to the lap sim (Q2: put a number to it) ───
if RUN_SIM:
    try:
        sys.path.insert(0, str(LAPSIM_ROOT))
        from TrackMap import loadTrack
        from CarProperties import MFE26
        from Tire import Tire
        from Acceleration import AccelSolver
        from SkidPad import SkidPadSolver
        from Autocross import solve
        from Endurance import solve as endurance_solve

        car_base = MFE26()
        tire_base = Tire(TIRE_DIR / "MF61_Parameters.mat", TIRE_DIR / "FZ_Reference.mat")
        track_autocross = loadTrack(TRACK_DIR, "Michigan2026_Autocross")
        track_endurance = loadTrack(TRACK_DIR, "Michigan2026_Endurance")
        EVENTS = {
            "Accel":     lambda car, tire: AccelSolver().simulate(tire, car)["lap_time"],
            "Skidpad":   lambda car, tire: SkidPadSolver().simulate(tire, car)["lap_time"],
            "Autocross": lambda car, tire: solve(tire, car, track_autocross)["lap_time"],
            "Endurance": lambda car, tire: endurance_solve(tire, car, track_endurance)["total_time"],
        }

        # camber the sim never sees: roll tilts both wheels out of the turn
        alpha = np.radians(np.linspace(0, 15, 300))
        c = car_base
        v_sp = np.sqrt(AY_TARGET * G * R_SKIDPAD)                              # m/s, skidpad speed at AY_TARGET
        Fa = 0.5 * c.air_density * c.CLA * v_sp ** 2                           # N
        LT = c.mass_total * AY_TARGET * G * c.CG_height / c.track              # N
        loss, total = 0.0, 0.0
        for frac, bal, lltd, cam0, gain in [(c.CGx, c.aero_balance, c.lltd, c.camber_front, CAMBER_GAIN_F),
                                            (1 - c.CGx, 1 - c.aero_balance, 1 - c.lltd, c.camber_rear, CAMBER_GAIN_R)]:
            Fz0 = (c.mass_total * G * frac + Fa * bal) / 2
            dcam = roll * (1 - gain)
            for Fz in (Fz0 + LT * lltd, Fz0 - LT * lltd):
                base = [np.max(np.abs(tire_base.Fy(Fz, sg * alpha, cam0))) for sg in (1, -1)]
                rolled = [np.max(np.abs(tire_base.Fy(Fz, sg * alpha, cam0 + dcam))) for sg in (1, -1)]
                i = int(np.argmax([base[0] - rolled[0], base[1] - rolled[1]]))
                loss += base[i] - rolled[i]
                total += base[i]
        camber_loss = loss / total                                             # -, fraction of peak Fy

        # (short name, label, where, attribute, factor)
        CASES = [
            ("Raw IMU skidpad target", f"{err.get('skidpad', 0) * 100:+.1f} % muyScale", "tire", "muyScale", 1 + err.get("skidpad", 0)),
            ("Raw IMU brake target",   f"{err.get('brake', 0) * 100:+.1f} % muxScale",   "tire", "muxScale", 1 + err.get("brake", 0)),
            ("Raw IMU accel target",   f"{err.get('accel', 0) * 100:+.1f} % efficiency", "car",  "efficiency_scale", 1 + err.get("accel", 0)),
            ("Roll camber (not in sim)", f"{-camber_loss * 100:+.1f} % muyScale",        "tire", "muyScale", 1 - camber_loss),
        ]
        base_t = {name: run(car_base, tire_base) for name, run in EVENTS.items()}
        print("\nD. EFFECT ON THE LAP SIM (s, + = slower)")
        print(f"   {'':44s}" + "".join(f"{n:>11s}" for n in EVENTS))
        print(f"   {'baseline':44s}" + "".join(f"{base_t[n]:11.3f}" for n in EVENTS))
        for short, label, where, attr, f in CASES:
            car, tire = copy.deepcopy(car_base), copy.deepcopy(tire_base)
            target = tire if where == "tire" else car
            setattr(target, attr, getattr(target, attr) * f)                  # tire scales rebuild the grip table on set
            dt = {n: run(car, tire) - base_t[n] for n, run in EVENTS.items()}
            print(f"   {short + ' (' + label + ')':44s}" + "".join(f"{dt[n]:+11.3f}" for n in EVENTS))
            summary.append((short, f"{label}: autocross {dt['Autocross']:+.2f} s, endurance {dt['Endurance']:+.1f} s",
                            "sim misses it: enter camber gains, or add camber to the sim" if "camber" in short
                            else "avoided by using the corrected IMU"))
    except Exception as e:
        print(f"\nD. skipped, lap sim not loaded ({e}). Set RUN_SIM = False to hide this.")

print(f"\nIMU position used: {R_CG_TO_IMU} m from the CG")

# ── E. summary ────────────────────────────────────────────────
w1 = max(len(r[0]) for r in summary) + 2
w2 = max(len(r[1]) for r in summary) + 2
print("\nE. SUMMARY")
print(f"   {'item':{w1}s}{'number':{w2}s}conclusion")
print("   " + "-" * (w1 + w2 + 55))
for item, number, conclusion in summary:
    print(f"   {item:{w1}s}{number:{w2}s}{conclusion}")

# ── IMU tilt graph ────────────────────────────────────────────
if PLOT:
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.5))
    for log, s in data.items():
        axs[0].plot(s["t"], s["pitch"], lw=1, label=f"{log[:-4]}  {s['pitch'].mean():.2f}")
    axs[0].axhline(all_pitch, color="k", ls="--", lw=2)
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Pitch channel (deg), + nose down")
    axs[0].set_title(f"1. Pitch channel: {all_pitch:.1f} deg in every log", fontweight="bold")

    xb = np.arange(len(tilt_names))
    axs[1].bar(xb - 0.2, tilt_pitch, 0.4, label="Pitch channel mean")
    axs[1].bar(xb + 0.2, tilt_off, 0.4, label="from raw accel - GPS d|v|/dt")
    axs[1].axhline(angle_stop, color="r", ls="--", lw=2, label=f"car stopped, atan(accX/accZ) {angle_stop:.1f}")
    axs[1].set_xticks(xb, tilt_names, fontsize=8)
    axs[1].set_ylabel("Angle (deg), + nose down")
    axs[1].set_title("2. The angle measured three ways", fontweight="bold")

    for log in GRADIENT_LOGS:
        s = data[log]
        m = s["v"] > 5
        idx = np.digitize(s["course"][m], np.arange(-180, 181, 30))
        keep = [i for i in np.unique(idx) if (idx == i).sum() > 20]
        axs[2].plot(s["course"][m], s["pitch"][m], ".", ms=1, alpha=0.2)
        axs[2].plot([s["course"][m][idx == i].mean() for i in keep], [s["pitch"][m][idx == i].mean() for i in keep],
                    "o-", lw=2, label=f"{log[:-4]} (mean per 30 deg)")
    axs[2].set_xlabel("Direction of travel (deg, 0 = East)")
    axs[2].set_ylabel("Pitch channel (deg), + nose down")
    axs[2].set_title("3. Same angle in every direction: not a slope", fontweight="bold")
    axs[2].set_xlim(-180, 180)
    for ax in axs:
        ax.grid(True)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()