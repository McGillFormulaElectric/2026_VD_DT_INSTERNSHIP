import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt

# Path to the Lap Sim folder
LAPSIM_ROOT = Path(__file__).resolve().parents[1]   # folder that holds MotecData.py
sys.path.insert(0, str(LAPSIM_ROOT))

from MotecData import MotecData
from TrackMap import loadTrack
from CarProperties import MFE26
from Tire import Tire
from SolverFunctions import corner_speed_ceiling, lap_profile, energy_and_time, ellipse, braking_decel
from Scoring import Scoring
import Physics as ph
import Limits

STANDING_START = False   # True = start from 0 m/s like Autocross.solve, False = flying lap
APEX_WINDOW = 5.0        # m, search window around each apex for the minimum speed
WHEELS = ["FL", "FR", "RL", "RR"]
IMU_FC = 5.0             # Hz, low pass cutoff on the IMU channels
BRAKE_G = 0.4            # g, test IMU decel above this = driver braking
CORNERS = (ph.FL_Fz, ph.FR_Fz, ph.RL_Fz, ph.RR_Fz)

TIRE_DIR = LAPSIM_ROOT / "1. Data" / "Tire"
car = MFE26()
tire = Tire(TIRE_DIR / "MF61_Parameters.mat", TIRE_DIR / "FZ_Reference.mat")   # correlated defaults in Tire.py
checks = []              # (name, value, status, note) for the master checklist at the end

def check(name, value, ok, note="", info=False):
    # add one line to the master checklist
    checks.append((name, value, "INFO" if info else ("PASS" if ok else "WARN"), note))

# ── measured lap ──────────────────────────────────────────────
data = MotecData(LAPSIM_ROOT / "1. Data" / "Motec" / "TorontoShootout2026.mat")
time = data.getTime("velX")                                           # s
velX = data.getValue("velX")                                          # m/s
velY = data.getValue("velY")                                          # m/s
power = data.getValue("power")                                        # kW, pack
torque = sum(data.getValue(f"TorqueActual{w}") for w in WHEELS)       # Nm, sum of 4 motors
dt = np.mean(np.diff(time))                                           # s

V_test = savgol_filter(np.sqrt(velX**2 + velY**2), 31, 2)   # m/s, smoothed over about 0.3 s
ax_test = np.gradient(V_test, dt) / car.g                    # g, + accelerating, from GPS speed
s_test = np.cumsum(V_test) * dt                              # m, distance travelled
s_test = s_test - s_test[0]
t_test = time - time[0]                                      # s

# ── IMU, yaw rate and GPS path on the velX time base ──────────
# units checked on the TorontoShootout file: accX/accY in m/s^2 (unit string empty), gyrZ_HR in rad/s
acc_scale = 1.0 / car.g      # m/s^2 to g
gyr_scale = 1.0              # rad/s

t_acc = data.getTime("accX")                                          # s
t_accy = data.getTime("accY")                                         # s
t_gyr = data.getTime("gyrZ_HR")                                       # s
b, a = butter(2, IMU_FC / (0.5 / np.mean(np.diff(t_acc))))            # cutoff / Nyquist
ax_imu = -np.interp(time, t_acc, filtfilt(b, a, data.getValue("accX"))) * acc_scale   # g, sign flipped: IMU reads negative when accelerating
ay_imu = np.interp(time, t_accy, filtfilt(b, a, data.getValue("accY"))) * acc_scale   # g, includes roll gravity leak

# yaw based: ay = V * yaw rate, reads high when sideslip is changing (slalom, oversteer)
yaw = np.interp(time, t_gyr, data.getValue("gyrZ_HR")) * gyr_scale   # rad/s
ay_test = V_test * yaw / car.g                                        # g

# path based: ay = V * course rate, the true path, no sideslip error
course = np.unwrap(np.arctan2(velY, velX))                            # rad, direction of travel
course_rate = savgol_filter(course, 31, 2, deriv=1, delta=dt)        # rad/s
ay_path = V_test * course_rate / car.g                                # g
k_path = course_rate / np.maximum(V_test, 1.0)                        # 1/m, 1 m/s floor
k_car = yaw / np.maximum(V_test, 1.0)                                 # 1/m

# ── data ──────────────────────────────────────────────────────
span_err = max(abs(t_acc[0] - time[0]), abs(t_acc[-1] - time[-1]), abs(t_gyr[0] - time[0]), abs(t_gyr[-1] - time[-1]))   # s
turn_yaw = np.sum(yaw) * dt / (2 * np.pi)                             # rev
turn_gps = (course[-1] - course[0]) / (2 * np.pi)                     # rev
ax_diff = np.mean(ax_imu - ax_test)                                   # g
print("── data ──")
print(f"duration {t_test[-1]:.2f} s   {1 / dt:.1f} Hz   time base mismatch {span_err * 1000:.0f} ms")
print(f"speed:    min {V_test.min() * 3.6:.1f}   max {V_test.max() * 3.6:.1f}   mean {V_test.mean() * 3.6:.1f} km/h")
print(f"torque:   max {torque.max():.0f} Nm   at zero {np.mean(np.abs(torque) < 1) * 100:.0f} % of lap")
print(f"power:    max {power.max():.1f} kW   time above {car.power_cap / 1000:.0f} kW {np.sum(power > car.power_cap / 1000) * dt:.2f} s")
print(f"ax GPS vs IMU: mean diff {ax_diff:+.3f} g   RMS {np.sqrt(np.mean((ax_imu - ax_test)**2)):.3f} g")
print(f"total turn: yaw {turn_yaw:+.2f} rev   GPS course {turn_gps:+.2f} rev")
print(f"lateral 99th pct:  V*yaw {np.percentile(np.abs(ay_test), 99):.2f} g   "
      f"GPS path {np.percentile(np.abs(ay_path), 99):.2f} g   IMU {np.percentile(np.abs(ay_imu), 99):.2f} g")
check("time bases aligned", f"{span_err * 1000:.0f} ms", span_err < 0.05)
check("closed lap, yaw", f"{turn_yaw:+.2f} rev", abs(abs(turn_yaw) - 1) < 0.03, "not +-1 = file is not exactly one lap")
check("closed lap, GPS course", f"{turn_gps:+.2f} rev", abs(abs(turn_gps) - 1) < 0.03)
check("ax GPS vs IMU offset", f"{ax_diff:+.3f} g", abs(ax_diff) < 0.03, "offset = IMU mount angle or pitch")

# ── car ───────────────────────────────────────────────────────
v_rev = ph.vehicle_speed(car, car.rpm_cap)                                                  # m/s
rpm_test = ph.motor_rpm(car, V_test)                                                        # rpm
P_mech = torque * rpm_test * 2 * np.pi / 60 / 1000                                          # kW, 4 motors
drive = (power > 20) & (torque > 20)                                                        # driving hard
eta_test = np.median(P_mech[drive] / power[drive])
eta_sim = np.median([ph.motor_eff(car, n, T / 4) * car.efficiency_scale * car.inverter_efficiency
                     for n, T in zip(rpm_test[drive], torque[drive])])
print("\n── car ──")
print(f"gear ratio {car.gear_ratio}   tire radius {car.tire_radius} m   wheel_force per Nm {ph.wheel_force(car, 1.0):.1f} N/Nm")
print(f"mass: total {car.mass_total:.1f} kg   effective {car.mass_effective:.1f} kg   ratio {car.mass_effective / car.mass_total:.3f}")
print(f"CDA {car.CDA}   CLA {car.CLA}   tire pressure {car.tire_pressure} psi   CGx {car.CGx}   CG height {car.CG_height} m")
print(f"Crr from rolling_resistance: {ph.rolling_resistance(car, 60 / 3.6, 1.0):.4f} at 60 km/h   "
      f"{ph.rolling_resistance(car, 100 / 3.6, 1.0):.4f} at 100 km/h")
print(f"torque_scale {car.torque_scale}   torque_cap {car.torque_cap} Nm   power_cap {car.power_cap / 1000:.1f} kW")
print(f"efficiency_scale {car.efficiency_scale}   inverter_efficiency {car.inverter_efficiency}")
print(f"brake_bias_front {car.brake_bias_front}   max_decel {car.max_decel / car.g:.2f} g")
print(f"rev limit: {car.rpm_cap} rpm = {v_rev * 3.6:.1f} km/h   test max {V_test.max() * 3.6:.1f} km/h")
print(f"drivetrain eta (mech / pack):  test {eta_test:.2f}   sim {eta_sim:.2f}   ({np.sum(drive)} pts)")

# tyre mu at static load: drive, brake, lateral
Fz_static = [f(car, 0.0, 0.0, 0.0) for f in CORNERS]                                        # N
mu_drv = np.mean([tire.peak_drive(f) / f for f in Fz_static])
mu_brk = np.mean([tire.peak(f)[1] / f for f in Fz_static])
mu_lat = np.mean([tire.peak(f)[2] / f for f in Fz_static])
print(f"tyre mu at static load: drive {mu_drv:.2f}   brake {mu_brk:.2f}   lateral {mu_lat:.2f}   "
      f"(Fz {Fz_static[0]:.0f} N front, {Fz_static[2]:.0f} N rear)")
check("drivetrain eta, test vs sim", f"{eta_test:.2f} vs {eta_sim:.2f}", abs(eta_test - eta_sim) < 0.03)
check("top speed within rev limit", f"{V_test.max() * 3.6:.1f} vs {v_rev * 3.6:.1f} km/h", V_test.max() <= 1.01 * v_rev,
      "above = rpm_cap or tire radius off")
check("test power within power_cap", f"{np.sum(power > car.power_cap / 1000) * dt:.2f} s above", np.sum(power > car.power_cap / 1000) == 0,
      "car exceeded the cap, sim cannot follow there")
check("mu brake / mu lateral", f"{mu_brk / mu_lat:.2f}", mu_brk / mu_lat > 0.9, "tyres usually give mu x close to mu y, low = muxScale low")
check("mu drive vs mu brake", f"{mu_drv:.2f} vs {mu_brk:.2f}", abs(mu_drv - mu_brk) < 0.05)

# ── track built from this same file ───────────────────────────
track = loadTrack(LAPSIM_ROOT / "1. Data" / "Track", "TorontoShootout2026")
dist_ratio = s_test[-1] / track.lap_length
print("\n── track ──")
print(f"track: {track.lap_length:.1f} m, {len(track.apex)} apexes, tightest R = {1 / np.max(np.abs(track.k)):.1f} m")
print(f"test distance {s_test[-1]:.1f} m   map length {track.lap_length:.1f} m   ratio {dist_ratio:.3f}")
check("test distance / map length", f"{dist_ratio:.3f}", abs(dist_ratio - 1) < 0.01)

# ── sim ───────────────────────────────────────────────────────
v_max = np.asarray(corner_speed_ceiling(track, car, tire))   # m/s, grip limit at each point
torque_profile = np.interp(track.s, s_test, torque)          # Nm, measured torque on the track grid
print(f"\n── sim ──")
print(f"grip limit v_max: min {v_max.min() * 3.6:.1f}   max {v_max.max() * 3.6:.1f} km/h")

def run_lap(torque_profile=None):
    # same steps as Autocross.solve, but with a torque input and a start choice
    v, v_fwd, _ = lap_profile(track, car, tire, v_max, standing_start=STANDING_START, torque_profile=torque_profile)
    t, P_pack, energy_J = energy_and_time(track, car, tire, v, v_max)
    lap_time = t[-1] + track.ds / max(v[-1], 0.1)   # s
    # returns speed [m/s], time [s], pack power [kW], lap time [s], energy [Wh], forward pass speed [m/s]
    return np.asarray(v), np.asarray(t), np.asarray(P_pack) / 1000, lap_time, energy_J / 3600, np.asarray(v_fwd)

v_free, t_free, P_free, lap_free, E_free, v_fwd_free = run_lap()           # full torque, pro driver
v_sim, t_sim, P_sim, lap_sim, E_sim, v_fwd_sim = run_lap(torque_profile)    # measured torque, compared to the test below
s_sim = np.asarray(track.s)                                                 # m
ax_sim = v_sim * np.gradient(v_sim, s_sim) / car.g                          # g, a = v dv/ds
ax_free = v_free * np.gradient(v_free, s_sim) / car.g                       # g
on_fwd = np.abs(v_sim - v_fwd_sim) < 0.01                                   # sim accelerating or coasting, not braking
on_fwd_free = np.abs(v_free - v_fwd_free) < 0.01

# sim ay = V^2 * k, sign matched to the test so both turn the same way
sgn = 1.0 if np.sum(np.interp(s_sim, s_test, ay_path) * track.k) > 0 else -1.0
ay_sim = sgn * v_sim**2 * np.asarray(track.k) / car.g                       # g
ay_free = sgn * v_free**2 * np.asarray(track.k) / car.g                     # g

print(f"full torque:      {lap_free:.2f} s   {Scoring.getAutocrossScore(lap_free):.1f} pts")
print(f"measured torque:  {lap_sim:.2f} s   {Scoring.getAutocrossScore(lap_sim):.1f} pts")
print(f"cost of throttle + wheelspin: {lap_sim - lap_free:.2f} s")
check("full torque lap <= measured torque lap", f"{lap_free:.2f} vs {lap_sim:.2f} s", lap_free <= lap_sim + 0.01)

# ── inside the sim: what limits the forward pass ──────────────
# rebuilt here from the final sim speed with the same Physics / Limits calls forward_pass uses
LIMITS = ["measured torque", "power cap", "motor torque cap", "grip", "ellipse (cornering)", "v_max", "rev limit"]
n = len(s_sim)
F_meas = np.zeros(n); F_cap = np.zeros(n); F_used = np.zeros(n)            # N
F_drag = np.zeros(n); F_roll = np.zeros(n); F_corner = np.zeros(n)         # N
P_cap = np.zeros(n)                                                         # W, pack power at the cap
lim = np.full(n, -1)                                                        # index into LIMITS
ax_ms2 = ax_sim * car.g                                                     # m/s^2
for i in range(n - 1):
    vi = max(v_sim[i], 0.1)                                                 # m/s
    DF = ph.downforce(car, vi)                                              # N
    ay = vi**2 * abs(track.k[i])                                            # m/s^2
    util = (vi / max(v_max[i], 1e-9))**2                                    # -, lateral grip used
    ax_prev = ax_ms2[i - 1] if i > 0 else 0.0                               # m/s^2, lagged one step like forward_pass
    Fz = [max(f(car, DF, ay, ax_prev), 0.0) for f in CORNERS]               # N
    r = Limits.tractive_force_4wd(tire, car, *Fz, vi)
    e = ellipse(vi, v_max[i]) if v_max[i] < 0.999 * v_rev else 1.0         # -, same guard as forward_pass
    F_cap[i] = r["Fx_total"] * e                                            # N, what the car could give
    F_meas[i] = ph.wheel_force(car, torque_profile[i])                      # N, measured torque
    F_used[i] = min(F_cap[i], F_meas[i])                                    # N, what forward_pass uses
    F_drag[i] = ph.drag(car, vi)
    F_roll[i] = ph.rolling_resistance(car, vi, car.mass_total * car.g + DF)
    F_corner[i] = ph.cornering_drag(car, ay, util, tire.slip_peak)
    P_cap[i] = r["PackPower"]
    if v_sim[i] >= 0.99 * v_rev:
        lim[i] = 6                                                          # rev limit
    elif v_sim[i] >= 0.99 * v_max[i]:
        lim[i] = 5                                                          # v_max
    elif F_meas[i] <= F_cap[i]:
        lim[i] = 0                                                          # measured torque
    elif e < 0.99:
        lim[i] = 4                                                          # friction ellipse
    elif r["PackPower"] >= 0.99 * car.power_cap:
        lim[i] = 1                                                          # power cap
    elif max(r["Tmotor"]) >= 0.99 * car.torque_cap * car.torque_scale:
        lim[i] = 2                                                          # motor torque cap
    else:
        lim[i] = 3                                                          # tyre grip

print("\n── sim limits, measured torque run (share of lap) ──")
for code, name in enumerate(LIMITS):
    print(f"  {name:<22} {np.mean((lim == code) & on_fwd) * 100:5.1f} %")
print(f"  {'braking':<22} {np.mean(~on_fwd) * 100:5.1f} %")

# forces on straights where the sim is driving
m = (torque_profile > 20) & on_fwd & (np.abs(ay_sim) < 0.3) & (lim <= 4)
ax_force = (F_used - F_drag - F_roll - F_corner) / car.mass_effective / car.g   # g, from the forces
recon_err = np.mean(ax_force[m]) - np.mean(ax_sim[m])                            # g
print(f"\nsim forces on straights ({np.sum(m)} pts), mean:")
print(f"  measured torque force {np.mean(F_meas[m]):7.0f} N")
print(f"  available (cap)       {np.mean(F_cap[m]):7.0f} N")
print(f"  used (min of both)    {np.mean(F_used[m]):7.0f} N")
print(f"  drag                  {np.mean(F_drag[m]):7.0f} N")
print(f"  rolling               {np.mean(F_roll[m]):7.0f} N")
print(f"  cornering drag        {np.mean(F_corner[m]):7.0f} N")
print(f"  ax from forces        {np.mean(ax_force[m]):7.3f} g   sim profile {np.mean(ax_sim[m]):.3f} g   "
      f"test GPS on its straights {np.mean(ax_test[(torque > 20) & (np.abs(ay_path) < 0.3)]):.3f} g")
print(f"  pack power at the cap {np.mean(P_cap[m]) / 1000:7.1f} kW   (power_cap {car.power_cap / 1000:.1f} kW)")
check("reconstruction matches solver", f"{recon_err:+.3f} g", abs(recon_err) < 0.02, "off = check script out of sync with SolverFunctions")

# ── braking vs accel capability on a straight, by speed ───────
# mu = peak force / Fz per wheel, bias use = braking force / sum of the 4 tyre peaks (1.00 = all 4 tyres at the limit)
print("\n── braking vs accel capability, straight line, full torque ──")
print(f"{'V':>4} {'brake':>6} {'accel':>6} {'bias':>5} {'Fz F':>5} {'Fz R':>5} {'F share':>7} {'capped':>6}")
print(f"{'m/s':>4} {'[g]':>6} {'[g]':>6} {'use':>5} {'[N]':>5} {'[N]':>5} {'[-]':>7} {'':>6}")
bias_use = {}
for V in [10, 14, 18, 22, 26, 29]:                                           # m/s
    DF = ph.downforce(car, V)                                                # N
    F_res = ph.drag(car, V) + ph.rolling_resistance(car, V, car.mass_total * car.g + DF)   # N
    dec = 0.0                                                                # m/s^2
    for _ in range(5):                                                       # converge load transfer
        dec = braking_decel(car, tire, V, 1e3, k=0.0, ax_prev=-dec)
    Fz_b = [max(f(car, DF, 0.0, -dec), 0.0) for f in CORNERS]                # N
    use = (dec * car.mass_effective - F_res) / sum(tire.peak(f)[1] for f in Fz_b)   # -
    acc = 0.0                                                                # m/s^2
    for _ in range(5):
        Fz_a = [max(f(car, DF, 0.0, acc), 0.0) for f in CORNERS]             # N
        r = Limits.tractive_force_4wd(tire, car, *Fz_a, V)
        acc = (r["Fx_total"] - F_res) / car.mass_effective
    bias_use[V] = use
    capped = "yes" if dec >= 0.999 * car.max_decel else ""
    print(f"{V:>4} {dec / car.g:>6.2f} {acc / car.g:>6.2f} {use:>5.2f} {Fz_b[0]:>5.0f} {Fz_b[2]:>5.0f} "
          f"{Fz_b[0] / (Fz_b[0] + Fz_b[2]):>7.2f} {capped:>6}")
print("  F share = ideal brake_bias_front at that speed, 'capped' = max_decel is the limit, not the tyres")
check("brake bias use at 10 m/s", f"{bias_use[10]:.2f}", bias_use[10] > 0.95, "low = brake_bias_front far from the front load share")

# ── sim vs test ───────────────────────────────────────────────
delta = np.interp(s_test, s_sim, t_sim - t_sim[0]) - t_test   # s, negative = sim ahead
E_test = np.sum(power) * dt / 3.6                              # Wh, kW x s / 3.6
print("\n── sim vs test ──")
print(f"lap time:  sim {lap_sim:.2f} s   test {t_test[-1]:.2f} s   gap {(lap_sim / t_test[-1] - 1) * 100:+.1f} %   (full torque {lap_free:.2f} s)")
print(f"power:     sim max {np.max(P_sim):.1f} kW   full torque max {np.max(P_free):.1f} kW   test max {np.max(power):.1f} kW")
print(f"energy:    sim {E_sim:.0f} Wh   full torque {E_free:.0f} Wh   test {E_test:.0f} Wh   ratio sim/test {E_sim / E_test:.2f}")
print(f"lateral:   sim {np.percentile(np.abs(ay_sim), 99):.2f} g   full torque {np.percentile(np.abs(ay_free), 99):.2f} g   "
      f"test GPS path {np.percentile(np.abs(ay_path), 99):.2f} g   (99th pct)")
check("energy sim / test", f"{E_sim / E_test:.2f}", 0.9 < E_sim / E_test < 1.1, "test also had > 80 kW and braking differs, regen not modelled")

# ── ax: the full picture ──────────────────────────────────────
print("\n── ax ──")
p = np.percentile
print(f"{'':<22} {'test GPS':>8} {'test IMU':>8} {'sim meas':>8} {'sim full':>8}")
print(f"{'accel 99th pct [g]':<22} {p(ax_test, 99):>8.2f} {p(ax_imu, 99):>8.2f} {p(ax_sim, 99):>8.2f} {p(ax_free, 99):>8.2f}")
print(f"{'accel 90th pct [g]':<22} {p(ax_test, 90):>8.2f} {p(ax_imu, 90):>8.2f} {p(ax_sim, 90):>8.2f} {p(ax_free, 90):>8.2f}")
print(f"{'braking 99th pct [g]':<22} {-p(ax_test, 1):>8.2f} {-p(ax_imu, 1):>8.2f} {-p(ax_sim, 1):>8.2f} {-p(ax_free, 1):>8.2f}")
print(f"{'braking 90th pct [g]':<22} {-p(ax_test, 10):>8.2f} {-p(ax_imu, 10):>8.2f} {-p(ax_sim, 10):>8.2f} {-p(ax_free, 10):>8.2f}")

# share of lap TIME spent accelerating, coasting, braking (sim weighted by time per step)
dt_sim = track.ds / np.maximum(v_sim, 0.1)                                   # s per step
dt_free = track.ds / np.maximum(v_free, 0.1)
def phases(ax, w):
    # share of time [%] with ax > +0.1 g, between, and below -0.3 g
    w = w / np.sum(w)
    return np.sum(w[ax > 0.1]) * 100, np.sum(w[(ax <= 0.1) & (ax >= -0.3)]) * 100, np.sum(w[ax < -0.3]) * 100
ph_t = phases(ax_imu, np.ones_like(ax_imu))
ph_s = phases(ax_sim, dt_sim)
ph_f = phases(ax_free, dt_free)
print(f"\nshare of lap time        accel   coast   brake")
print(f"  test (IMU)             {ph_t[0]:5.1f}%  {ph_t[1]:5.1f}%  {ph_t[2]:5.1f}%")
print(f"  sim, measured torque   {ph_s[0]:5.1f}%  {ph_s[1]:5.1f}%  {ph_s[2]:5.1f}%")
print(f"  sim, full torque       {ph_f[0]:5.1f}%  {ph_f[1]:5.1f}%  {ph_f[2]:5.1f}%")
print("  sim measured coasts where the driver brakes: it gets his torque, not his brakes")

# accel on straights by speed: test vs sim, driving points only
print(f"\naccel on straights by speed, driving only (|ay| < 0.3 g)")
print(f"{'V [m/s]':>8} {'test':>6} {'sim meas':>8} {'sim full':>8} {'meas err':>8} {'test kW':>7} {'sim kW':>6}")
st_t = (torque > 20) & (np.abs(ay_path) < 0.3)
st_s = (torque_profile > 20) & on_fwd & (np.abs(ay_sim) < 0.3) & (v_sim < 0.98 * v_rev)
st_f = on_fwd_free & (np.abs(ay_free) < 0.3) & (v_free < 0.98 * v_max) & (v_free < 0.98 * v_rev)
for lo, hi in [(10, 14), (14, 18), (18, 21), (21, 24), (24, 27), (27, 29.5)]:
    mt = st_t & (V_test >= lo) & (V_test < hi)
    ms = st_s & (v_sim >= lo) & (v_sim < hi)
    mf = st_f & (v_free >= lo) & (v_free < hi)
    if mt.sum() < 5 or ms.sum() < 5:
        continue
    a_t, a_s = np.mean(ax_test[mt]), np.mean(ax_sim[ms])
    a_f = np.mean(ax_free[mf]) if mf.sum() >= 5 else np.nan
    print(f"{lo:>3}-{hi:<4} {a_t:>6.2f} {a_s:>8.2f} {a_f:>8.2f} {(a_s / a_t - 1) * 100:>+7.1f}% "
          f"{np.mean(power[mt]):>7.1f} {np.mean(P_sim[ms]):>6.1f}")

# accel per torque on straights, same torque should give the same accel
straight_s = st_s & (v_sim < 0.98 * v_max)
gain_t = np.median(ax_test[st_t] / torque[st_t]) * 1000                     # mg per Nm
gain_s = np.median(ax_sim[straight_s] / torque_profile[straight_s]) * 1000  # mg per Nm
n_pow = np.sum(straight_s & (P_cap >= 0.99 * car.power_cap))
print(f"\naccel per torque on straights: test {gain_t:.2f} mg/Nm   sim {gain_s:.2f} mg/Nm   sim/test {gain_s / gain_t:.2f}   "
      f"({np.sum(st_t)} test pts, {np.sum(straight_s)} sim pts, {n_pow} at the power cap)")
check("accel per torque sim / test", f"{gain_s / gain_t:.2f}", 0.95 < gain_s / gain_t < 1.05,
      f"{n_pow} sim pts power capped, test went over the cap")

# coasting: no torque and not braking, should be drag + rolling + cornering drag only
coast = (torque_profile < 1) & on_fwd
if coast.sum() > 5:
    ax_coast_expect = -(F_drag + F_roll + F_corner) / car.mass_effective / car.g   # g
    print(f"sim coasting ({coast.sum()} pts): ax {np.mean(ax_sim[coast]):.2f} g   expected from resistances {np.mean(ax_coast_expect[coast]):.2f} g")
    check("sim coast decel = resistances", f"{np.mean(ax_sim[coast]):.2f} vs {np.mean(ax_coast_expect[coast]):.2f} g",
          abs(np.mean(ax_sim[coast]) - np.mean(ax_coast_expect[coast])) < 0.03)

# braking zones: where the driver braked vs where the sim brakes
zones = []
brk_t = ax_imu < -BRAKE_G
i = 0
while i < len(brk_t):
    if brk_t[i]:
        j = i
        while j + 1 < len(brk_t) and brk_t[j + 1]:
            j += 1
        if t_test[j] - t_test[i] >= 0.2:                                     # s, ignore blips
            zones.append((i, j))
        i = j + 1
    else:
        i += 1
print(f"\nbraking zones (test IMU below -{BRAKE_G} g for 0.2 s or more): {len(zones)}")
print(f"{'s test':>6} {'V in':>6} {'V out':>6} {'test pk':>7} {'sim pk':>6} {'full pk':>7} {'sim later':>9} {'time won':>8}")
print(f"{'[m]':>6} {'[km/h]':>6} {'[km/h]':>6} {'[g]':>7} {'[g]':>6} {'[g]':>7} {'[m]':>9} {'[s]':>8}")
later_all, peak_ratio = [], []
for i, j in zones:
    s0, s1 = s_test[i], s_test[j]                                            # m
    w = (s_sim >= s0 - 15) & (s_sim <= s1 + 10)                              # window around the zone
    brk_s = w & (ax_sim < -BRAKE_G)
    pk_t = -np.min(ax_imu[i:j + 1])                                          # g
    pk_s = -np.min(ax_sim[w])                                                # g
    pk_f = -np.min(ax_free[w])                                               # g
    later = s_sim[np.argmax(brk_s)] - s0 if brk_s.any() else np.nan         # m, + = sim brakes later
    won = (np.interp(s0 - 15, s_test, delta) - np.interp(s1 + 10, s_test, delta))   # s, + = sim gained time here
    later_all.append(later)
    peak_ratio.append(pk_s / pk_t)
    print(f"{s0:>6.0f} {V_test[i] * 3.6:>6.1f} {V_test[j] * 3.6:>6.1f} {pk_t:>7.2f} {pk_s:>6.2f} {pk_f:>7.2f} {later:>+9.1f} {won:>+8.2f}")
if zones:
    print(f"  sim brakes on average {np.nanmean(later_all):+.1f} m later, peak sim / test {np.mean(peak_ratio):.2f}")
    check("sim braking peak / test peak", f"{np.mean(peak_ratio):.2f}", np.mean(peak_ratio) >= 1.0,
          "below 1 = sim brakes softer than the driver, check muxScale and bias", info=np.mean(peak_ratio) >= 1.0)

# max_decel cap and numerical spikes
n_cap = np.sum(ax_free <= -0.999 * car.max_decel / car.g)
jumps = np.abs(np.diff(ax_sim))
spikes = np.where(jumps > 0.3)[0]
print(f"\nmax_decel binding on {n_cap} pts of the full torque lap")
print(f"ax steps above 0.3 g between grid points: {len(spikes)}   at s = {', '.join(f'{s_sim[k]:.0f}' for k in spikes[:8])} m")
print("  a few are normal, QSS switches from throttle to brakes in one step")
check("max_decel not binding", f"{n_cap} pts", n_cap == 0, "tyres should be the limit, raise max_decel", info=n_cap == 0)
check("ax step jumps", f"{len(spikes)}", True, "one per braking zone is QSS, many more = noise in k or v_max", info=True)

# ── apex table ────────────────────────────────────────────────
# ds = test minimum minus map apex, 'edge' = minimum at the window edge, row less reliable
print(f"\n{'apex':>4} {'s':>5} {'ds':>5} {'R map':>6} {'R car':>6} {'test':>6} {'sim':>6} {'full':>6} {'err':>6} "
      f"{'ay path':>7} {'ay sim':>6}")
print(f"{'':>4} {'[m]':>5} {'[m]':>5} {'[m]':>6} {'[m]':>6} {'[km/h]':>6} {'[km/h]':>6} {'[km/h]':>6} {'[%]':>6} "
      f"{'[g]':>7} {'[g]':>6}")
good_path, good_sim, good_R = [], [], []
for n_a, i in enumerate(track.apex):
    s_apex = track.s[i]                                                       # m
    R_map = 1 / abs(track.k[i])                                               # m
    idx = np.where(np.abs(s_test - s_apex) < APEX_WINDOW)[0]
    j = idx[np.argmin(V_test[idx])]                                           # test minimum speed near the apex
    ds = s_test[j] - s_apex                                                   # m
    edge = abs(ds) > APEX_WINDOW - 0.5
    wn = np.abs(s_sim - s_apex) < APEX_WINDOW
    v_a_test = V_test[j]                                                      # m/s
    v_a_sim = np.min(v_sim[wn])                                               # m/s
    v_a_free = np.min(v_free[wn])                                             # m/s
    R_car = 1 / np.max(np.abs(k_path[idx]))                                   # m, tightest in the window
    ay_p = np.max(np.abs(ay_path[idx]))                                       # g
    ay_s = v_a_sim**2 / R_map / car.g                                         # g
    if not edge:
        good_path.append(ay_p)
        good_sim.append(ay_s)
        good_R.append(R_map / R_car)
    flag = "  edge" if edge else ""
    print(f"{n_a + 1:>4} {s_apex:>5.0f} {ds:>+5.1f} {R_map:>6.1f} {R_car:>6.1f} {v_a_test * 3.6:>6.1f} "
          f"{v_a_sim * 3.6:>6.1f} {v_a_free * 3.6:>6.1f} {(v_a_sim / v_a_test - 1) * 100:>+6.1f} {ay_p:>7.2f} {ay_s:>6.2f}{flag}")

good_path, good_sim, good_R = np.array(good_path), np.array(good_sim), np.array(good_R)
print(f"\nreliable apexes (not at window edge): {len(good_path)} of {len(track.apex)}")
print(f"  R map / R car mean {good_R.mean():.2f}   (1.00 = map matches the car's path)")
print(f"  ay path mean {good_path.mean():.2f} g   ay sim mean {good_sim.mean():.2f} g   ratio {good_path.mean() / good_sim.mean():.2f}")
check("map radius / car radius", f"{good_R.mean():.2f}", 0.95 < good_R.mean() < 1.05, "off = rebuild track with curvature='car'")
check("apex lateral test / sim", f"{good_path.mean() / good_sim.mean():.2f}", 0.95 < good_path.mean() / good_sim.mean() < 1.05,
      "above 1 = sim lateral grip low (muyScale), below 1 = sim corners faster than the car did")

# ── master checklist ──────────────────────────────────────────
print("\n══ master check ══════════════════════════════════════════════")
for name, value, status, note in checks:
    tail = f"   <- {note}" if status == "WARN" and note else ""
    print(f"  [{status}] {name:<36} {value:>22}{tail}")
n_warn = sum(c[2] == "WARN" for c in checks)
print(f"  {len(checks) - n_warn} ok, {n_warn} to look at")
print("══════════════════════════════════════════════════════════════")

# ── plots vs distance ─────────────────────────────────────────
fig, axs = plt.subplots(4, 1, sharex=True, figsize=(12, 11))

axs[0].plot(s_test, V_test * 3.6, label="test")
axs[0].plot(s_sim, v_sim * 3.6, label="sim, measured torque")
axs[0].plot(s_sim, v_free * 3.6, "--", label="sim, full torque")
axs[0].plot(s_sim, v_max * 3.6, ":", color="gray", label="grip limit")
axs[0].set_ylim(0, 1.2 * np.max(v_free) * 3.6)
axs[0].set_ylabel("speed [km/h]")
axs[0].legend()

axs[1].plot(s_test, delta)
axs[1].axhline(0, color="k", linewidth=0.8)
axs[1].set_ylabel("sim - test [s]\n(down = sim ahead)")

axs[2].plot(s_test, ax_test, label="test GPS")
axs[2].plot(s_test, ax_imu, label=f"test IMU {IMU_FC:.0f} Hz", alpha=0.7)
axs[2].plot(s_sim, ax_sim, label="sim, measured torque")
axs[2].plot(s_sim, ax_free, "--", color="tab:red", linewidth=1.0, label="sim, full torque")
axs[2].axhline(-car.max_decel / car.g, color="gray", linestyle=":", linewidth=0.8)   # g, max_decel cap
axs[2].set_ylabel("ax [g]")
axs[2].legend(loc="lower right", fontsize=8)

axs[3].plot(s_test, power, label="test")
axs[3].plot(s_sim, P_sim, label="sim, measured torque")
axs[3].plot(s_sim, P_free, "--", label="sim, full torque")
axs[3].axhline(car.power_cap / 1000, color="r", linestyle="--")   # kW, sim power cap
axs[3].set_ylabel("pack power [kW]")
axs[3].set_xlabel("distance [m]")
axs[3].legend()

for ax in axs:
    ax.grid(True)
    for i in track.apex:
        ax.axvline(track.s[i], color="k", alpha=0.1)   # apex markers
plt.tight_layout()

# ── g-g, lateral g and curvature ──────────────────────────────
fig2, (ax_gg, ax_ay, ax_k) = plt.subplots(1, 3, figsize=(19, 6))

ax_gg.scatter(ay_path, ax_imu, s=3, c="gray", alpha=0.4, label="test (GPS path ay, IMU ax)")
ax_gg.scatter(ay_sim, ax_sim, s=3, c="tab:green", alpha=0.6, label="sim, measured torque")
ax_gg.scatter(ay_free, ax_free, s=3, c="tab:red", alpha=0.4, label="sim, full torque")
ax_gg.set_xlabel("ay [g]")
ax_gg.set_ylabel("ax [g]")
ax_gg.set_title("g-g")
ax_gg.set_aspect("equal")
ax_gg.grid(True)
ax_gg.legend()

# lateral g three ways, gaps between them show sideslip transients or roll leak
ax_ay.plot(s_test, np.abs(ay_path), label="GPS path")
ax_ay.plot(s_test, np.abs(ay_test), label="V*yaw", alpha=0.7)
ax_ay.plot(s_test, np.abs(ay_imu), label="IMU", alpha=0.5)
ax_ay.plot(s_sim, np.abs(ay_sim), label="sim", color="tab:red")
ax_ay.set_xlabel("distance [m]")
ax_ay.set_ylabel("|ay| [g]")
ax_ay.set_title("lateral accel")
ax_ay.grid(True)
ax_ay.legend()

ax_k.plot(s_sim, track.k, label="map")
ax_k.plot(s_test, sgn * k_path, label="car path (course rate / V)", alpha=0.8)
ax_k.plot(s_test, sgn * k_car, label="car yaw (yaw rate / V)", alpha=0.5)
ax_k.plot(s_sim[track.apex], np.asarray(track.k)[track.apex], "kx", label="apex")
ax_k.set_xlabel("distance [m]")
ax_k.set_ylabel("k [1/m]")
ax_k.set_title("curvature")
ax_k.grid(True)
ax_k.legend()
plt.tight_layout()

plt.show()