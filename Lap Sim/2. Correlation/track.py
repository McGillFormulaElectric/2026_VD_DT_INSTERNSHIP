# Author: Anne-Sophie
# Summary: Load both tracks, run the autocross event, and draw each track as an
#          x-y graph coloured by what limits the car at each point:
#          TIRE limited (grip: cornering, braking, low-speed exit) vs
#          POWER/TORQUE limited (motor envelope on straights above crossover).
#          Same template + imports as GearSweep.
import sys, pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

CORE = pathlib.Path(__file__).resolve().parents[1]   # 2027 -> PowerTrain -> Analysis -> Lap Sim
sys.path.insert(0, str(CORE))                         # (match this to where you save the file)

from CarProperties import MFE27
from Tire import Tire
from TrackMap import loadTrack
from Autocross import solve as autocross_solve        # <-- FIX module name if yours differs

DATA      = CORE / "Data"
TRACK_DIR = DATA / "Track"

TIRE_COLOR  = "#1f77b4"    # grip limited
POWER_COLOR = "#d62728"    # power / torque limited


# ----------------------------------------------------------------------------
# pull the few physics numbers off your car/tire; fall back if a name differs
# ----------------------------------------------------------------------------
def phys(car, tire):
    g   = 9.81
    m   = getattr(car,  "mass",      getattr(car, "m", 300.0))
    P   = getattr(car,  "power_max", getattr(car, "P_max", 80_000.0))
    vmx = getattr(car,  "v_max",     30.0)
    cda = getattr(car,  "CdA",       getattr(car, "cda", 1.10))
    mu  = getattr(tire, "mu_peak",   getattr(tire, "mu", None))
    grip_g = float(mu) if mu else 1.40
    p = dict(g=g, m=m, P=P, vmax=vmx, cda=cda, rho=1.20, grip_g=grip_g)
    print("  physics used:", {k: round(v, 3) for k, v in p.items()})
    return p


# ----------------------------------------------------------------------------
# QSS speed profile (used only if the solver doesn't hand back a per-node trace)
# ----------------------------------------------------------------------------
def _seg(s, lap_length):
    seg = np.empty_like(s, dtype=float)
    seg[:-1] = np.diff(s)
    seg[-1]  = max(lap_length - s[-1], np.median(np.diff(s)))
    return seg


def qss_profile(track, p):
    s, k = np.asarray(track.s, float), np.asarray(track.k, float)
    lap  = float(getattr(track, "lap_length", s[-1]))
    n, seg, kabs = len(s), _seg(s, lap), np.abs(k)
    a_lat_max = p["grip_g"] * p["g"]

    v_corner = np.minimum(np.sqrt(a_lat_max / np.maximum(kabs, 1e-9)), p["vmax"])
    v = v_corner.copy()

    for _ in range(2):                                   # forward accel, wrapped
        for i in range(n):
            j = (i + 1) % n
            a_lat  = v[i] ** 2 * kabs[i]
            a_grip = a_lat_max * np.sqrt(max(0.0, 1.0 - min(1.0, a_lat / a_lat_max) ** 2))
            a_pow  = p["P"] / (p["m"] * max(v[i], 1.0))
            a_drag = 0.5 * p["rho"] * p["cda"] * v[i] ** 2 / p["m"]
            a_x    = min(a_grip, a_pow) - a_drag
            vt = np.sqrt(max(0.0, v[i] ** 2 + 2.0 * a_x * seg[i]))
            if vt < v[j]:
                v[j] = vt

    for _ in range(2):                                   # backward braking, wrapped
        for i in range(n - 1, -1, -1):
            j = (i - 1) % n
            a_lat   = v[i] ** 2 * kabs[i]
            a_brake = a_lat_max * np.sqrt(max(0.0, 1.0 - min(1.0, a_lat / a_lat_max) ** 2))
            vt = np.sqrt(max(0.0, v[i] ** 2 + 2.0 * a_brake * seg[j]))
            if vt < v[j]:
                v[j] = vt
    return v


def classify(track, v, p):
    """0 = tyre/grip limited, 1 = power/torque limited, read off the profile."""
    s, k = np.asarray(track.s, float), np.asarray(track.k, float)
    lap  = float(getattr(track, "lap_length", s[-1]))
    n, seg, kabs = len(s), _seg(s, lap), np.abs(k)
    a_lat_max = p["grip_g"] * p["g"]
    v_corner  = np.minimum(np.sqrt(a_lat_max / np.maximum(kabs, 1e-9)), p["vmax"])
    lab = np.zeros(n, dtype=int)
    for i in range(n):
        j = (i + 1) % n
        if v[i] >= v_corner[i] * 0.999:          # cornering cap -> lateral grip
            continue
        if (v[j] - v[i]) / seg[i] < -1e-4:        # slowing -> braking -> grip
            continue
        a_lat  = v[i] ** 2 * kabs[i]              # accelerating -> lower ceiling wins
        a_grip = a_lat_max * np.sqrt(max(0.0, 1.0 - min(1.0, a_lat / a_lat_max) ** 2))
        a_pow  = p["P"] / (p["m"] * max(v[i], 1.0))
        lab[i] = 1 if a_pow < a_grip else 0
    return lab


def get_v(track, res, p):
    """Prefer the solver's own per-node speed; fall back to the QSS above."""
    if isinstance(res, dict):
        for key in ("v_profile", "v", "speed", "vx"):
            arr = res.get(key)
            if arr is not None and len(np.ravel(arr)) == len(track.s):
                return np.asarray(arr, float).ravel()
    return qss_profile(track, p)


# ----------------------------------------------------------------------------
# plot
# ----------------------------------------------------------------------------
def plot_limits(track, v, lab, title, out):
    x, y = np.asarray(track.x, float), np.asarray(track.y, float)
    pts  = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    cols = np.where(lab[:-1] == 1, POWER_COLOR, TIRE_COLOR)
    frac = 100.0 * lab.mean()

    fig, (ax, axv) = plt.subplots(2, 1, figsize=(11, 11),
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=3.5))
    ax.set_xlim(x.min() - 5, x.max() + 5); ax.set_ylim(y.min() - 5, y.max() + 5)
    ax.set_aspect("equal"); ax.grid(alpha=0.25)
    ax.set_title(f"{title}\n{frac:.0f}% of the lap is power/torque limited")
    ax.plot([], [], color=TIRE_COLOR,  lw=3.5, label="tyre / grip limited")
    ax.plot([], [], color=POWER_COLOR, lw=3.5, label="power / torque limited")
    ax.legend(loc="upper right")

    d = np.arange(len(v))
    vpts = np.array([d, v * 3.6]).T.reshape(-1, 1, 2)
    axv.add_collection(LineCollection(np.concatenate([vpts[:-1], vpts[1:]], axis=1),
                                      colors=cols, linewidths=2.0))
    axv.set_xlim(0, len(v)); axv.set_ylim(0, v.max() * 3.6 * 1.1)
    axv.set_xlabel("node index along lap"); axv.set_ylabel("speed [km/h]")
    axv.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"  {title}: {frac:.1f}% power-limited  ->  {out}")
    return fig


# ----------------------------------------------------------------------------
def main():
    car  = MFE27()
    tire = Tire(str(DATA / "Tire" / "MF61_Coefficients.csv"))
    p    = phys(car, tire)

    # load BOTH tracks
    track_ax = loadTrack(TRACK_DIR, "autocross")
    track_en = loadTrack(TRACK_DIR, "Endurance_Michigan_2024")
    print(f"loaded: {track_ax.event} ({len(track_ax.s)} pts) | "
          f"{track_en.event} ({len(track_en.s)} pts)")

    # run autocross
    ax_res = autocross_solve(track_ax, car, tire)
    
    # limit map per track (uses solver speed if the result carries one, else QSS)
    v_ax = get_v(track_ax, ax_res, p)
    plot_limits(track_ax, v_ax, classify(track_ax, v_ax, p),
                "autocross -- limit map", CORE / "limits_autocross.png")

    v_en = get_v(track_en, None, p)          # endurance not run here -> QSS map
    plot_limits(track_en, v_en, classify(track_en, v_en, p),
                "endurance -- limit map", CORE / "limits_endurance.png")

    plt.show()


if __name__ == "__main__":
    main()