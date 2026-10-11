import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Path to the Lap Sim folder
LAPSIM_ROOT = Path(__file__).resolve().parents[3]   # folder that holds CarProperties.py
sys.path.insert(0, str(LAPSIM_ROOT))

from TrackMap import loadTrack
from CarProperties import MFE27
from Tire import Tire
from SolverFunctions import corner_speed_ceiling
from Acceleration import AccelSolver
from Autocross import solve as autocross_solve
from Endurance import solve as endurance_solve

# setup
TIRE_DIR = LAPSIM_ROOT / "1. Data" / "Tire"
TRACK_DIR = LAPSIM_ROOT / "1. Data" / "Track"

TRACK = "Endurance_Michigan_2024"
RATIOS = np.arange(6, 18.01, 0.25)
SENS_RATIOS = np.arange(7, 17.01, 0.5)
VARY = 0.10   # +/- 10 %
EVENTS = ["accel", "autocross", "endurance"]

PARAMS = [
    ("total mass",         "mass_total"),
    ("effective mass",     "mass_effective"),
    ("torque scale",       "torque_scale"),
    ("motor efficiency",   "efficiency_scale"),
    ("tyre radius",        "tire_radius"),
    ("CG height",          "CG_height"),      
    ("torque split",       "torque_split"),
    ("CdA",                "CDA"),
    ("ClA",                "CLA"),
    ("aero balance",       "aero_balance"),
    ("long. grip",         "muxScale"),
    ("lat. grip",          "muyScale"),
    ("CGX",                "CGx"),
]
TIRE_ATTRS = ("muxScale", "muyScale")

car = MFE27()
tire = Tire(TIRE_DIR / "MF61_Parameters.mat", TIRE_DIR / "FZ_Reference.mat")
track = loadTrack(TRACK_DIR, TRACK)
baseline = car.gear_ratio

def run_events():
    v_max = corner_speed_ceiling(track, car, tire)   # depends on the ratio, recompute
    en = endurance_solve(tire, car, track, v_max=v_max)
    return {"accel": AccelSolver().simulate(tire, car)["lap_time"],                     # s
            "autocross": autocross_solve(tire, car, track, v_max=v_max)["lap_time"],    # s
            "endurance": en["total_time"],                                               # s
            "energy": en["total_energy_kWh"]}                                            # kWh

def sweep(ratios):
    res = {key: [] for key in EVENTS + ["energy"]}
    for g in ratios:
        car.gear_ratio = g
        for key, val in run_events().items():
            res[key].append(val)
    car.gear_ratio = baseline
    return {key: np.asarray(val) for key, val in res.items()}

def set_value(target, attr, value):
    # read-only properties (like mass_total) are replaced on the class for the run, returns an undo function
    cls = type(target)
    prop = getattr(cls, attr, None)
    if isinstance(prop, property) and prop.fset is None:
        had = attr in cls.__dict__
        old = cls.__dict__.get(attr)
        setattr(cls, attr, property(lambda self: value))
        return lambda: setattr(cls, attr, old) if had else delattr(cls, attr)
    old = getattr(target, attr)
    setattr(target, attr, value)
    return lambda: setattr(target, attr, old)

def best(ratios, y):
    # (ratio, value) at the minimum, parabola through the 3 points around it
    i = int(np.argmin(y))
    if i == 0 or i == len(ratios) - 1:
        return ratios[i], y[i]
    a, b, c = np.polyfit(ratios[i-1:i+2], y[i-1:i+2], 2)
    if a <= 0:
        return ratios[i], y[i]
    return -b / (2 * a), c - b**2 / (4 * a)

# run
res = sweep(RATIOS)

ref = sweep(SENS_RATIOS)
ref_best = {e: best(SENS_RATIOS, ref[e]) for e in EVENTS}
d_ratio = {e: np.zeros((len(PARAMS), 2)) for e in EVENTS}   # -, shift in best ratio, columns -10 % / +10 %
d_time = {e: np.zeros((len(PARAMS), 2)) for e in EVENTS}    # s, change in time at the best ratio

for p, (label, attr) in enumerate(PARAMS):
    target = tire if attr in TIRE_ATTRS else car
    old = getattr(target, attr)
    for j, factor in enumerate((1 - VARY, 1 + VARY)):
        undo = set_value(target, attr, old * factor if np.isscalar(old) else np.asarray(old) * factor)
        r = sweep(SENS_RATIOS)
        undo()
        for e in EVENTS:
            g, t = best(SENS_RATIOS, r[e])
            d_ratio[e][p, j] = g - ref_best[e][0]
            d_time[e][p, j] = t - ref_best[e][1]
    print(f"{label} done")

# plots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
plots = [(axs[0, 0], "energy", "Endurance Energy (kWh)"),
         (axs[0, 1], "endurance", "Endurance Time (s)"),
         (axs[1, 0], "autocross", "Autocross Time (s)"),
         (axs[1, 1], "accel", "Accel Time (s)")]

for ax, key, title in plots:
    y = res[key]
    i = np.argmin(y)
    ax.plot(RATIOS, y, linewidth=2)
    ax.plot(RATIOS[i], y[i], "ro", label=f"best {RATIOS[i]:.2f}")
    ax.axvline(baseline, color="k", linestyle="--", label=f"current {baseline:.2f}")
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Gear Ratio (-)")
    ax.grid(True)
    ax.legend()
plt.tight_layout()

names = [label for label, _ in PARAMS]
y = np.arange(len(PARAMS))
for e in EVENTS:
    order = np.argsort(np.abs(d_ratio[e][:, 1] - d_ratio[e][:, 0]))   # biggest ratio shift on top
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, d, xlabel in [(a1, d_ratio[e], f"Change in best gear ratio from {ref_best[e][0]:.2f} (-)"),
                          (a2, d_time[e], f"Change in best time from {ref_best[e][1]:.2f} s (s)")]:
        ax.barh(y - 0.2, d[order, 0], height=0.4, label="-10 %")
        ax.barh(y + 0.2, d[order, 1], height=0.4, label="+10 %")
        ax.axvline(0, color="k")
        ax.set_xlabel(xlabel)
        ax.grid(True, axis="x")
        ax.legend()
    a1.set_yticks(y, [names[i] for i in order])
    fig.suptitle(f"{e.capitalize()} Sensitivity", fontweight="bold")
    plt.tight_layout()

plt.show()