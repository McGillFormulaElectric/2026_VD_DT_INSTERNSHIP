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
from CarProperties import MFE26
from Tire import Tire
from SolverFunctions import corner_speed_ceiling
from Acceleration import AccelSolver
from Autocross import solve as autocross_solve
from Endurance import solve as endurance_solve

# setup
TIRE_DIR = LAPSIM_ROOT / "1. Data" / "Tire"
TRACK_DIR = LAPSIM_ROOT / "1. Data" / "Track"

TRACK = "Endurance_Michigan_2024"
RATIOS = np.arange(8, 16.01, 1)
EVENTS = ["accel", "autocross", "endurance"]

car = MFE26()
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

# run
res = sweep(RATIOS)

# plot
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
plt.show()