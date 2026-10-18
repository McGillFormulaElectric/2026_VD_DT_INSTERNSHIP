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

AUTOCROSS_TRACK = "Endurance_Michigan_2024"            # name of the .npz in 1. Data/Track, used for autocross
ENDURANCE_TRACK = "Endurance_Michigan_2024"  # name of the .npz in 1. Data/Track, used for endurance
RATIOS = np.arange(2, 18.01, 0.1)   # gear ratios to sweep

car = MFE27()
tire = Tire(TIRE_DIR / "MF61_Parameters.mat", TIRE_DIR / "FZ_Reference.mat")
track_ax = loadTrack(TRACK_DIR, AUTOCROSS_TRACK)
track_en = loadTrack(TRACK_DIR, ENDURANCE_TRACK)
baseline = car.gear_ratio

# run
KEYS = ["accel", "autocross", "endurance", "accel_pts", "autocross_pts", "endurance_pts", "efficiency_pts"]
res = {key: [] for key in KEYS}
for n, g in enumerate(RATIOS):
    car.gear_ratio = g
    v_max_ax = corner_speed_ceiling(track_ax, car, tire)   # m/s, depends on the ratio through the rev limit, recompute
    v_max_en = corner_speed_ceiling(track_en, car, tire)   # m/s, one per track
    acc = AccelSolver().simulate(tire, car)
    ax_run = autocross_solve(tire, car, track_ax, v_max=v_max_ax)
    en = endurance_solve(tire, car, track_en, v_max=v_max_en)
    res["accel"].append(acc["lap_time"])               # s
    res["autocross"].append(ax_run["lap_time"])        # s
    res["endurance"].append(en["total_time"])          # s
    res["accel_pts"].append(acc["comp_point"])
    res["autocross_pts"].append(ax_run["comp_point"])
    res["endurance_pts"].append(en["endurance_score"])
    res["efficiency_pts"].append(en["efficiency_score"])
    print(f"\rsweep: {(n + 1) / len(RATIOS) * 100:5.1f} %  (ratio {g:.2f})", end="")
print()
car.gear_ratio = baseline
res = {key: np.asarray(val) for key, val in res.items()}

def plot_page(plots, best_is_min, suptitle, nrows, ncols, figsize):
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    for ax, (key, title) in zip(np.atleast_1d(axs).flat, plots):
        y = res[key]
        i = np.argmin(y) if best_is_min else np.argmax(y)
        ax.plot(RATIOS, y, linewidth=2)
        ax.plot(RATIOS[i], y[i], "ro", label=f"best {RATIOS[i]:.2f}")
        ax.axvline(baseline, color="k", linestyle="--", label=f"current {baseline:.2f}")
        ax.ticklabel_format(axis="y", useOffset=False)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Gear Ratio (-)")
        ax.grid(True)
        ax.legend()
    fig.suptitle(suptitle, fontweight="bold")
    plt.tight_layout()

# page 1: event times, best = lowest
plot_page([("accel", "Accel Time (s)"),
           ("autocross", "Autocross Time (s)"),
           ("endurance", "Endurance Time (s)")],
          best_is_min=True, suptitle=f"Event time vs gear ratio, autocross {AUTOCROSS_TRACK}, endurance {ENDURANCE_TRACK}",
          nrows=1, ncols=3, figsize=(17, 5))

# page 2: competition points, best = highest
plot_page([("accel_pts", "Accel Points (-)"),
           ("autocross_pts", "Autocross Points (-)"),
           ("endurance_pts", "Endurance Points (-)"),
           ("efficiency_pts", "Efficiency Points (-)")],
          best_is_min=False, suptitle=f"Competition points vs gear ratio, autocross {AUTOCROSS_TRACK}, endurance {ENDURANCE_TRACK}",
          nrows=2, ncols=2, figsize=(12, 8))

plt.show()