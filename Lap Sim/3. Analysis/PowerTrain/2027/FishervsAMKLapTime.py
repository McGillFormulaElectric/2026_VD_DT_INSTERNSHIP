import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Path to the Lap Sim folder
LAPSIM_ROOT = Path(__file__).resolve().parents[3]   # folder that holds CarProperties.py, change the number if this file moves
sys.path.insert(0, str(LAPSIM_ROOT))

from TrackMap import loadTrack
from CarProperties import MFE26, MFE27
from Tire import Tire
from Acceleration import AccelSolver
from SkidPad import SkidPadSolver
from Autocross import solve
from Endurance import solve as endurance_solve

TIRE_DIR = LAPSIM_ROOT / "1. Data" / "Tire"
TRACK_DIR = LAPSIM_ROOT / "1. Data" / "Track"

tire = Tire(TIRE_DIR / "MF61_Parameters.mat", TIRE_DIR / "FZ_Reference.mat")   # correlated defaults in Tire.py
track_autocross = loadTrack(TRACK_DIR, "Michigan2026_Autocross")
track_endurance = loadTrack(TRACK_DIR, "Michigan2026_Endurance")

# car name: (car, plot colour)
CARS = {
    "MFE26 AMK":    (MFE26(), "tab:blue"),
    "MFE27 Fisher": (MFE27(), "tab:orange"),
}

# event name: (function that takes a car and returns the result dict, key for the event time [s])
EVENTS = {
    "Accel":     (lambda car: AccelSolver().simulate(tire, car),               "lap_time"),
    "Skidpad":   (lambda car: SkidPadSolver().simulate(tire, car),             "lap_time"),
    "Autocross": (lambda car: solve(tire, car, track_autocross),               "lap_time"),
    "Endurance": (lambda car: endurance_solve(tire, car, track_endurance),     "total_time"),
}

# run every event for every car
results = {}   # (car name, event) -> result dict
for car_name, (car, _) in CARS.items():
    for ev, (run, _) in EVENTS.items():
        results[(car_name, ev)] = run(car)
        print(f"\r{car_name}: {ev} done   ", end="")
print()

# table
names = list(CARS)
print(f"\n{'event':<10}" + "".join(f"{n:>15}" for n in names) + f"{'difference':>13}")
for ev, (_, key) in EVENTS.items():
    t = [results[(n, ev)][key] for n in names]   # s
    print(f"{ev:<10}" + "".join(f"{x:>13.3f} s" for x in t) + f"{t[1] - t[0]:>+11.3f} s")

# plots: speed vs distance, both cars overlaid, event time in the legend
fig, axs = plt.subplots(2, 2, figsize=(14, 9))
for ax, (ev, (_, key)) in zip(axs.flat, EVENTS.items()):
    times = []
    trace = all("s" in results[(n, ev)] and "v" in results[(n, ev)] for n in CARS)   # speed trace returned by the solver
    for car_name, (_, colour) in CARS.items():
        r = results[(car_name, ev)]
        t = r[key]                                                   # s
        times.append(t)
        if trace:
            ax.plot(np.asarray(r["s"]), np.asarray(r["v"]) * 3.6, color=colour, label=f"{car_name}  {t:.3f} s")
        else:
            ax.bar(car_name, t, color=colour, label=f"{car_name}  {t:.3f} s")   # no speed trace, show the time
    ax.set_title(f"{ev}: {times[1] - times[0]:+.3f} s ({(times[1] / times[0] - 1) * 100:+.1f} %)", fontweight="bold")
    if trace:
        ax.set_xlabel("distance [m]")
        ax.set_ylabel("speed [km/h]")
    else:
        ax.set_ylabel("time [s]")
        ax.set_ylim(0.9 * min(times), 1.05 * max(times))             # zoom so the difference is visible
    ax.grid(True)
    ax.legend()
fig.suptitle("MFE26 (AMK) vs MFE27 (Fisher)", fontweight="bold")
plt.tight_layout()
plt.show()