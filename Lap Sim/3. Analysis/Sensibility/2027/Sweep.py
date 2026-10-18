import copy
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
from Acceleration import AccelSolver
from SkidPad import SkidPadSolver
from Autocross import solve
from Endurance import solve as endurance_solve

TIRE_DIR = LAPSIM_ROOT / "1. Data" / "Tire"
TRACK_DIR = LAPSIM_ROOT / "1. Data" / "Track"

car_base = MFE26()
tire_base = Tire(TIRE_DIR / "MF61_Parameters.mat", TIRE_DIR / "FZ_Reference.mat")   # correlated defaults in Tire.py
track_autocross = loadTrack(TRACK_DIR, "Michigan2026_Autocross")
track_endurance = loadTrack(TRACK_DIR, "Michigan2026_Endurance")

# event name: function that takes a car and a tire and returns the event time [s]
EVENTS = {
    "Accel":     lambda car, tire: AccelSolver().simulate(tire, car)["lap_time"],
    "Skidpad":   lambda car, tire: SkidPadSolver().simulate(tire, car)["lap_time"],
    "Autocross": lambda car, tire: solve(tire, car, track_autocross)["lap_time"],
    "Endurance": lambda car, tire: endurance_solve(tire, car, track_endurance)["total_time"],
}

# ── changes to test, edit these ───────────────────────────────
# (label, "car" or "tire", attribute, change in %), each one is run alone from the baseline
CHANGES = [
    ("MUX +10 %",                "tire", "muxScale",     +10),   # longitudinal grip
    ("MUY +10 %",                "tire", "muyScale",     +10),   # lateral grip
    ("ClA +5 %",                 "car",  "CLA",           +5),
    ("CdA -5 %",                 "car",  "CDA",           -5),
    ("Mass -5 %",                "car",  "mass_total",    -5),   # total mass with driver, removed from mass_no_driver
    ("CG height -5 %",           "car",  "CG_height",     -5),
    ("Torque +5 %",              "car",  "torque_scale",  +5),
]

def apply_change(car, tire, where, attr, pct):
    # applies one change in % to a copy of the car or tire
    f = 1 + pct / 100
    if attr == "mass_total":
        car.mass_no_driver -= car.mass_total * (1 - f)   # kg, driver and aero mass kept fixed
        return
    target = tire if where == "tire" else car
    setattr(target, attr, getattr(target, attr) * f)    # a tire scale rebuilds its grip table on set

# baseline
base = {name: run_event(car_base, tire_base) for name, run_event in EVENTS.items()}
print("baseline:")
for name, t in base.items():
    print(f"  {name:10s} {t:.3f} s")

# loop over changes, one run per event and change
dt = {name: np.zeros(len(CHANGES)) for name in EVENTS}   # s, change in event time vs baseline
for c, (label, where, attr, pct) in enumerate(CHANGES):
    car = copy.deepcopy(car_base)
    tire = copy.deepcopy(tire_base)
    apply_change(car, tire, where, attr, pct)
    for name, run_event in EVENTS.items():
        dt[name][c] = run_event(car, tire) - base[name]
    print(f"\rsensitivity: {(c + 1) / len(CHANGES) * 100:5.1f} %  ({c + 1}/{len(CHANGES)}, {label})", end="")
print()

# table, negative = faster
labels = [label for label, _, _, _ in CHANGES]
print(f"\nchange in event time vs baseline [s], negative = faster")
print(f"{'':24s}" + "".join(f"{name:>12s}" for name in EVENTS))
for c, label in enumerate(labels):
    print(f"{label:24s}" + "".join(f"{dt[name][c]:>+12.3f}" for name in EVENTS))

# plot: one page per event, biggest gain on top
for name in EVENTS:
    d = dt[name]
    order = np.argsort(d)[::-1]                        # most negative (biggest gain) ends on top
    y = np.arange(len(CHANGES))
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(y, d[order], color=["tab:green" if v < 0 else "tab:red" for v in d[order]])
    for yi, v in zip(y, d[order]):
        ax.text(v, yi, f" {v:+.3f} s ({v / base[name] * 100:+.2f} %) ", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0, color="k")
    ax.set_yticks(y, [labels[i] for i in order])
    ax.set_xlabel("Change in lap time vs baseline (s), negative = faster")
    ax.set_title(f"{name} sensitivity, baseline {base[name]:.3f} s", fontweight="bold")
    ax.margins(x=0.25)                                 # room for the labels
    ax.grid(True, axis="x")
    plt.tight_layout()

plt.show()