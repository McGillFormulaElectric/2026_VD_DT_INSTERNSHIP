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

TIRE_DIR = LAPSIM_ROOT / "1. Data" / "Tire"
TRACK_DIR = LAPSIM_ROOT / "1. Data" / "Track"

car_base = MFE26()
tire = Tire(TIRE_DIR / "MF61_Parameters.mat", TIRE_DIR / "FZ_Reference.mat")   # correlated defaults in Tire.py
track = loadTrack(TRACK_DIR, "Michigan2026_Endurance")   # only autocross needs a track

# event name: function that takes a car and returns the event time [s]
EVENTS = {
    "Accel":     lambda car: AccelSolver().simulate(tire, car)["lap_time"],
    "Skidpad":   lambda car: SkidPadSolver().simulate(tire, car)["lap_time"],
    "Autocross": lambda car: solve(tire, car, track)["lap_time"],
}

# Sweep parameters
CLA_START, CLA_STOP, CLA_STEP = 3.8, 4.8, 0.1        # m^2, ClA range and step
MASS_START, MASS_STOP, MASS_STEP = 12.0, 20.0, 1.0    # kg, AERO package mass, range and step

CLA_LIST = np.arange(CLA_START, CLA_STOP + CLA_STEP / 2, CLA_STEP)       # m^2
MASS_LIST = np.arange(MASS_START, MASS_STOP + MASS_STEP / 2, MASS_STEP)  # kg
print(f"baseline: ClA {car_base.CLA:.2f} m^2, aero mass {car_base.mass_aero:.1f} kg, total mass {car_base.mass_total:.1f} kg")
print(f"{len(CLA_LIST)} ClA points x {len(MASS_LIST)} mass points = {len(CLA_LIST) * len(MASS_LIST)} runs per event")

fig, axs = plt.subplots(1, len(EVENTS), figsize=(17, 5))

for ax, (name, run_event) in zip(axs, EVENTS.items()):

    # loop over ClA and aero mass
    T = np.zeros((len(CLA_LIST), len(MASS_LIST)))   # event time [s], rows = ClA, cols = aero mass
    n_total = len(CLA_LIST) * len(MASS_LIST)
    n = 0
    for i, cla in enumerate(CLA_LIST):
        for j, mass in enumerate(MASS_LIST):
            car = copy.deepcopy(car_base)
            car.CLA = cla
            car.mass_aero = mass                           # kg, aero package mass, mass_total and mass_effective follow, CG position unchanged
            T[i, j] = run_event(car)
            n += 1
            print(f"\r{name}: {n / n_total * 100:5.1f} %  ({n}/{n_total})", end="")
        print()   # new line when the event is done

    # plot
    pc = ax.pcolormesh(MASS_LIST, CLA_LIST, T, shading="nearest", cmap="viridis", edgecolors="k", linewidth=0.5)
    fig.colorbar(pc, ax=ax)
    ax.plot(car_base.mass_aero, car_base.CLA, "ro", ms=6)   # baseline car
    ax.set_xlabel("Aero mass (kg)")
    ax.set_ylabel("ClA (m^2)")
    ax.set_title(f"{name} Time (s)", fontweight="bold")

plt.tight_layout()
plt.show()