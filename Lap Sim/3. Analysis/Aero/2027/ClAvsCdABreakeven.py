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
track = loadTrack(TRACK_DIR, "Michigan2026_Autocross")   # only autocross needs a track

# event name: function that takes a car and returns the event time [s]
EVENTS = {
    "Accel":     lambda car: AccelSolver().simulate(tire, car)["lap_time"],
    "Skidpad":   lambda car: SkidPadSolver().simulate(tire, car)["lap_time"],
    "Autocross": lambda car: solve(tire, car, track)["lap_time"],
}

#Sweep parameters
CLA_START, CLA_STOP, CLA_STEP = 3.8, 4.8, 0.1   # m^2, ClA range and step
CDA_START, CDA_STOP, CDA_STEP = 1.0, 2.0, 0.1   # m^2, CdA range and step

CLA_LIST = np.arange(CLA_START, CLA_STOP + CLA_STEP / 2, CLA_STEP)   # m^2
CDA_LIST = np.arange(CDA_START, CDA_STOP + CDA_STEP / 2, CDA_STEP)   # m^2
print(f"baseline: ClA {car_base.CLA:.2f} m^2, CdA {car_base.CDA:.2f} m^2")
print(f"{len(CLA_LIST)} ClA points x {len(CDA_LIST)} CdA points = {len(CLA_LIST) * len(CDA_LIST)} runs per event")

fig, axs = plt.subplots(1, len(EVENTS), figsize=(17, 5))

for ax, (name, run_event) in zip(axs, EVENTS.items()):

    # loop over ClA and CdA
    T = np.zeros((len(CLA_LIST), len(CDA_LIST)))   # event time [s], rows = ClA, cols = CdA
    n_total = len(CLA_LIST) * len(CDA_LIST)
    n = 0
    for i, cla in enumerate(CLA_LIST):
        for j, cda in enumerate(CDA_LIST):
            car = copy.deepcopy(car_base)
            car.CLA = cla
            car.CDA = cda
            T[i, j] = run_event(car)
            n += 1
            print(f"\r{name}: {n / n_total * 100:5.1f} %  ({n}/{n_total})", end="")
        print()   # new line when the event is done

    # plot
    pc = ax.pcolormesh(CDA_LIST, CLA_LIST, T, shading="nearest", cmap="viridis", edgecolors="k", linewidth=0.5)
    fig.colorbar(pc, ax=ax)
    ax.plot(car_base.CDA, car_base.CLA, "ro", ms=6)   # baseline car
    ax.set_xlabel("CdA (m^2)")
    ax.set_ylabel("ClA (m^2)")
    ax.set_title(f"{name} Time (s)", fontweight="bold")

plt.tight_layout()
plt.show()