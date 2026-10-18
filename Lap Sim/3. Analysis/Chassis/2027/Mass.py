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
tire = Tire(TIRE_DIR / "MF61_Parameters.mat", TIRE_DIR / "FZ_Reference.mat")   # correlated defaults in Tire.py
track_autocross = loadTrack(TRACK_DIR, "Michigan2026_Autocross")
track_endurance = loadTrack(TRACK_DIR, "Michigan2026_Endurance")

# event name: function that takes a car and returns the event time [s]
EVENTS = {
    "Accel":     lambda car: AccelSolver().simulate(tire, car)["lap_time"],
    "Skidpad":   lambda car: SkidPadSolver().simulate(tire, car)["lap_time"],
    "Autocross": lambda car: solve(tire, car, track_autocross)["lap_time"],
    "Endurance": lambda car: endurance_solve(tire, car, track_endurance)["total_time"],
}

MASS_LIST = car_base.mass_total + np.arange(-70, 40.01, 1)   # kg, total mass with driver, -70 to +40 kg around baseline, 1 kg steps

fig, axs = plt.subplots(2, 2, figsize=(12, 9))

for ax, (name, run_event) in zip(axs.flat, EVENTS.items()):

    # loop over mass
    T = np.zeros(len(MASS_LIST))   # event time [s]
    for i, m in enumerate(MASS_LIST):
        car = copy.deepcopy(car_base)
        car.mass_no_driver = m - car.mass_driver   # kg, driver mass kept fixed
        T[i] = run_event(car)
        print(f"\r{name}: {(i + 1) / len(MASS_LIST) * 100:5.1f} %  ({i + 1}/{len(MASS_LIST)})", end="")
    print()   # new line when the event is done

    slope = np.polyfit(MASS_LIST, T, 1)[0]   # s/kg

    # plot
    ax.plot(MASS_LIST, T, "o-")
    ax.axvline(car_base.mass_total, color="r", ls="--", label=f"baseline {car_base.mass_total:.0f} kg")
    ax.set_xlabel("Total mass (kg)")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"{name} Time (s)   {slope * 10:.3f} s per 10 kg", fontweight="bold")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()