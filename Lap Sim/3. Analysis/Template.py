import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Path to the Lap Sim folder
LAPSIM_ROOT = Path(__file__).resolve().parent.parent   # this file sits in the Lap Sim folder
sys.path.insert(0, str(LAPSIM_ROOT))

from TrackMap import loadTrack
from CarProperties import MFE26
from Tire import Tire
from Acceleration import AccelSolver
from SkidPad import SkidPadSolver
from Autocross import solve as autocross_solve
from Endurance import solve as endurance_solve

# setup
TIRE_DIR = LAPSIM_ROOT / "1. Data" / "Tire"
TRACK_DIR = LAPSIM_ROOT / "1. Data" / "Track"

car = MFE26()
tire = Tire(TIRE_DIR / "MF61_Parameters.mat", TIRE_DIR / "FZ_Reference.mat")
track_ax = loadTrack(TRACK_DIR, "Michigan2026_Autocross")
track_en = loadTrack(TRACK_DIR, "Michigan2026_Endurance")

# run
acc = AccelSolver().simulate(tire, car)
skid = SkidPadSolver().simulate(tire, car)
ax = autocross_solve(tire, car, track_ax)
en = endurance_solve(tire, car, track_en)

# print
total = acc["comp_point"] + skid["comp_point"] + ax["comp_point"] + en["endurance_score"] + en["efficiency_score"]
print(f"{'event':<12} {'lap time':>10} {'points':>8}")
print(f"{'accel':<12} {acc['lap_time']:>9.3f}s {acc['comp_point']:>8.1f}")
print(f"{'skidpad':<12} {skid['lap_time']:>9.3f}s {skid['comp_point']:>8.1f}")
print(f"{'autocross':<12} {ax['lap_time']:>9.3f}s {ax['comp_point']:>8.1f}")
print(f"{'endurance':<12} {en['flying_lap_time']:>9.3f}s {en['endurance_score']:>8.1f}   "
      f"({en['n_laps']} laps, total {en['total_time']:.1f} s)")
print(f"{'efficiency':<12} {en['total_energy_kWh']:>8.2f}kWh {en['efficiency_score']:>8.1f}")
print(f"{'total':<12} {'':>10} {total:>8.1f}")

# speed and accel vs time, accel = dv/dt
runs = [("accel", acc["t"], acc["v"]),
        ("skidpad", np.array([0.0, skid["lap_time"]]), np.array([skid["v"], skid["v"]])),   # constant speed
        ("autocross", ax["t"], ax["v"]),
        ("endurance lap", en["t"], en["v"])]

fig, (ax_v, ax_a) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
for name, t, v in runs:
    ax_v.plot(t, v * 3.6, label=name)                         # km/h
    ax_a.plot(t, np.gradient(v, t) / car.g, label=name)       # g
ax_v.set_ylabel("speed [km/h]")
ax_a.set_ylabel("ax [g]")
ax_a.set_xlabel("time [s]")
for a in (ax_v, ax_a):
    a.grid(True)
    a.legend()
plt.tight_layout()
plt.show()