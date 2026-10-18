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
from Autocross import solve

TIRE_DIR = LAPSIM_ROOT / "1. Data" / "Tire"
TRACK_DIR = LAPSIM_ROOT / "1. Data" / "Track"

car_base = MFE26()
tire = Tire(TIRE_DIR / "MF61_Parameters.mat", TIRE_DIR / "FZ_Reference.mat")   # correlated defaults in Tire.py
track = loadTrack(TRACK_DIR, "Michigan2026_Autocross")

# Sweep parameters
CLA_START, CLA_STOP, CLA_STEP = 2.2, 4.5, 0.1     # m^2, ClA range and step
CDA_START, CDA_STOP, CDA_STEP = 1.0, 1.95, 0.05   # m^2, CdA range and step

CLA_LIST = np.arange(CLA_START, CLA_STOP + CLA_STEP / 2, CLA_STEP)   # m^2
CDA_LIST = np.arange(CDA_START, CDA_STOP + CDA_STEP / 2, CDA_STEP)   # m^2
print(f"baseline: ClA {car_base.CLA:.2f} m^2, CdA {car_base.CDA:.2f} m^2")
print(f"{len(CLA_LIST)} ClA points x {len(CDA_LIST)} CdA points = {len(CLA_LIST) * len(CDA_LIST)} runs")

# LAptime on grid
T = np.zeros((len(CLA_LIST), len(CDA_LIST)))   # lap time [s], rows = ClA, cols = CdA
n_total = T.size
n = 0
for i, cla in enumerate(CLA_LIST):
    for j, cda in enumerate(CDA_LIST):
        car = copy.deepcopy(car_base)
        car.CLA = cla
        car.CDA = cda
        T[i, j] = solve(tire, car, track)["lap_time"]
        n += 1
        print(f"\rAutocross: {n / n_total * 100:5.1f} %  ({n}/{n_total})", end="")
print()

# slopes and break even
dt_dCLA = np.gradient(T, CLA_LIST, axis=0)   # s per m^2 ClA, negative (more downforce = faster)
dt_dCDA = np.gradient(T, CDA_LIST, axis=1)   # s per m^2 CdA, positive (more drag = slower)
BE = -dt_dCLA / dt_dCDA                      # m^2 CdA per m^2 ClA

# value at the baseline car (nearest grid cell)
i0 = np.argmin(np.abs(CLA_LIST - car_base.CLA))
j0 = np.argmin(np.abs(CDA_LIST - car_base.CDA))
print(f"\nat ClA {CLA_LIST[i0]:.2f}, CdA {CDA_LIST[j0]:.2f}:")
print(f"  dt/dClA  {dt_dCLA[i0, j0] * 1000:+.0f} ms per m^2")
print(f"  dt/dCdA  {dt_dCDA[i0, j0] * 1000:+.0f} ms per m^2")
print(f"  break-even {BE[i0, j0]:.2f}: +0.1 ClA is worth up to +{BE[i0, j0] * 0.1:.2f} CdA")

# plot
fig, ax = plt.subplots(figsize=(9, 7))
pc = ax.pcolormesh(CDA_LIST, CLA_LIST, BE, shading="nearest", cmap="viridis", edgecolors="k", linewidth=0.5)
fig.colorbar(pc, ax=ax)
ax.plot(car_base.CDA, car_base.CLA, "ro", ms=6)   # baseline car
ax.invert_yaxis()                                 # more downforce at the bottom, same as the MATLAB plot with negative CL
ax.set_xlabel("CdA (m^2)")
ax.set_ylabel("ClA (m^2)")
ax.set_title("Aero Break-even (-dt/dClA / dt/dCdA)", fontweight="bold")
plt.tight_layout()
plt.show()