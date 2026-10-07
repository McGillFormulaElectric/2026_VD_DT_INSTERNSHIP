import sys
from pathlib import Path
import numpy as np

# Path to the Lap Sim folder, so we can import MotecData.py
LAPSIM_ROOT = Path(__file__).resolve().parents[1]   # folder that holds MotecData.py
sys.path.insert(0, str(LAPSIM_ROOT))

from CarProperties import MFE26
from Tire import Tire
from SolverFunctions import braking_decel

# target from BrakeCorrelation.py, brake.mat, 8 to 14 m/s window, all 5 checks GOOD
# low speed on purpose: drag is ~0.04 g here, so the sim CdA error moves the result ~1 %
DECEL_TARGET = 0.928   # g
V_TARGET = 11.0        # m/s, average speed in the window

# tyre fit files
TIRE_DIR = LAPSIM_ROOT / "1. Data" / "Tire"

# car and tire, muyScale already correlated on skidpad
car = MFE26()
tire = Tire(TIRE_DIR / "MF61_Parameters.mat", TIRE_DIR / "FZ_Reference.mat", muyScale=0.59)

# scale mux until sim braking decel is within 1 % of target
for i in range(5):
    # decel moves load to the front, which changes decel, so loop until it settles
    dec = 0.0
    for j in range(10):
        dec = braking_decel(car, tire, V_TARGET, 1e3, k=0.0, ax_prev=-dec)   # 1e3 = no corner speed limit, straight line
    dec_sim = dec / car.g
    error = (dec_sim - DECEL_TARGET) / DECEL_TARGET * 100

    print(f"iter {i}: muxScale = {tire.muxScale:.4f}  decel_sim = {dec_sim:.3f} g  error = {error:+.2f} %")

    if np.isclose(dec, car.max_decel):
        print(f"\nSTOP: sim is capped by car.max_decel = {car.max_decel / car.g:.3f} g, raise it in CarProperties")
        break

    if abs(error) < 1:
        print(f"\nDONE: muxScale = {tire.muxScale:.4f}, put this value in Tire.py")
        break

    tire.muxScale = tire.muxScale * DECEL_TARGET / dec_sim
else:
    print("\nNOT CONVERGED after 5 iterations, check car setup")

print(f"target:  decel = {DECEL_TARGET:.3f} g at V = {V_TARGET:.1f} m/s")

# where is the brake peak in the tyre model, and was the test past it
kappa = np.linspace(0, -0.3, 300)
Fx = np.abs(tire.Fx(1400, kappa))       # 1400 N, roughly a front wheel under braking
print(f"peak slip = {kappa[np.argmax(Fx)] * 100:.1f} %")
print(f"Fx at -15 % / peak = {np.abs(tire.Fx(1400, -0.15)) / Fx.max():.3f}")

# brake.mat, 0.928 g at 11 m/s, corrected for -15 % slip past -5.7 % peak