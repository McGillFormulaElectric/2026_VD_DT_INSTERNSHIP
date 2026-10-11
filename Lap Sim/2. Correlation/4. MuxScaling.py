import sys
from pathlib import Path
import numpy as np

# Path to the Lap Sim folder
LAPSIM_ROOT = Path(__file__).resolve().parents[1]   # folder that holds CarProperties.py
sys.path.insert(0, str(LAPSIM_ROOT))

from CarProperties import MFE26
from Tire import Tire
from SolverFunctions import braking_decel

# target from BrakeCorrelation.py, brake.mat, 8 to 14 m/s window, all 5 checks GOOD
# low speed on purpose, drag is about 0.04 g here so a CdA error moves the result about 1 %
DECEL_TARGET = 0.928   # g
V_TARGET = 11.0        # m/s, average speed in the window
TOLERANCE = 1.0        # %, stop when sim is this close to target
FZ_FRONT = 1400        # N, roughly a front wheel under braking
TEST_SLIP = -0.15      # -, median slip in the brake test

# car and tire, muyScale already correlated on skidpad
TIRE_DIR = LAPSIM_ROOT / "1. Data" / "Tire"
car = MFE26()
tire = Tire(TIRE_DIR / "MF61_Parameters.mat", TIRE_DIR / "FZ_Reference.mat")

def sim_decel():
    # braking moves load to the front, which changes decel, so repeat until it settles
    dec = 0.0   # m/s^2
    for _ in range(10):
        dec = braking_decel(car, tire, V_TARGET, 1e3, k=0.0, ax_prev=-dec)   # 1e3 m/s = no corner limit, k = 0 straight line
    return dec   # m/s^2

# ── scale mux until sim decel matches the test ────────────────
for i in range(5):
    dec = sim_decel()                                    # m/s^2
    dec_g = dec / car.g                                  # g
    error = (dec_g / DECEL_TARGET - 1) * 100             # %, + means sim brakes harder than test
    print(f"iter {i}: muxScale {tire.muxScale:.4f}   sim {dec_g:.3f} g   error {error:+.2f} %")

    if np.isclose(dec, car.max_decel):
        print(f"\nSTOP: sim is capped by car.max_decel = {car.max_decel / car.g:.3f} g, raise it in CarProperties")
        break
    if abs(error) < TOLERANCE:
        print(f"\nDONE: muxScale = {tire.muxScale:.4f}, put this value in Tire.py")
        break

    tire.muxScale *= DECEL_TARGET / dec_g                # scale grip by the ratio test / sim
else:
    print("\nNOT CONVERGED after 5 iterations, check car setup")

print(f"target: {DECEL_TARGET:.3f} g at {V_TARGET:.1f} m/s")

# ── was the test past the tyre model's brake peak ─────────────
kappa = np.linspace(0, -0.3, 300)                        # -, slip ratio
Fx = np.abs(tire.Fx(FZ_FRONT, kappa))                    # N
Fx_test = np.abs(tire.Fx(FZ_FRONT, TEST_SLIP))           # N
print(f"peak slip = {kappa[np.argmax(Fx)] * 100:.1f} %")
print(f"Fx at {TEST_SLIP * 100:.0f} % / peak = {Fx_test / Fx.max():.3f}")   # below 1 = test was past the peak