import sys
from pathlib import Path

# Path to the Lap Sim folder
LAPSIM_ROOT = Path(__file__).resolve().parents[1]   # folder that holds CarProperties.py
sys.path.insert(0, str(LAPSIM_ROOT))

from CarProperties import MFE26
from Tire import Tire
from SkidPad import SkidPadSolver

# target from SkidPadCorrelation.py, skidpad2.mat
AY_TARGET = 1.406   # g
R_TARGET = 9.67     # m, measured path radius
V_TARGET = 41.5     # km/h, only to compare
TOLERANCE = 1.0     # %, stop when sim is this close to target

# car, tire and solver at the measured radius
TIRE_DIR = LAPSIM_ROOT / "1. Data" / "Tire"
car = MFE26()
tire = Tire(TIRE_DIR / "MF61_Parameters.mat", TIRE_DIR / "FZ_Reference.mat")
solver = SkidPadSolver(path_radius=R_TARGET)

# ── scale muy until sim ay matches the test ───────────────────
for i in range(5):
    V_sim = solver.simulate(tire, car)["v"]              # m/s
    ay_sim = V_sim**2 / R_TARGET / car.g                 # g, ay = V^2 / R
    error = (ay_sim / AY_TARGET - 1) * 100               # %, + means sim has more grip than test
    print(f"iter {i}: muyScale {tire.muyScale:.4f}   sim {ay_sim:.3f} g at {V_sim * 3.6:.1f} km/h   error {error:+.2f} %")

    if abs(error) < TOLERANCE:
        print(f"\nDONE: muyScale = {tire.muyScale:.4f}, put this value in Tire.py")
        break

    tire.muyScale *= AY_TARGET / ay_sim                  # scale grip by the ratio test / sim
else:
    print("\nNOT CONVERGED after 5 iterations, check car setup and radius")

print(f"target: {AY_TARGET:.3f} g at {V_TARGET:.1f} km/h")