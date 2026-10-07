import sys
from pathlib import Path

# Path to the Lap Sim folder, so we can import MotecData.py
LAPSIM_ROOT = Path(__file__).resolve().parents[1]   # folder that holds MotecData.py
sys.path.insert(0, str(LAPSIM_ROOT))

from CarProperties import MFE26
from Tire import Tire
from SkidPad import SkidPadSolver

# target from SkidPadCorrelation.py, skidpad1 second half
AY_TARGET = 1.347   # g
R_TARGET = 9.66     # m
V_TARGET = 40.6     # km/h, only to compare

# tyre fit files
TIRE_DIR = LAPSIM_ROOT / "1. Data" / "Tire"

# car, tire and solver at the measured radius
car = MFE26()
tire = Tire(TIRE_DIR / "MF61_Parameters.mat", TIRE_DIR / "FZ_Reference.mat")
solver = SkidPadSolver(path_radius=R_TARGET)

# scale muy until sim ay is within 1 % of target
for i in range(5):
    out = solver.simulate(tire, car)
    V_sim = out["v"]                            # m/s
    ay_sim = V_sim**2 / R_TARGET / car.g        # g
    error = (ay_sim - AY_TARGET) / AY_TARGET * 100

    print(f"iter {i}: muyScale = {tire.muyScale:.4f}  ay_sim = {ay_sim:.3f} g  "
          f"V_sim = {V_sim * 3.6:.1f} km/h  error = {error:+.2f} %")

    if abs(error) < 1:
        print(f"\nDONE: muyScale = {tire.muyScale:.4f}, put this value in Tire.py")
        break

    tire.muyScale = tire.muyScale * AY_TARGET / ay_sim
else:
    print("\nNOT CONVERGED after 5 iterations, check car setup and radius")

print(f"target:  ay = {AY_TARGET:.3f} g  V = {V_TARGET:.1f} km/h")