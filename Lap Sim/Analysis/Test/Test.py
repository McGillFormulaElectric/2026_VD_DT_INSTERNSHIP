import time
from pathlib import Path

import sys, pathlib
CORE = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CORE))


from CarProperties import MFE27
from Tire import Tire
from Acceleration import AccelSolver
from SkidPad import SkidPadSolver

LAPSIM = Path(__file__).resolve().parent.parent.parent

car = MFE27()
tire = Tire(LAPSIM / "Data" / "Tire" / "MF61_Coefficients.csv")

import cProfile, pstats
cProfile.run("AccelSolver().simulate(tire, car)", "accel.prof")
pstats.Stats("accel.prof").sort_stats("cumulative").print_stats(12)

# ACCELERATION
print("=" * 50)
print("ACCELERATION")
print("=" * 50)

t0 = time.perf_counter()
accel = AccelSolver().simulate(tire, car)      # capture the result AND time it
solve_time = time.perf_counter() - t0

print(f"solve time  : {solve_time*1000:.0f} ms  ")
print(f"75 m time   : {accel['lap_time']:.3f} s      (expect 3.6 to 4.2)")
print(f"trap speed  : {accel['v_max']*3.6:.1f} km/h  (expect 100 to 110)")
print(f"pack energy : {accel['energy_kWh']*3600:.2f} kJ")
print(f"comp point  : {accel['comp_point']:.1f}")


# SKIDPAD
print()
print("=" * 50)
print("SKIDPAD")
print("=" * 50)

t0 = time.perf_counter()
skid = SkidPadSolver().simulate(tire, car)
solve_time = time.perf_counter() - t0

print(f"solve time  : {solve_time*1000:.0f} ms")
print(f"lap time    : {skid['time']:.3f} s      (expect 4.8 to 5.3)")
print(f"speed       : {skid['v']*3.6:.1f} km/h")
print(f"lateral     : {skid['AyG']:.2f} g       (expect around 1.4)")
print(f"RL {skid['FzRL']:.0f}  RR {skid['FzRR']:.0f}")

if min(skid["FzFL"], skid["FzFR"], skid["FzRL"], skid["FzRR"]) <= 1.0:
    print("  NOTE: an inside wheel has lifted (load ~ 0).")
