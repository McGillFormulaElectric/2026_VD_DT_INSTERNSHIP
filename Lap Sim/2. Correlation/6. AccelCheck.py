import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Path to the Lap Sim folder
LAPSIM_ROOT = Path(__file__).resolve().parents[1]   # folder that holds CarProperties.py
sys.path.insert(0, str(LAPSIM_ROOT))

from CarProperties import MFE26
from Tire import Tire
from Acceleration import AccelSolver

# test targets from accel.mat, full throttle, power limited at about 81 kW
# one row per speed bin: (low speed [m/s], high speed [m/s], accel [g], pack power [kW])
TEST = [
    (18, 21, 0.844, 81.2),
    (21, 24, 0.691, 81.1),
    (24, 27, 0.554, 80.9),
    (27, 29, 0.455, 80.9),
]
V_TOP_TEST = 29.0   # m/s, test hit the 19,000 rpm rev limit here

# car and tire with the correlated grip scales
TIRE_DIR = LAPSIM_ROOT / "1. Data" / "Tire"
car = MFE26()
tire = Tire(TIRE_DIR / "MF61_Parameters.mat", TIRE_DIR / "FZ_Reference.mat")

# run the sim
accel = AccelSolver().simulate(tire, car)
v = np.asarray(accel["v"])                    # m/s
t = np.asarray(accel["t"])                    # s
P = np.asarray(accel["P_pack"]) / 1000        # kW, solver returns W
a = np.gradient(v, t) / car.g                     # g, accel from the speed trace

# compare the sim average in each speed bin to the test
print(f"{'bin [m/s]':>10} {'test [g]':>9} {'sim [g]':>8} {'error':>7} {'test [kW]':>10} {'sim [kW]':>9}")
for lo, hi, a_test, P_test in TEST:
    in_bin = (v > lo) & (v < hi)              # sim points inside this speed bin
    a_bin = np.mean(a[in_bin])                # g
    P_bin = np.mean(P[in_bin])                # kW
    error = (a_bin / a_test - 1) * 100        # %, + means sim pulls harder than test
    print(f"{lo:>4}-{hi:<5} {a_test:>9.3f} {a_bin:>8.3f} {error:>+6.1f}% {P_test:>10.1f} {P_bin:>9.1f}")

print(f"\ntop speed: test {V_TOP_TEST:.1f} m/s, sim {accel['v_max']:.1f} m/s, rev limited = {accel['rev_limited']}")

# plot sim vs speed, test bins as dots at the bin centre
v_test = [(lo + hi) / 2 for lo, hi, _, _ in TEST]   # m/s
a_test = [row[2] for row in TEST]                    # g
P_test = [row[3] for row in TEST]                    # kW

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(9, 7))

axs[0].plot(v, a, label="sim")
axs[0].plot(v_test, a_test, "ko", label="test")
axs[0].set_ylabel("accel [g]")
axs[0].legend()

axs[1].plot(v, P, label="sim")
axs[1].plot(v_test, P_test, "ko", label="test")
axs[1].set_ylabel("pack power [kW]")
axs[1].set_xlabel("speed [m/s]")
axs[1].legend()

for ax in axs:
    ax.grid(True)
plt.tight_layout()
plt.show()