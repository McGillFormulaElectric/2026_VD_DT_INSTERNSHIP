from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from CarProperties import MFE27
from Tire import Tire
from Acceleration import AccelSolver

G = 9.81

# target from AccelCorrelation.py, accel.mat, full throttle, all 5 checks GOOD
# power limited at 81 kW, so this checks the powertrain, not grip
SPEED_BINS = [18, 21, 24, 27, 29]               # m/s
ACCEL_TARGET = [0.844, 0.691, 0.554, 0.455]     # g, one per bin
PACK_TARGET = [81.2, 81.1, 80.9, 80.9]          # kW, one per bin
V_REV_LIMIT = 29.0                              # m/s, where the car hit 19,000 rpm

# tyre fit files
LAPSIM_ROOT = Path(__file__).resolve().parents[0]
TIRE_DIR = LAPSIM_ROOT / "1. Data" / "Tire"

# car and tire with both correlated grip scales
car = MFE27()
tire = Tire(TIRE_DIR / "MF61_Parameters.mat", TIRE_DIR / "FZ_Reference.mat", muxScale=0.451, muyScale=0.59)

# run the sim accel event
accel = AccelSolver().simulate(tire, car)
v = np.asarray(accel["v"])                      # m/s
t = np.asarray(accel["t"])                      # s
P_pack = np.asarray(accel["P_pack"]) / 1000     # kW, assumes the solver returns W, remove /1000 if it is already kW

# sim accel from the speed trace
a_sim = np.gradient(v, t) / G                   # g

# compare per speed bin
print(f"{'bin [m/s]':>10} {'test [g]':>9} {'sim [g]':>8} {'error':>8} {'test [kW]':>10} {'sim [kW]':>9}")
for lo, hi, a_target, p_target in zip(SPEED_BINS[:-1], SPEED_BINS[1:], ACCEL_TARGET, PACK_TARGET):
    in_bin = (v > lo) & (v < hi)
    if in_bin.sum() == 0:
        print(f"{lo:>4}-{hi:<5} sim never reaches this speed")
        continue
    a_bin = np.mean(a_sim[in_bin])
    p_bin = np.mean(P_pack[in_bin])
    error = (a_bin - a_target) / a_target * 100
    print(f"{lo:>4}-{hi:<5} {a_target:>9.3f} {a_bin:>8.3f} {error:>+7.1f}% {p_target:>10.1f} {p_bin:>9.1f}")

print()
print(f"top speed: test {V_REV_LIMIT:.1f} m/s at rev limit, sim {accel['v_max']:.1f} m/s, rev_limited = {accel['rev_limited']}")

# plot sim accel vs speed with the test bins on top
bin_centers = [(lo + hi) / 2 for lo, hi in zip(SPEED_BINS[:-1], SPEED_BINS[1:])]

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(9, 7))

axs[0].plot(v, a_sim, label="sim")
axs[0].plot(bin_centers, ACCEL_TARGET, "o", color="k", label="test")
axs[0].set_ylabel("accel [g]")
axs[0].legend()

axs[1].plot(v, P_pack, label="sim")
axs[1].plot(bin_centers, PACK_TARGET, "o", color="k", label="test")
axs[1].set_ylabel("pack power [kW]")
axs[1].set_xlabel("V [m/s]")
axs[1].legend()

for ax in axs:
    ax.grid(True)
fig.suptitle("Accel, sim vs test")
plt.tight_layout()
plt.show()