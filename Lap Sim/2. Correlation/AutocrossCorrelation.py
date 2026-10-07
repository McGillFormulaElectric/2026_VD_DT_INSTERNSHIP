import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Path to the Lap Sim folder, so we can import MotecData.py
LAPSIM_ROOT = Path(__file__).resolve().parents[1]   # folder that holds MotecData.py
sys.path.insert(0, str(LAPSIM_ROOT))

from MotecData import MotecData

G = 9.81
REF = 647.1          # motor rpm per m/s when free rolling, from BrakeCorrelation.py
POWER_CAP = 80.0     # kW
SPIN = 0.20          # slip above this = wheelspin

# load autocross lap
data = MotecData(LAPSIM_ROOT / "1. Data" / "Motec" / "autocross.mat")

time = data.getTime("velX")
velX = data.getValue("velX")              # m/s, GPS north/east, use the norm
velY = data.getValue("velY")
yawRate = data.getValue("gyrZ_HR")        # rad/s, positive left
throttle = data.getValue("THROTTLE_PERCENT")
power = data.getValue("power")            # kW, pack
rpmFL = data.getValue("SpeedActualFL")    # motor rpm
rpmFR = data.getValue("SpeedActualFR")
rpmRL = data.getValue("SpeedActualRL")
rpmRR = data.getValue("SpeedActualRR")
torque = (data.getValue("TorqueActualFL") + data.getValue("TorqueActualFR")
          + data.getValue("TorqueActualRL") + data.getValue("TorqueActualRR"))   # Nm, 4 motors

dt = np.mean(np.diff(time))

# speed, distance, accelerations (GPS and gyro, the IMU accX/accY are mounted backwards)
V = savgol_filter(np.sqrt(velX**2 + velY**2), 31, 2)
dist = np.cumsum(V) * dt
ax = np.gradient(V, dt) / G                          # g, + accelerating
ay = V * savgol_filter(yawRate, 31, 2) / G           # g, + left

# slip per wheel, + spinning, - locking
slipFL = rpmFL / REF / V - 1
slipFR = rpmFR / REF / V - 1
slipRL = rpmRL / REF / V - 1
slipRR = rpmRR / REF / V - 1
spinning = (slipFL > SPIN) | (slipFR > SPIN) | (slipRL > SPIN) | (slipRR > SPIN)

# what the car is doing at each moment
braking = ax < -0.3
full_throttle = throttle > 90
part_throttle = (throttle > 5) & ~full_throttle & ~braking
coasting = ~braking & ~full_throttle & ~part_throttle

print(f"lap time = {time[-1] - time[0]:.2f} s   distance = {dist[-1]:.0f} m   avg V = {np.mean(V) * 3.6:.1f} km/h")
print()
print("time spent:")
print(f"  full throttle  {np.mean(full_throttle) * 100:5.1f} %")
print(f"  part throttle  {np.mean(part_throttle) * 100:5.1f} %")
print(f"  coasting       {np.mean(coasting) * 100:5.1f} %")
print(f"  braking        {np.mean(braking) * 100:5.1f} %")
print()
print(f"throttle:  max = {np.max(throttle):.0f} %   median when on = {np.median(throttle[throttle > 5]):.0f} %")
print(f"power:     max = {np.max(power):.1f} kW   mean = {np.mean(power):.1f} kW   time above 75 kW = {np.mean(power > 75) * 100:.1f} %")
print(f"torque:    max = {np.max(torque):.1f} Nm (4 motors)")
print(f"wheelspin: {np.mean(spinning) * 100:.1f} % of the lap above {SPIN * 100:.0f} % slip")
print()
print(f"lateral:   99th percentile = {np.percentile(np.abs(ay), 99):.2f} g   (skidpad target 1.35 g at 40 km/h)")
print(f"braking:   99th percentile = {-np.percentile(ax, 1):.2f} g   (brake target 0.93 g at 11 m/s)")
print(f"accel:     99th percentile = {np.percentile(ax, 99):.2f} g")

# right turns past skidpad at low speed: real grip or a spike?
right_hard = (ay < -1.35) & (V * 3.6 < 45)
print(f"right turns past -1.35 g below 45 km/h: {right_hard.sum() * dt:.2f} s total")
if right_hard.any():
    print(f"  at distance {dist[right_hard].min():.0f} to {dist[right_hard].max():.0f} m, peak {ay[right_hard].min():.2f} g")
    
# plots vs distance
fig, axs = plt.subplots(4, 1, sharex=True, figsize=(12, 10))

axs[0].plot(dist, V * 3.6)
axs[0].set_ylabel("V [km/h]")

axs[1].plot(dist, throttle, label="throttle [%]")
axs[1].plot(dist, ax * 100, alpha=0.6, label="ax [g x 100]")
axs[1].set_ylabel("throttle / ax")
axs[1].legend()

axs[2].plot(dist, power)
axs[2].axhline(POWER_CAP, color="r", linestyle="--", label="80 kW cap")
axs[2].set_ylabel("pack power [kW]")
axs[2].legend()

axs[3].plot(dist, slipFL * 100, label="FL")
axs[3].plot(dist, slipFR * 100, label="FR")
axs[3].plot(dist, slipRL * 100, label="RL")
axs[3].plot(dist, slipRR * 100, label="RR")
axs[3].set_ylim(-30, 60)
axs[3].set_ylabel("slip [%]")
axs[3].set_xlabel("distance [m]")
axs[3].legend()

for a in axs:
    a.grid(True)
fig.suptitle("Autocross lap vs distance")
plt.tight_layout()

# g-g diagram, colored by speed
fig2, ax2 = plt.subplots(figsize=(7, 7))
sc = ax2.scatter(ay, ax, c=V * 3.6, s=4, cmap="viridis")
ax2.axvline(1.35, color="k", linestyle="--", alpha=0.5)
ax2.axvline(-1.35, color="k", linestyle="--", alpha=0.5, label="skidpad 1.35 g")
ax2.axhline(-0.93, color="r", linestyle="--", alpha=0.5, label="brake test 0.93 g")
ax2.set_xlabel("ay [g]")
ax2.set_ylabel("ax [g]")
ax2.set_aspect("equal")
ax2.legend()
ax2.grid(True)
fig2.colorbar(sc, label="V [km/h]")
fig2.suptitle("Autocross g-g")
plt.tight_layout()
plt.show()