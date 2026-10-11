import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Path to the Lap Sim folder
LAPSIM_ROOT = Path(__file__).resolve().parents[1]   # folder that holds MotecData.py
sys.path.insert(0, str(LAPSIM_ROOT))

from MotecData import MotecData

G = 9.81             # m/s^2
REF = 647.1          # motor rpm per m/s when free rolling, from BrakeCorrelation.py
POWER_CAP = 80.0     # kW
SPIN = 0.20          # -, slip above this = wheelspin
AY_SKIDPAD = 1.35    # g, skidpad target
AX_BRAKE = 0.93      # g, brake test target
WHEELS = ["FL", "FR", "RL", "RR"]

# ── load lap ──────────────────────────────────────────────────
data = MotecData(LAPSIM_ROOT / "1. Data" / "Motec" / "autocross.mat")
time = data.getTime("velX")                                       # s
velX = data.getValue("velX")                                      # m/s
velY = data.getValue("velY")                                      # m/s
yaw_rate = data.getValue("gyrZ_HR")                               # rad/s, + left
throttle = data.getValue("THROTTLE_PERCENT")                      # %
power = data.getValue("power")                                    # kW, pack
rpm = {w: data.getValue(f"SpeedActual{w}") for w in WHEELS}       # rpm, per motor
torque = sum(data.getValue(f"TorqueActual{w}") for w in WHEELS)   # Nm, sum of 4 motors
dt = np.mean(np.diff(time))                                       # s

# ── derived channels (IMU accX/accY are mounted backwards, so use GPS and gyro) ──
V = savgol_filter(np.sqrt(velX**2 + velY**2), 31, 2)     # m/s
dist = np.cumsum(V) * dt                                 # m
ax = np.gradient(V, dt) / G                              # g, + accelerating
ay = V * savgol_filter(yaw_rate, 31, 2) / G              # g, + left
slip = {w: rpm[w] / REF / V - 1 for w in WHEELS}         # -, + spinning, - locking
spinning = np.any([slip[w] > SPIN for w in WHEELS], axis=0)

# what the car is doing at each point
braking = ax < -0.3
full_throttle = throttle > 90
part_throttle = (throttle > 5) & ~full_throttle & ~braking
coasting = ~braking & ~full_throttle & ~part_throttle

# ── summary ───────────────────────────────────────────────────
print(f"lap {time[-1] - time[0]:.2f} s   distance {dist[-1]:.0f} m   avg {np.mean(V) * 3.6:.1f} km/h\n")

print("time spent:")
print(f"  full throttle  {np.mean(full_throttle) * 100:5.1f} %")
print(f"  part throttle  {np.mean(part_throttle) * 100:5.1f} %")
print(f"  coasting       {np.mean(coasting) * 100:5.1f} %")
print(f"  braking        {np.mean(braking) * 100:5.1f} %\n")

print(f"throttle:  max {np.max(throttle):.0f} %   median when on {np.median(throttle[throttle > 5]):.0f} %")
print(f"power:     max {np.max(power):.1f} kW   mean {np.mean(power):.1f} kW   above 75 kW {np.mean(power > 75) * 100:.1f} % of lap")
print(f"torque:    max {np.max(torque):.1f} Nm")
print(f"wheelspin: {np.mean(spinning) * 100:.1f} % of lap above {SPIN * 100:.0f} % slip\n")

print(f"lateral:   {np.percentile(np.abs(ay), 99):.2f} g   (99th pct, skidpad {AY_SKIDPAD} g)")
print(f"braking:   {-np.percentile(ax, 1):.2f} g   (99th pct, brake test {AX_BRAKE} g)")
print(f"accel:     {np.percentile(ax, 99):.2f} g   (99th pct)")

# right turns past the skidpad value at low speed, real grip or a spike
right_hard = (ay < -AY_SKIDPAD) & (V * 3.6 < 45)
print(f"right turns past -{AY_SKIDPAD} g below 45 km/h: {right_hard.sum() * dt:.2f} s")
if right_hard.any():
    print(f"  from {dist[right_hard].min():.0f} to {dist[right_hard].max():.0f} m, peak {ay[right_hard].min():.2f} g")

# ── plots vs distance ─────────────────────────────────────────
fig, axs = plt.subplots(4, 1, sharex=True, figsize=(12, 10))

axs[0].plot(dist, V * 3.6)
axs[0].set_ylabel("speed [km/h]")

axs[1].plot(dist, throttle, label="throttle [%]")
axs[1].plot(dist, ax * 100, label="ax [g x 100]")
axs[1].legend()

axs[2].plot(dist, power)
axs[2].axhline(POWER_CAP, color="r", linestyle="--")
axs[2].set_ylabel("pack power [kW]")

for w in WHEELS:
    axs[3].plot(dist, slip[w] * 100, label=w)
axs[3].set_ylim(-30, 60)
axs[3].set_ylabel("slip [%]")
axs[3].set_xlabel("distance [m]")
axs[3].legend()

for a in axs:
    a.grid(True)
plt.tight_layout()

# ── g-g diagram, colour = speed ───────────────────────────────
fig2, ax2 = plt.subplots(figsize=(7, 7))
sc = ax2.scatter(ay, ax, c=V * 3.6, s=4, cmap="viridis")
ax2.axvline(AY_SKIDPAD, color="k", linestyle="--")
ax2.axvline(-AY_SKIDPAD, color="k", linestyle="--", label="skidpad")
ax2.axhline(-AX_BRAKE, color="r", linestyle="--", label="brake test")
ax2.set_xlabel("ay [g]")
ax2.set_ylabel("ax [g]")
ax2.set_aspect("equal")
ax2.legend()
ax2.grid(True)
fig2.colorbar(sc, label="speed [km/h]")
plt.tight_layout()
plt.show()