import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Path to the Lap Sim folder
LAPSIM_ROOT = Path(__file__).resolve().parents[1]   # folder that holds MotecData.py
sys.path.insert(0, str(LAPSIM_ROOT))

from MotecData import MotecData

G = 9.81        # m/s^2
CUTOFF = 10     # Hz, low pass on IMU accY, removes changes faster than 10 times per second

# ── load skidpad run ──────────────────────────────────────────
data = MotecData(LAPSIM_ROOT / "1. Data" / "Motec" / "skidpad5.mat")
time = data.getTime("velX")             # s
velX = data.getValue("velX")            # m/s
velY = data.getValue("velY")            # m/s
yaw_rate = data.getValue("gyrZ_HR")     # rad/s
accY = data.getValue("accY")            # m/s^2, IMU, only a check

fs = 1 / np.mean(np.diff(time))         # Hz, sample rate, about 100 Hz
b, a = butter(2, CUTOFF / (fs / 2))
accY = filtfilt(b, a, accY)             # m/s^2, filtered

# ── speed, lateral accel and radius ───────────────────────────
V = np.sqrt(velX**2 + velY**2)          # m/s
ay = np.abs(V * yaw_rate) / G           # g, ay = V x yaw rate, sign removed so both directions read +
R = np.abs(V / yaw_rate)                # m, R = V / yaw rate
ay_imu = np.abs(accY) / G               # g

half = len(ay) // 2                     # second half = tyres and driver settled

print(f"ay  = {np.mean(ay):.3f} g")
print(f"R   = {np.mean(R):.2f} m")
print(f"V   = {np.mean(V) * 3.6:.1f} km/h")
print(f"IMU = {np.mean(ay_imu):.3f} g (check)")

# ── checks: is this run a clean grip test ─────────────────────
imu_diff = abs(np.mean(ay) - np.mean(ay_imu))             # g
drop = np.mean(ay[:half]) - np.mean(ay[half:])            # g, + means grip fell during the run

def check(text, good):
    print(f"{text:<55} {'GOOD' if good else 'BAD'}")

print()
check(f"1. IMU agrees:     diff {imu_diff:.3f} g (< 0.02 g)", imu_diff < 0.02)
check(f"2. ay flat:        drop {drop:.3f} g (< 0.03 g)", abs(drop) < 0.03)
check(f"3. radius sane:    R {np.mean(R):.2f} m (8.5 to 11 m)", 8.5 < np.mean(R) < 11)
print("4. both directions: only one file, run the other way and compare (< 0.03 g)")

print(f"\nSIM TARGET: ay = {np.mean(ay[half:]):.3f} g at R = {np.mean(R[half:]):.2f} m "
      f"(V = {np.mean(V[half:]) * 3.6:.1f} km/h)")

# ── plots vs time ─────────────────────────────────────────────
fig, axs = plt.subplots(2, 1, sharex=True)

axs[0].plot(time, ay, label="V x yaw rate")
axs[0].plot(time, ay_imu, label="IMU accY")
axs[0].axhline(np.mean(ay), color="k", linestyle="--")
axs[0].set_ylabel("ay [g]")
axs[0].legend()

axs[1].plot(time, R)
axs[1].axhline(np.mean(R), color="k", linestyle="--")
axs[1].set_ylabel("R [m]")
axs[1].set_xlabel("time [s]")

for ax in axs:
    ax.grid(True)
plt.tight_layout()
plt.show()