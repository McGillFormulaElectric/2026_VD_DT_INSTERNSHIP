from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from MotecData import MotecData

G = 9.81

# load skidpad run
LAPSIM_ROOT = Path(__file__).resolve().parents[0]
data = MotecData(LAPSIM_ROOT / "1. Data" / "Motec" / "skidpad4.mat")

time = data.getTime("velX")
velX = data.getValue("velX")        # m/s
velY = data.getValue("velY")        # m/s
yawRate = data.getValue("gyrZ_HR")  # rad/s
accY = data.getValue("accY")        # m/s^2, IMU, only used as a check

# low pass filter on IMU accY
fs = 1 / np.mean(np.diff(time))     # sample rate, 100 Hz
cutoff = 10                         # Hz, removes changes faster than 10 times per second
b, a = butter(2, cutoff / (fs / 2))
accY = filtfilt(b, a, accY)

# speed, lateral accel and radius
V = np.sqrt(velX**2 + velY**2)
ay = V * yawRate
R = V / yawRate

# target for the sim
ay_target = np.mean(np.abs(ay)) / G
R_target = np.mean(R)
V_target = np.mean(V) * 3.6

print(f"ay  = {ay_target:.3f} g")
print(f"R   = {R_target:.2f} m")
print(f"V   = {V_target:.1f} km/h")
print(f"IMU = {np.mean(np.abs(accY)) / G:.3f} g (check)")

# checks
imu_diff = abs(ay_target - np.mean(np.abs(accY)) / G)
half = len(ay) // 2
ay_first = np.mean(np.abs(ay[:half])) / G
ay_second = np.mean(np.abs(ay[half:])) / G
drop = ay_first - ay_second

print()
print(f"1. IMU agrees with V*yaw rate:  diff = {imu_diff:.3f} g  -> {'GOOD' if imu_diff < 0.02 else 'BAD'} (< 0.02 g)")
print(f"2. ay flat across the lap:      drop = {drop:.3f} g  -> {'GOOD' if abs(drop) < 0.03 else 'BAD'} (< 0.03 g)")
print(f"3. radius in sane range:        R = {R_target:.2f} m -> {'GOOD' if 8.5 < R_target < 11 else 'BAD'} (8.5 to 11 m)")
print("4. both directions:             only one file, run the other direction and compare (< 0.03 g)")

# grip limit from the second half of the lap, once tyres and driver have settled
R_second = np.mean(R[half:])
V_second = np.mean(V[half:]) * 3.6
print()
print(f"SIM TARGET: ay = {ay_second:.3f} g at R = {R_second:.2f} m (V = {V_second:.1f} km/h)")

# plots
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].plot(time, np.abs(ay) / G, label="V * yaw rate")
axs[0].plot(time, np.abs(accY) / G, label="IMU accY")
axs[0].axhline(ay_target, color="k", linestyle="--")
axs[0].set_ylabel("ay [g]")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(time, R)
axs[1].axhline(R_target, color="k", linestyle="--")
axs[1].set_ylabel("R [m]")
axs[1].set_xlabel("time [s]")
axs[1].grid(True)

plt.show()