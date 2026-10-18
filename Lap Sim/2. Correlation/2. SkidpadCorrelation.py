import sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt

# Path to the Lap Sim folder
LAPSIM_ROOT = Path(__file__).resolve().parents[1]   # folder that holds MotecData.py
sys.path.insert(0, str(LAPSIM_ROOT))

from MotecData import MotecData

G = 9.81        # m/s^2
CUTOFF = 2      # Hz, low pass on every channel, same filter on all so they compare fairly
IMU_SIGN = -1   # IMU mounted backwards: accY reads the opposite sign of the car
R_X = -0.255    # m, IMU is 255 mm behind the CG (x forward)

# load skidpad run
data = MotecData(LAPSIM_ROOT / "1. Data" / "Motec" / "skidpad2.mat")
time = data.getTime("velX")             # s
velX = data.getValue("velX")            # m/s
velY = data.getValue("velY")            # m/s
yaw_rate = data.getValue("gyrZ_HR")     # rad/s
accY = data.getValue("accY")            # m/s^2, IMU
roll = np.radians(data.getValue("Roll"))   # rad, IMU frame

fs = 1 / np.mean(np.diff(time))         # Hz, sample rate, about 100 Hz
b, a = butter(2, CUTOFF / (fs / 2))
lp = lambda x: filtfilt(b, a, x)

# lateral accel: IMU corrected (target) and V x yaw rate (check)
V = lp(np.sqrt(velX**2 + velY**2))      # m/s
r = lp(yaw_rate)                        # rad/s
r_dot = np.gradient(r, time)            # rad/s^2, yaw acceleration

ay_imu = IMU_SIGN * (accY - G * np.sin(roll))   # m/s^2, gravity from body roll removed
ay_imu = lp(ay_imu - R_X * r_dot)               # m/s^2, moved to the CG: a_y,CG = a_y,IMU - r_x * yaw accel
ay_imu = np.abs(ay_imu) / G                     # g, sign removed so both directions read +

ay = np.abs(V * r) / G                  # g, ay = V x yaw rate, check
R = np.abs(V / r)                       # m, R = V / yaw rate

half = len(ay) // 2                     # second half = tyres and driver settled

print(f"IMU = {np.mean(ay_imu):.3f} g (target)")
print(f"ay  = {np.mean(ay):.3f} g (V x yaw rate, check)")
print(f"R   = {np.mean(R):.2f} m")
print(f"V   = {np.mean(V) * 3.6:.1f} km/h")

# checks: is this run a clean grip test
imu_diff = abs(np.mean(ay) - np.mean(ay_imu))             # g
drop = np.mean(ay_imu[:half]) - np.mean(ay_imu[half:])    # g, + means grip fell during the run

def check(text, good):
    print(f"{text:<55} {'GOOD' if good else 'BAD'}")

print()
check(f"1. IMU agrees:     diff {imu_diff:.3f} g (< 0.02 g)", imu_diff < 0.02)
check(f"2. ay flat:        drop {drop:.3f} g (< 0.03 g)", abs(drop) < 0.03)
check(f"3. radius sane:    R {np.mean(R):.2f} m (8.5 to 11 m)", 8.5 < np.mean(R) < 11)
print("4. both directions: only one file, run the other way and compare (< 0.03 g)")

print(f"\nSIM TARGET: ay = {np.mean(ay_imu[half:]):.3f} g at R = {np.mean(R[half:]):.2f} m "
      f"(V = {np.mean(V[half:]) * 3.6:.1f} km/h)")