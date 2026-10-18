import sys
from pathlib import Path
import numpy as np
from scipy.signal import savgol_filter

# Path to the Lap Sim folder
LAPSIM_ROOT = Path(__file__).resolve().parents[1]   # folder that holds MotecData.py
sys.path.insert(0, str(LAPSIM_ROOT))

from MotecData import MotecData

G = 9.81                    # m/s^2
V_LOW, V_HIGH = 8.0, 14.0   # m/s, target window, low speed so drag is small
R_X = -0.255                # m, IMU is 255 mm behind the CG (x forward)
WHEELS = ["FL", "FR", "RL", "RR"]

# ── load brake run ────────────────────────────────────────────
data = MotecData(LAPSIM_ROOT / "1. Data" / "Motec" / "brake.mat")
time = data.getTime("velX")                                        # s
velX = data.getValue("velX")                                       # m/s
velY = data.getValue("velY")                                       # m/s
accX = data.getValue("accX")                                       # m/s^2, IMU mounted backwards, so + = braking
pitch = np.radians(data.getValue("Pitch"))                         # rad, IMU frame, includes the 4.2 deg tilt
yaw_rate = data.getValue("gyrZ_HR")                                # rad/s
throttle = data.getValue("THROTTLE_PERCENT")                       # %
rpm = {w: data.getValue(f"SpeedActual{w}") for w in WHEELS}        # rpm, per motor
torque = {w: data.getValue(f"TorqueActual{w}") for w in WHEELS}    # Nm, per motor
dt = np.mean(np.diff(time))                                        # s

# ── derived channels ──────────────────────────────────────────
V = savgol_filter(np.sqrt(velX**2 + velY**2), 31, 2)   # m/s, smoothed over about 0.3 s
decel_gps = -np.gradient(V, dt) / G                    # g, + braking, check
rpm_avg = sum(rpm.values()) / 4                        # rpm

# IMU corrected: tilt gravity removed, moved to the CG (a_x,CG = a_x,IMU + r_x * yaw rate^2)
decel_imu = savgol_filter((accX + G * np.sin(pitch)) / G, 31, 2)   # g, + braking
decel_imu = decel_imu - R_X * yaw_rate**2 / G                      # g, at the CG (tiny in a straight line)

# free rolling reference, motor rpm per m/s while cruising before the stop
cruise = (throttle > 50) & (np.abs(decel_gps) < 0.1)
REF = np.mean(rpm_avg[cruise] / V[cruise])             # rpm per m/s

slip = {w: rpm[w] / REF / V - 1 for w in WHEELS}       # -, negative when braking

# target window: off throttle, between V_LOW and V_HIGH
window = (throttle < 1) & (V > V_LOW) & (V < V_HIGH)

# ── target ────────────────────────────────────────────────────
decel_target = np.mean(decel_imu[window])                           # g
V_target = np.mean(V[window])                                       # m/s
decel_check = np.mean(decel_gps[window])                            # g

print(f"IMU   = {decel_target:.3f} g (target, tilt + CG corrected)")
print(f"GPS   = {decel_check:.3f} g (check)")
print(f"raw   = {np.mean(accX[window]) / G:.3f} g (raw IMU, reads high from the 4.2 deg tilt)")
print(f"V     = {V_target:.1f} m/s")
print(f"free rolling ref = {REF:.1f} rpm per m/s")
print("median slip  " + "  ".join(f"{w} {np.median(slip[w][window]) * 100:.1f} %" for w in WHEELS))

# ── checks: is this run a clean grip test ─────────────────────
slip_all = np.concatenate([slip[w][window] for w in WHEELS])                         # -
gps_diff = abs(decel_target - decel_check)                                           # g
torque_max = np.max(np.abs(np.concatenate([torque[w][window] for w in WHEELS])))     # Nm
slip_median = np.median(slip_all)                                                    # -
slip_min = np.min(slip_all)                                                          # -
yaw_max = np.max(np.abs(yaw_rate[window]))                                           # rad/s

def check(text, good):
    print(f"{text:<55} {'GOOD' if good else 'BAD'}")

print()
check(f"1. GPS agrees:      diff {gps_diff:.3f} g (< 0.02 g)", gps_diff < 0.02)
check(f"2. no regen:        max torque {torque_max:.2f} Nm (< 1 Nm)", torque_max < 1)
check(f"3. at grip limit:   median slip {slip_median * 100:.1f} % (-5 to -20 %)", -0.20 < slip_median < -0.05)
check(f"4. no wheel locked: min slip {slip_min * 100:.1f} % (> -50 %)", slip_min > -0.5)
check(f"5. straight line:   yaw rate {yaw_max:.2f} rad/s (< 0.3)", yaw_max < 0.3)

print(f"\nSIM TARGET: decel = {decel_target:.3f} g at V = {V_target:.1f} m/s")