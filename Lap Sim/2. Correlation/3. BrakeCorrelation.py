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

G = 9.81                    # m/s^2
V_LOW, V_HIGH = 8.0, 14.0   # m/s, target window, low speed so drag is small
WHEELS = ["FL", "FR", "RL", "RR"]

# ── load brake run ────────────────────────────────────────────
data = MotecData(LAPSIM_ROOT / "1. Data" / "Motec" / "brake.mat")
time = data.getTime("velX")                                        # s
velX = data.getValue("velX")                                       # m/s
velY = data.getValue("velY")                                       # m/s
accX = data.getValue("accX")                                       # m/s^2, IMU, + braking, only a check
yaw_rate = data.getValue("gyrZ_HR")                                # rad/s
throttle = data.getValue("THROTTLE_PERCENT")                       # %
rpm = {w: data.getValue(f"SpeedActual{w}") for w in WHEELS}        # rpm, per motor
torque = {w: data.getValue(f"TorqueActual{w}") for w in WHEELS}    # Nm, per motor
dt = np.mean(np.diff(time))                                        # s

# ── derived channels ──────────────────────────────────────────
V = savgol_filter(np.sqrt(velX**2 + velY**2), 31, 2)   # m/s, smoothed over about 0.3 s
decel = -np.gradient(V, dt) / G                        # g, + braking
rpm_avg = sum(rpm.values()) / 4                        # rpm

# free rolling reference, motor rpm per m/s while cruising before the stop
cruise = (throttle > 50) & (np.abs(decel) < 0.1)
REF = np.mean(rpm_avg[cruise] / V[cruise])             # rpm per m/s

slip = {w: rpm[w] / REF / V - 1 for w in WHEELS}       # -, negative when braking
V_wheel = rpm_avg / REF                                # m/s, only to plot, reads low because the wheels slip

# target window: off throttle, between V_LOW and V_HIGH
window = (throttle < 1) & (V > V_LOW) & (V < V_HIGH)

# ── target ────────────────────────────────────────────────────
decel_target = np.mean(decel[window])                               # g
V_target = np.mean(V[window])                                       # m/s
decel_fit = -np.polyfit(time[window], V[window], 1)[0] / G          # g, slope of a straight line fit, a check

print(f"decel = {decel_target:.3f} g")
print(f"V     = {V_target:.1f} m/s")
print(f"fit   = {decel_fit:.3f} g (check)")
print(f"IMU   = {np.mean(accX[window]) / G:.3f} g (check, reads high from pitch)")
print(f"free rolling ref = {REF:.1f} rpm per m/s")
print("median slip  " + "  ".join(f"{w} {np.median(slip[w][window]) * 100:.1f} %" for w in WHEELS))

# ── checks: is this run a clean grip test ─────────────────────
slip_all = np.concatenate([slip[w][window] for w in WHEELS])                         # -
fit_diff = abs(decel_target - decel_fit)                                             # g
torque_max = np.max(np.abs(np.concatenate([torque[w][window] for w in WHEELS])))     # Nm
slip_median = np.median(slip_all)                                                    # -
slip_min = np.min(slip_all)                                                          # -
yaw_max = np.max(np.abs(yaw_rate[window]))                                           # rad/s

def check(text, good):
    print(f"{text:<55} {'GOOD' if good else 'BAD'}")

print()
check(f"1. fit agrees:      diff {fit_diff:.3f} g (< 0.02 g)", fit_diff < 0.02)
check(f"2. no regen:        max torque {torque_max:.2f} Nm (< 1 Nm)", torque_max < 1)
check(f"3. at grip limit:   median slip {slip_median * 100:.1f} % (-5 to -20 %)", -0.20 < slip_median < -0.05)
check(f"4. no wheel locked: min slip {slip_min * 100:.1f} % (> -50 %)", slip_min > -0.5)
check(f"5. straight line:   yaw rate {yaw_max:.2f} rad/s (< 0.3)", yaw_max < 0.3)

print(f"\nSIM TARGET: decel = {decel_target:.3f} g at V = {V_target:.1f} m/s")

# ── plots vs time, target window in green ─────────────────────
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

axs[0].plot(time, V, label="GPS")
axs[0].plot(time, V_wheel, label="wheels")
axs[0].set_ylabel("speed [m/s]")
axs[0].legend()

axs[1].plot(time, decel, label="GPS")
axs[1].plot(time, accX / G, alpha=0.4, label="IMU accX")
axs[1].axhline(decel_target, color="k", linestyle="--")
axs[1].set_ylabel("decel [g]")
axs[1].legend()

for w in WHEELS:
    axs[2].plot(time, slip[w] * 100, label=w)
axs[2].set_ylim(-40, 10)
axs[2].set_ylabel("slip [%]")
axs[2].set_xlabel("time [s]")
axs[2].legend()

for ax in axs:
    ax.fill_between(time, 0, 1, where=window, transform=ax.get_xaxis_transform(), color="tab:green", alpha=0.15)
    ax.grid(True)
plt.tight_layout()
plt.show()