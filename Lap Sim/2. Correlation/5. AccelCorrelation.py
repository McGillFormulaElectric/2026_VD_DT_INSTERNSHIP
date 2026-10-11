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

G = 9.81                            # m/s^2
REF = 647.1                         # motor rpm per m/s when free rolling, from BrakeCorrelation.py
RPM_LIMIT = 18900                   # rpm, stop the window just before the rev limit
SPEED_BINS = [18, 21, 24, 27, 29]   # m/s, accel is averaged in each bin for the sim target
WHEELS = ["FL", "FR", "RL", "RR"]

# ── load accel run ────────────────────────────────────────────
data = MotecData(LAPSIM_ROOT / "1. Data" / "Motec" / "accel.mat")

time = data.getTime("velX")                       # s
velX = data.getValue("velX")                      # m/s
velY = data.getValue("velY")                      # m/s
yaw_rate = data.getValue("gyrZ_HR")               # rad/s
throttle = data.getValue("THROTTLE_PERCENT")      # %
power = data.getValue("power")                    # kW, electrical from the pack
rpm = {w: data.getValue(f"SpeedActual{w}") for w in WHEELS}                           # rpm, per motor
torque = sum(data.getValue(f"TorqueActual{w}") for w in WHEELS)                      # Nm, sum of 4 motors
torque_limit = sum(data.getValue(f"TorqueLimitPositive{w}") for w in WHEELS)         # Nm, sum of 4 motors
dt = np.mean(np.diff(time))                       # s

# ── derived channels ──────────────────────────────────────────
V = savgol_filter(np.sqrt(velX**2 + velY**2), 31, 2)       # m/s, car speed smoothed over about 0.3 s
accel = np.gradient(V, dt) / G                              # g
slip = {w: rpm[w] / REF / V - 1 for w in WHEELS}            # -, positive when driving
rpm_avg = sum(rpm.values()) / 4                             # rpm
power_mech = torque * rpm_avg * 2 * np.pi / 60 / 1000       # kW, at the motor shafts

# full throttle, above the first bin, before the rev limit
window = (throttle > 90) & (V > SPEED_BINS[0]) & (rpm_avg < RPM_LIMIT)

# ── checks: is this run a clean powertrain test ───────────────
throttle_min = np.min(throttle[window])                                                       # %
power_mean = np.mean(power[window])                                                           # kW
torque_gap = np.mean(np.abs(torque[window] - torque_limit[window]) / torque_limit[window])    # -, fraction
worst_slip = max(np.median(slip[w][window]) for w in WHEELS)                                 # -, fraction
yaw_max = np.max(np.abs(yaw_rate[window]))                                                   # rad/s
efficiency = np.mean(power_mech[window]) / power_mean                                        # -, shaft / pack

def check(text, good):
    print(f"{text:<55} {'GOOD' if good else 'BAD'}")

check(f"1. full throttle:     min {throttle_min:.0f} % (> 90 %)", throttle_min > 90)
check(f"2. power limited:     pack {power_mean:.1f} kW (75 to 85 kW)", 75 < power_mean < 85)
check(f"3. torque on limit:   gap {torque_gap * 100:.1f} % (< 3 %)", torque_gap < 0.03)
check(f"4. no wheelspin:      worst slip {worst_slip * 100:.1f} % (< 5 %)", worst_slip < 0.05)
check(f"5. straight line:     yaw rate {yaw_max:.2f} rad/s (< 0.3)", yaw_max < 0.3)

print(f"\nmotor + inverter efficiency = {efficiency:.3f}")
print(f"rev limit reached at V = {V[np.argmax(rpm_avg > RPM_LIMIT)]:.1f} m/s")

# ── sim targets, average per speed bin ────────────────────────
print("\nSIM TARGET:")
print(f"{'bin [m/s]':>10} {'accel [g]':>10} {'pack [kW]':>10} {'torque [Nm]':>12}")
for lo, hi in zip(SPEED_BINS[:-1], SPEED_BINS[1:]):
    in_bin = window & (V > lo) & (V < hi)
    print(f"{lo:>4}-{hi:<5} {np.mean(accel[in_bin]):>10.3f} {np.mean(power[in_bin]):>10.1f} {np.mean(torque[in_bin]):>12.1f}")

# ── plots vs speed, full throttle window only ─────────────────
Vw = V[window]   # m/s

fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 10))

axs[0].plot(Vw, accel[window])
axs[0].set_ylabel("accel [g]")

axs[1].plot(Vw, power[window], label="pack")
axs[1].plot(Vw, power_mech[window], label="motor shafts")
axs[1].set_ylabel("power [kW]")
axs[1].legend()

axs[2].plot(Vw, torque[window], label="actual")
axs[2].plot(Vw, torque_limit[window], "--", label="limit")
axs[2].set_ylabel("torque, 4 motors [Nm]")
axs[2].legend()

for w in WHEELS:
    axs[3].plot(Vw, slip[w][window] * 100, label=w)
axs[3].set_ylim(-5, 20)
axs[3].set_ylabel("slip [%]")
axs[3].set_xlabel("speed [m/s]")
axs[3].legend()

for ax in axs:
    ax.grid(True)
plt.tight_layout()
plt.show()