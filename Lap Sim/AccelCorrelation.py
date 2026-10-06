from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from MotecData import MotecData

G = 9.81
REF = 647.1                         # motor rpm per m/s when free rolling, from BrakeCorrelation.py
SPEED_BINS = [18, 21, 24, 27, 29]   # m/s, accel is averaged in each bin for the sim target

# load accel run
LAPSIM_ROOT = Path(__file__).resolve().parents[0]
data = MotecData(LAPSIM_ROOT / "1. Data" / "Motec" / "accel.mat")

time = data.getTime("velX")
velX = data.getValue("velX")              # m/s
velY = data.getValue("velY")              # m/s
yawRate = data.getValue("gyrZ_HR")        # rad/s
throttle = data.getValue("THROTTLE_PERCENT")
power = data.getValue("power")            # kW, electrical from the pack
rpmFL = data.getValue("SpeedActualFL")    # motor rpm
rpmFR = data.getValue("SpeedActualFR")
rpmRL = data.getValue("SpeedActualRL")
rpmRR = data.getValue("SpeedActualRR")
torque = (data.getValue("TorqueActualFL") + data.getValue("TorqueActualFR")
          + data.getValue("TorqueActualRL") + data.getValue("TorqueActualRR"))            # Nm, sum of 4 motors
torque_limit = (data.getValue("TorqueLimitPositiveFL") + data.getValue("TorqueLimitPositiveFR")
                + data.getValue("TorqueLimitPositiveRL") + data.getValue("TorqueLimitPositiveRR"))

dt = np.mean(np.diff(time))

# speed and accel from GPS speed, smoothed over 0.3 s before the derivative
V = savgol_filter(np.sqrt(velX**2 + velY**2), 31, 2)
accel = np.gradient(V, dt) / G           # g

# slip per wheel, positive when driving
slipFL = rpmFL / REF / V - 1
slipFR = rpmFR / REF / V - 1
slipRL = rpmRL / REF / V - 1
slipRR = rpmRR / REF / V - 1

# mechanical power at the motor shafts, to compare with electrical power
rpm_avg = (rpmFL + rpmFR + rpmRL + rpmRR) / 4
power_mech = torque * rpm_avg * 2 * np.pi / 60 / 1000   # kW

# window: full throttle, before the driver lifts at the rpm limit
window = (throttle > 90) & (V > SPEED_BINS[0]) & (rpm_avg < 18900)

# checks
power_mean = np.mean(power[window])
torque_gap = np.mean(np.abs(torque[window] - torque_limit[window]) / torque_limit[window])
median_slip = max(np.median(slipFL[window]), np.median(slipFR[window]), np.median(slipRL[window]), np.median(slipRR[window]))
max_yaw = np.max(np.abs(yawRate[window]))
efficiency = np.mean(power_mech[window]) / power_mean

print(f"1. full throttle:            min = {np.min(throttle[window]):.0f} %      -> {'GOOD' if np.min(throttle[window]) > 90 else 'BAD'} (> 90 %)")
print(f"2. power limited at 80 kW:   pack = {power_mean:.1f} kW   -> {'GOOD' if 75 < power_mean < 85 else 'BAD'} (75 to 85 kW)")
print(f"3. torque follows its limit: gap = {torque_gap * 100:.1f} %      -> {'GOOD' if torque_gap < 0.03 else 'BAD'} (< 3 %)")
print(f"4. no wheelspin:             worst median slip = {median_slip * 100:.1f} % -> {'GOOD' if median_slip < 0.05 else 'BAD'} (< 5 %)")
print(f"5. straight line:            max yaw rate = {max_yaw:.2f} rad/s -> {'GOOD' if max_yaw < 0.3 else 'BAD'} (< 0.3 rad/s)")

print()
print(f"motor + inverter efficiency = {efficiency:.3f} (shaft power / pack power)")
print(f"rpm limit reached at V = {V[np.argmax(rpm_avg > 18900)]:.1f} m/s")
print()
print("SIM TARGET (accel vs speed):")
for lo, hi in zip(SPEED_BINS[:-1], SPEED_BINS[1:]):
    in_bin = window & (V > lo) & (V < hi)
    print(f"  {lo}-{hi} m/s:  accel = {np.mean(accel[in_bin]):.3f} g   pack = {np.mean(power[in_bin]):.1f} kW   torque = {np.mean(torque[in_bin]):.1f} Nm")

# plots
fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 10))

axs[0].plot(V[window], accel[window])
axs[0].set_ylabel("accel [g]")

axs[1].plot(V[window], power[window], label="pack (electrical)")
axs[1].plot(V[window], power_mech[window], label="motor shafts")
axs[1].set_ylabel("power [kW]")
axs[1].legend()

axs[2].plot(V[window], torque[window], label="actual")
axs[2].plot(V[window], torque_limit[window], "--", label="limit")
axs[2].set_ylabel("torque, 4 motors [Nm]")
axs[2].legend()

axs[3].plot(V[window], slipFL[window] * 100, label="FL")
axs[3].plot(V[window], slipFR[window] * 100, label="FR")
axs[3].plot(V[window], slipRL[window] * 100, label="RL")
axs[3].plot(V[window], slipRR[window] * 100, label="RR")
axs[3].set_ylim(-5, 20)
axs[3].set_ylabel("slip [%]")
axs[3].set_xlabel("V [m/s]")
axs[3].legend()

for ax in axs:
    ax.grid(True)
fig.suptitle("Accel run, full throttle window, vs speed")
plt.tight_layout()
plt.show()

#-------------------------------------------------------------
from pathlib import Path
from CarProperties import MFE27
from Tire import Tire
from Acceleration import AccelSolver

LAPSIM_ROOT = Path(__file__).resolve().parents[0]
TIRE_DIR = LAPSIM_ROOT / "1. Data" / "Tire"

car = MFE27()
tire = Tire(TIRE_DIR / "MF61_Parameters.mat", TIRE_DIR / "FZ_Reference.mat", muxScale=0.451, muyScale=0.59)

accel = AccelSolver().simulate(tire, car)
print(accel.keys())