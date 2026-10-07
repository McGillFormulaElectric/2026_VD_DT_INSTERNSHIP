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
V_LOW, V_HIGH = 8.0, 14.0   # m/s, target window, low speed so drag is small

# load brake run
data = MotecData(LAPSIM_ROOT / "1. Data" / "Motec" / "brake.mat")

time = data.getTime("velX")
velX = data.getValue("velX")              # m/s
velY = data.getValue("velY")              # m/s
accX = data.getValue("accX")              # m/s^2, IMU, positive under braking, only a check
yawRate = data.getValue("gyrZ_HR")        # rad/s
throttle = data.getValue("THROTTLE_PERCENT")
rpmFL = data.getValue("SpeedActualFL")    # motor rpm
rpmFR = data.getValue("SpeedActualFR")
rpmRL = data.getValue("SpeedActualRL")
rpmRR = data.getValue("SpeedActualRR")
torqueFL = data.getValue("TorqueActualFL")
torqueFR = data.getValue("TorqueActualFR")
torqueRL = data.getValue("TorqueActualRL")
torqueRR = data.getValue("TorqueActualRR")

dt = np.mean(np.diff(time))

# speed and decel from GPS speed, smoothed over 0.3 s before taking the derivative
V = savgol_filter(np.sqrt(velX**2 + velY**2), 31, 2)
decel = -np.gradient(V, dt) / G           # g, positive when braking

# free rolling reference: motor rpm per m/s while cruising, before the stop
cruise = (throttle > 50) & (np.abs(decel) < 0.1)
ref = np.mean((rpmFL[cruise] + rpmFR[cruise] + rpmRL[cruise] + rpmRR[cruise]) / 4 / V[cruise])

# slip ratio per wheel, negative when braking
slipFL = rpmFL / ref / V - 1
slipFR = rpmFR / ref / V - 1
slipRL = rpmRL / ref / V - 1
slipRR = rpmRR / ref / V - 1

# wheel speed as a ground speed, only to plot, it reads low because the wheels slip
V_wheel = (rpmFL + rpmFR + rpmRL + rpmRR) / 4 / ref

# target window: off throttle, between V_LOW and V_HIGH
window = (throttle < 1) & (V > V_LOW) & (V < V_HIGH)

decel_target = np.mean(decel[window])
V_target = np.mean(V[window])

# second way: straight line fit of speed vs time over the window, slope = decel
decel_fit = -np.polyfit(time[window], V[window], 1)[0] / G

print(f"decel = {decel_target:.3f} g")
print(f"V     = {V_target:.1f} m/s")
print(f"fit   = {decel_fit:.3f} g (check)")
print(f"IMU   = {np.mean(accX[window]) / G:.3f} g (check, reads high from pitch)")
print(f"free rolling ref = {ref:.1f} rpm per m/s")
print(f"median slip  FL {np.median(slipFL[window]) * 100:.1f} %  FR {np.median(slipFR[window]) * 100:.1f} %  "
      f"RL {np.median(slipRL[window]) * 100:.1f} %  RR {np.median(slipRR[window]) * 100:.1f} %")

# checks
fit_diff = abs(decel_target - decel_fit)
max_torque = np.max(np.abs(np.concatenate([torqueFL[window], torqueFR[window], torqueRL[window], torqueRR[window]])))
median_slip = np.median(np.concatenate([slipFL[window], slipFR[window], slipRL[window], slipRR[window]]))
min_slip = np.min(np.concatenate([slipFL[window], slipFR[window], slipRL[window], slipRR[window]]))
max_yaw = np.max(np.abs(yawRate[window]))

print()
print(f"1. fit agrees with average: diff = {fit_diff:.3f} g      -> {'GOOD' if fit_diff < 0.02 else 'BAD'} (< 0.02 g)")
print(f"2. no regen:                max torque = {max_torque:.2f}  -> {'GOOD' if max_torque < 1 else 'BAD'} (< 1)")
print(f"3. at the grip limit:       median slip = {median_slip * 100:.1f} % -> {'GOOD' if -0.20 < median_slip < -0.05 else 'BAD'} (-5 to -20 %)")
print(f"4. no wheel locked:         min slip = {min_slip * 100:.1f} %   -> {'GOOD' if min_slip > -0.5 else 'BAD'} (> -50 %)")
print(f"5. straight line:           max yaw rate = {max_yaw:.2f} rad/s -> {'GOOD' if max_yaw < 0.3 else 'BAD'} (< 0.3 rad/s)")

print()
print(f"SIM TARGET: decel = {decel_target:.3f} g at V = {V_target:.1f} m/s")

# plots
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

axs[0].plot(time, V, label="GPS")
axs[0].plot(time, V_wheel, label="wheels")
axs[0].set_ylabel("V [m/s]")
axs[0].legend()

axs[1].plot(time, decel, label="GPS")
axs[1].plot(time, accX / G, alpha=0.4, label="IMU accX")
axs[1].axhline(decel_target, color="k", linestyle="--")
axs[1].set_ylabel("decel [g]")
axs[1].legend()

axs[2].plot(time, slipFL * 100, label="FL")
axs[2].plot(time, slipFR * 100, label="FR")
axs[2].plot(time, slipRL * 100, label="RL")
axs[2].plot(time, slipRR * 100, label="RR")
axs[2].set_ylim(-40, 10)
axs[2].set_ylabel("slip [%]")
axs[2].set_xlabel("time [s]")
axs[2].legend()

for ax in axs:
    ax.fill_between(time, 0, 1, where=window, transform=ax.get_xaxis_transform(), color="tab:green", alpha=0.15)
    ax.grid(True)
fig.suptitle("Brake run, target window in green")
plt.tight_layout()
plt.show()