# Author: Ludih
# Summary: Script to plot Motor Curves and Efficiency Maps as well as compare two motors

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat

###########################################

#___________SELECT_A_MOTOR_____________#

motor_type = "AMK"  # "AMK" or "Fisher"

###########################################


MOTOR_DIR = Path(__file__).resolve().parents[2].parent / "Data" / "PowerTrain_mat" / motor_type

# ── Load data ─────────────────────────────────────────────
m = loadmat(MOTOR_DIR / (motor_type + "_Efficiency.mat"))
eta_torque = np.squeeze(m["Torque"]).astype(float)
eta_rpm    = np.squeeze(m["RPM"]).astype(float)
eta_map    = np.asarray(m["Efficiency"], dtype=float)

c = loadmat(MOTOR_DIR / (motor_type + "_MotorCurve.mat"))
curve_rpm    = np.squeeze(c["RPM"]).astype(float)
curve_torque = np.squeeze(c["Tmotor"]).astype(float)

# ── Plot ──────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left: torque envelope
ax1.plot(curve_rpm, curve_torque, "o-", lw=2)
ax1.set_xlabel("Motor speed [rpm]")
ax1.set_ylabel("Motor torque [Nm]")
ax1.set_title(f"{motor_type} torque envelope")
ax1.grid(alpha=0.3)

# Right: efficiency map as filled contours + labelled iso-efficiency lines
eta_pct = eta_map * (100 if eta_map.max() <= 1.5 else 1)

###############################
levels  = np.arange(82, 100.1, 2) #Adjust resolution of colors
###############################

cf = ax2.contourf(eta_rpm, eta_torque, eta_pct, levels=levels, cmap="RdYlGn", extend="min")

###############################
cs = ax2.contour (eta_rpm, eta_torque, eta_pct, levels=levels[::1], colors="k", linewidths=0.6) #Adjust resolution of lines
###############################


ax2.clabel(cs, fmt="%d%%", fontsize=8, inline=True)
fig.colorbar(cf, ax=ax2, label="Efficiency [%]")
ax2.plot(curve_rpm, curve_torque, "k-", lw=2, label="Envelope")

################################
#Adjust Axis Limits
ax2.set_xlim(eta_rpm.min(), curve_rpm.max())
ax2.set_ylim(eta_torque.min(), curve_torque.max())
################################

ax2.set_xlabel("Motor speed [rpm]"); ax2.set_ylabel("Motor torque [Nm]")
ax2.set_title(f"{motor_type} efficiency map"); ax2.legend(loc="lower right")

plt.tight_layout()
plt.show()