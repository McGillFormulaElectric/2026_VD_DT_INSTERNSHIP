# Author: Ludih
# Summary: Script to plot Motor Curves and Efficiency Maps as well as compare two motors

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat

motor_type = "Fisher"  # "AMK" or "Fisher"
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

# Right: efficiency map with envelope overlaid
pc = ax2.pcolormesh(eta_rpm, eta_torque, eta_map,
                    cmap="RdYlGn", shading="gouraud") #Use shading = "nearest" for no blending, "gouraud" for smooth blending
fig.colorbar(pc, ax=ax2, label="Efficiency η")
ax2.plot(curve_rpm, curve_torque, "k-", lw=2, label="Envelope")
ax2.set_xlabel("Motor speed [rpm]")
ax2.set_ylabel("Motor torque [Nm]")
ax2.set_title(f"{motor_type} efficiency map")
ax2.legend(loc="lower right")

plt.tight_layout()
plt.show()