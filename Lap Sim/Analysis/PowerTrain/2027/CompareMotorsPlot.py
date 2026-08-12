# Author: Ludih 
# Summary: Compare torque curves and efficiency maps between motor types.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator

MOTORS = ["AMK", "Fisher"]           # folders/prefixes under Powertrain_mat
MOTOR_DIR = Path(__file__).resolve().parents[2].parent / "Data" / "PowerTrain_mat"

rdgn_deep = LinearSegmentedColormap.from_list("rdgn_deep",
    ["#a50026", "#d73027", "#f46d43", "#fdae61", "#f5c710",
     "#a6d96a", "#66bd63", "#1a9850", "#006837"])


def load_motor(name):
    """Load one motor's efficiency map and torque envelope into a dict."""
    m = loadmat(MOTOR_DIR / name / f"{name}_Efficiency.mat")
    c = loadmat(MOTOR_DIR / name / f"{name}_MotorCurve.mat")
    eta_map = np.asarray(m["Efficiency"], dtype=float)
    if eta_map.max() > 1.5:          # map stored in percent -> convert to fraction
        eta_map = eta_map / 100.0
    return {
        "name":         name,
        "eta_torque":   np.squeeze(m["Torque"]).astype(float),
        "eta_rpm":      np.squeeze(m["RPM"]).astype(float),
        "eta_map":      eta_map,
        "curve_rpm":    np.squeeze(c["RPM"]).astype(float),
        "curve_torque": np.squeeze(c["Tmotor"]).astype(float),
    }


motors = [load_motor(n) for n in MOTORS]

# shared color scale so the two maps are directly comparable
vmin = min(mo["eta_map"].min() for mo in motors)
vmax = max(mo["eta_map"].max() for mo in motors)

# ── Figure 1: torque curves overlaid ─────────────────────────
fig1, ax = plt.subplots(figsize=(8, 5))
for mo, color in zip(motors, ["#1a9850", "#d73027", "#2166ac"]):
    ax.plot(mo["curve_rpm"], mo["curve_torque"], "o-", lw=2,
            color=color, label=mo["name"])
ax.set_xlabel("Motor speed [rpm]")
ax.set_ylabel("Motor torque [Nm]")
ax.set_title("Torque envelope comparison")
ax.legend()
ax.grid(alpha=0.3)
fig1.tight_layout()

# ── Figure 2: efficiency maps side by side, shared scale ─────
fig2, axes = plt.subplots(1, len(motors), figsize=(6.5 * len(motors), 5))
for axm, mo in zip(np.atleast_1d(axes), motors):
    pc = axm.pcolormesh(mo["eta_rpm"], mo["eta_torque"], mo["eta_map"],
                        cmap=rdgn_deep, shading="gouraud",
                        vmin=vmin, vmax=vmax)
    axm.plot(mo["curve_rpm"], mo["curve_torque"], "k-", lw=2)
    axm.set_xlabel("Motor speed [rpm]")
    axm.set_ylabel("Motor torque [Nm]")
    axm.set_title(f'{mo["name"]} efficiency map')
fig2.colorbar(pc, ax=axes, label="Efficiency η", fraction=0.03)

# ── Figure 3: difference map on a common grid ────────────────
# Only meaningful for exactly two motors.
if len(motors) == 2:
    a, b = motors

    # overlap region where BOTH maps have data (no extrapolation)
    t_lo = max(a["eta_torque"][0],  b["eta_torque"][0])
    t_hi = min(a["eta_torque"][-1], b["eta_torque"][-1])
    r_lo = max(a["eta_rpm"][0],     b["eta_rpm"][0])
    r_hi = min(a["eta_rpm"][-1],    b["eta_rpm"][-1])

    T = np.linspace(t_lo, t_hi, 60)
    R = np.linspace(r_lo, r_hi, 60)
    TT, RR = np.meshgrid(T, R, indexing="ij")
    pts = np.stack([TT.ravel(), RR.ravel()], axis=-1)

    interp_a = RegularGridInterpolator((a["eta_torque"], a["eta_rpm"]), a["eta_map"])
    interp_b = RegularGridInterpolator((b["eta_torque"], b["eta_rpm"]), b["eta_map"])
    diff = (interp_a(pts) - interp_b(pts)).reshape(TT.shape)

    lim = np.abs(diff).max()
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    pc3 = ax3.pcolormesh(R, T, diff, cmap="RdBu_r", shading="gouraud",
                         vmin=-0.3, vmax=0.3)
    fig3.colorbar(pc3, ax=ax3, label=f'η {a["name"]} − η {b["name"]}')
    ax3.set_xlabel("Motor speed [rpm]")
    ax3.set_ylabel("Motor torque [Nm]")
    ax3.set_title(f'Efficiency difference: red = {a["name"]} better, '
                  f'blue = {b["name"]} better')
    fig3.tight_layout()

plt.show()