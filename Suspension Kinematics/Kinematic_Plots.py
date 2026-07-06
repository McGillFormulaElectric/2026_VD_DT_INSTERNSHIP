# Author: Ludih
# Summary: Plots suspension kinematics outputs (left side only).

import matplotlib.pyplot as plt
from Suspension_Geometry import FL, FR, RL, RR, F_len, R_len
from Kinematic_Solver import sweep_heave

print("Solving front axle...")
front = sweep_heave(FL, FR, F_len)
print("Solving rear axle...")
rear  = sweep_heave(RL, RR, R_len)

wt_f = [r["wheel_travel"] * 1000 for r in front]
wt_r = [r["wheel_travel"] * 1000 for r in rear]

plots = [
    ("Camber",            "deg", "camber",   True),
    ("Toe",               "deg", "toe",      True),
    ("Caster",            "deg", "caster",   True),
    ("KPI",               "deg", "kpi",      True),
    ("Scrub Radius",      "m",   "scrub",    True),
    ("CP Lateral",        "m",   "cp_y",     True),
    ("CP Longitudinal",   "m",   "cp_x",     True),
    ("Heave Motion Ratio","",    "heave_MR", False),
    ("Roll Motion Ratio", "",    "roll_MR",  False),
]

for title, ylabel, key, per_corner in plots:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(title, fontweight="bold")
    ax.set_xlabel("Wheel Travel (mm)")
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.grid(True, alpha=0.3)

    if per_corner:
        ax.plot(wt_f, [r["L"][key] for r in front], label="FL")
        ax.plot(wt_r, [r["L"][key] for r in rear],  label="RL")
    else:
        ax.plot(wt_f, [r[key] for r in front], label="Front")
        ax.plot(wt_r, [r[key] for r in rear],  label="Rear")

    ax.legend()
    plt.tight_layout()

plt.show()