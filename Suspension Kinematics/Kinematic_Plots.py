# Author: Ludih
# Summary: Plots suspension kinematics.
#
# HOW TO USE:
#   Every graph in section 3 is ONE self-contained kplot(...) call.
#   Comment out the ones you don't want. That's it.
#
#   Per-corner outputs ("L"/"R"): camber, toe, wheel_angle, caster, kpi,
#                                 scrub, trail, cp_x, cp_y
#   Per-axle outputs:             rc_height, rc_lateral, ackermann,
#                                 heave_damper_len, roll_damper_len

import numpy as np
import matplotlib.pyplot as plt
from Suspension_Geometry import (FL, FR, RL, RR, F_len, R_len,
                                 wheelbase, track_front)
from Kinematic_Solver import sweep_heave, sweep_steer, sweep_roll

# ════════════════════════ 1. RUN THE SWEEPS ════════════════════════

print("Solving heave sweep (front)...")
heave_f = sweep_heave(FL, FR, F_len,bump_range=np.linspace(-0.0125, 0.040, 50))   # metres
print("Solving heave sweep (rear)...")
heave_r = sweep_heave(RL, RR, R_len,bump_range=np.linspace(-0.0125, 0.040, 50))   # metres

print("Solving steer sweep...")
steer   = sweep_steer(FL, FR, F_len, rack_range=np.linspace(-0.025, 0.025, 50),    # metres
                      wheelbase=wheelbase, track=track_front)

print("Solving roll sweep (marching)...")
roll = sweep_roll(FL, FR, RL, RR, F_len, R_len,
                  roll_range=(np.radians(-2.0), np.radians(2.0)), steps=50)

# x-axes
x_heave_f = [r["h_bump"]      * 1000 for r in heave_f]   # mm
x_heave_r = [r["h_bump"]      * 1000 for r in heave_r]   # mm
x_steer   = [r["rack_travel"] * 1000 for r in steer]     # mm
x_roll    = [r["roll_deg"]           for r in roll]      # deg

# value extractors:  y(results, key)  /  y(results, key, side or axle)
def y(rows, key, *path):
    out = []
    for r in rows:
        v = r
        for p in path:
            v = v[p]
        out.append(v[key])
    return out

# ════════════════════════ 2. PLOT HELPER ═══════════════════════════

def kplot(title, xlabel, ylabel, series):
    """series = list of (label, x, y). One figure per call."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.grid(True, which="both", alpha=0.3)
    ax.minorticks_on()
    for label, xs, ys in series:
        ax.plot(xs, ys, label=label)
    ax.legend()
    plt.tight_layout()

# ═══════════ 3. THE PLOTS — comment out what you don't need ════════

# ── Heave ──────────────────────────────────────────────────────────

kplot("Camber vs Heave", "Wheel travel (mm)", "deg", [
    ("FL", x_heave_f, y(heave_f, "camber", "L")),
    ("RL", x_heave_r, y(heave_r, "camber", "L")),
])

kplot("Toe vs Heave (bump steer)", "Wheel travel (mm)", "deg", [
    ("FL", x_heave_f, y(heave_f, "toe", "L")),
    ("RL", x_heave_r, y(heave_r, "toe", "L")),
])

kplot("Caster vs Heave", "Wheel travel (mm)", "deg", [
    ("FL", x_heave_f, y(heave_f, "caster", "L")),
    ("RL", x_heave_r, y(heave_r, "caster", "L")),
])

kplot("KPI vs Heave", "Wheel travel (mm)", "deg", [
    ("FL", x_heave_f, y(heave_f, "kpi", "L")),
    ("RL", x_heave_r, y(heave_r, "kpi", "L")),
])

kplot("Scrub Radius vs Heave", "Wheel travel (mm)", "mm", [
    ("FL", x_heave_f, y(heave_f, "scrub", "L")),
    ("RL", x_heave_r, y(heave_r, "scrub", "L")),
])

kplot("Mechanical Trail vs Heave", "Wheel travel (mm)", "mm", [
    ("FL", x_heave_f, y(heave_f, "trail", "L")),
    ("RL", x_heave_r, y(heave_r, "trail", "L")),
])

kplot("Roll Center Height vs Heave", "Wheel travel (mm)", "mm", [
    ("Front", x_heave_f, y(heave_f, "rc_height")),
    ("Rear",  x_heave_r, y(heave_r, "rc_height")),
])

kplot("Contact Patch Lateral Migration vs Heave", "Wheel travel (mm)", "mm", [
    ("FL", x_heave_f, y(heave_f, "cp_y", "L")),
    ("RL", x_heave_r, y(heave_r, "cp_y", "L")),
])

kplot("Heave Motion Ratio vs Heave", "Wheel travel (mm)", "MR (wheel / damper)", [
    ("Front", x_heave_f, y(heave_f, "heave_MR")),
    ("Rear",  x_heave_r, y(heave_r, "heave_MR")),
])

# ── Steer ──────────────────────────────────────────────────────────

kplot("Wheel Angle vs Rack Travel", "Rack travel (mm)", "deg", [
    ("FL", x_steer, y(steer, "wheel_angle", "L")),
    ("FR", x_steer, y(steer, "wheel_angle", "R")),
])

kplot("Ackermann vs Rack Travel", "Rack travel (mm)", "%", [
    ("Front", x_steer, y(steer, "ackermann")),
])

kplot("Camber vs Rack Travel (steer camber)", "Rack travel (mm)", "deg", [
    ("FL", x_steer, y(steer, "camber", "L")),
    ("FR", x_steer, y(steer, "camber", "R")),
])

# ── Roll (left AND right differ here — both plotted) ──────────────

kplot("Camber vs Roll", "Roll (deg)", "deg", [
    ("FL", x_roll, y(roll, "camber", "front", "L")),
    ("FR", x_roll, y(roll, "camber", "front", "R")),
    ("RL", x_roll, y(roll, "camber", "rear",  "L")),
    ("RR", x_roll, y(roll, "camber", "rear",  "R")),
])

kplot("Toe vs Roll (roll steer)", "Roll (deg)", "deg", [
    ("FL", x_roll, y(roll, "toe", "front", "L")),
    ("FR", x_roll, y(roll, "toe", "front", "R")),
    ("RL", x_roll, y(roll, "toe", "rear",  "L")),
    ("RR", x_roll, y(roll, "toe", "rear",  "R")),
])

kplot("Roll Center Height vs Roll", "Roll (deg)", "mm", [
    ("Front", x_roll, y(roll, "rc_height", "front")),
    ("Rear",  x_roll, y(roll, "rc_height", "rear")),
])

kplot("Roll Center Lateral Migration vs Roll", "Roll (deg)", "mm", [
    ("Front", x_roll, y(roll, "rc_lateral", "front")),
    ("Rear",  x_roll, y(roll, "rc_lateral", "rear")),
])

kplot("Roll Damper Length vs Roll", "Roll (deg)", "m", [
    ("Front", x_roll, y(roll, "roll_damper_len", "front")),
    ("Rear",  x_roll, y(roll, "roll_damper_len", "rear")),
])

kplot("Roll Motion Ratio vs Roll", "Roll (deg)", "MR (wheel / damper)", [
    ("Front", x_roll, y(roll, "roll_MR", "front")),
    ("Rear",  x_roll, y(roll, "roll_MR", "rear")),
])

plt.show()