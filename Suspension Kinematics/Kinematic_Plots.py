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
    """
    Extract one metric from every sweep row into a flat list (a y-axis
    column for plotting).

    Sweeps return lists of nested dicts, and how deep a metric sits
    varies: heave/steer rows are one axle ("camber" lives under "L" or
    "R"), roll rows hold both axles ("camber" lives under "front" ->
    "L"), and some metrics sit at the top level ("rc_height", no path).
    This helper walks an arbitrary chain of keys down to the right
    sub-dict, then grabs the metric.

    Args:
        rows (list of dict): output of a sweep (heave_f, steer, roll...).
        key (str): the metric to extract, e.g. "camber", "rc_height".
        *path: zero or more dict keys to descend through first, in
            order, e.g. ("L",) or ("front", "L").

    Returns:
        list: the metric's value from each row, in row order.

    Examples:
        y(heave_f, "rc_height")            # r["rc_height"]
        y(heave_f, "camber", "L")          # r["L"]["camber"]
        y(roll,    "camber", "front", "L") # r["front"]["L"]["camber"]
    """
    out = []
    for r in rows:
        v = r                    # start at the top of this row's dict
        for p in path:
            v = v[p]             # descend one level per path key
        out.append(v[key])       # grab the metric at the final level
    return out

# ════════════════════════ 2. PLOT HELPER ═══════════════════════════

def kplot(title, xlabel, ylabel, series):
    """
    Create one standardized kinematics figure with any number of curves.

    Wraps the repetitive matplotlib boilerplate (figure creation, axis
    labels, zero reference lines, grid, legend) so every graph in the
    script is a single call and all figures share the same styling.

    Args:
        title (str): figure title.
        xlabel (str): x-axis label, e.g. "Wheel travel (mm)".
        ylabel (str): y-axis label, e.g. "deg".
        series (list of tuple): one (label, x, y) triple per curve —
            label (str) for the legend, x and y as equal-length
            sequences. All curves share the axes.

    Returns:
        None: the figure is created and styled but not shown; the
        script calls plt.show() once at the end to display all figures.
    """
    # New figure + one axes per call (so each graph is its own window)
    fig, ax = plt.subplots(figsize=(8, 5))    # size in inches (w, h)
    fig.suptitle(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Dashed zero reference lines: kinematics curves are read relative
    # to zero (static pose / neutral setting), so mark both axes
    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")   # y = 0
    ax.axvline(0, color="gray", linewidth=0.7, linestyle="--")   # x = 0

    # Grid on both major and minor ticks, faint (alpha = transparency)
    ax.grid(True, which="both", alpha=0.3)
    ax.minorticks_on()      # enable minor ticks so "both" has effect

    # Plot every curve; tuple-unpack each (label, x, y) triple.
    # No explicit colors: matplotlib cycles its defaults, so curves
    # in one figure are automatically distinct.
    for label, xs, ys in series:
        ax.plot(xs, ys, label=label)

    ax.legend()             # built from the label= kwargs above
    plt.tight_layout()      # fix margins so labels don't clip

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