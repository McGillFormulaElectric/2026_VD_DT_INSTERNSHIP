# Author: Ludih
# Summary: 3D animation of the suspension mechanism through wheel travel.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import sys
sys.path.insert(0, '/mnt/user-data/uploads')
sys.path.insert(0, '/home/claude')
from Suspension_Geometry import FL, FR, F_len
from Kinematic_Solver import build_lbj_circle, solve_corner

# ── Solve all frames ─────────────────────────────────────────────────────────
bump_range   = np.linspace(-0.025, 0.025, 60)
circle_L     = build_lbj_circle(FL)
circle_R     = build_lbj_circle(FR)

frames_L = [solve_corner(FL, F_len, circle_L, h) for h in bump_range]
frames_R = [solve_corner(FR, F_len, circle_R, h) for h in bump_range]

# ── Chassis points (heaveed per frame) ───────────────────────────────────────
def chassis_points(c, h):
    heave = np.array([0, 0, h])
    return {
        "lw_fore": c.lower_wishbone_fore - heave,
        "lw_aft" : c.lower_wishbone_aft  - heave,
        "uw_fore": c.upper_wishbone_fore  - heave,
        "uw_aft" : c.upper_wishbone_aft   - heave,
        "tr_chas": c.tie_rod_chassis      - heave,
        "bc_fore": c.bc_fore_mount        - heave,
        "bc_aft" : c.bc_aft_mount         - heave,
    }

# ── Figure setup ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor("#0f0f0f")
ax  = fig.add_subplot(111, projection="3d")
ax.set_facecolor("#0f0f0f")

# Axis limits — set once from static geometry
all_pts = [
    FL.lower_wishbone_fore, FL.lower_wishbone_aft, FL.lower_ball_joint,
    FL.upper_wishbone_fore, FL.upper_wishbone_aft, FL.upper_ball_joint,
    FL.tie_rod_chassis,     FL.tie_rod_outer,
    FL.upper_wheel_pt,      FL.contact_patch,
    FL.fore_wheel_pt,       FL.aft_wheel_pt,
    FL.pushrod_outboard,    FL.pushrod_inboard,
    FL.bc_fore_mount,       FL.bc_aft_mount,
    FL.heave_damper,        FL.roll_damper,
    FR.lower_wishbone_fore, FR.lower_wishbone_aft, FR.lower_ball_joint,
    FR.upper_wishbone_fore, FR.upper_wishbone_aft, FR.upper_ball_joint,
    FR.tie_rod_chassis,     FR.tie_rod_outer,
    FR.upper_wheel_pt,      FR.contact_patch,
    FR.fore_wheel_pt,       FR.aft_wheel_pt,
    FR.pushrod_outboard,    FR.pushrod_inboard,
    FR.bc_fore_mount,       FR.bc_aft_mount,
    FR.heave_damper,        FR.roll_damper,
]
all_pts = np.array(all_pts)
pad = 0.05
ax.set_xlim(all_pts[:,0].min()-pad, all_pts[:,0].max()+pad)
ax.set_ylim(all_pts[:,1].min()-pad, all_pts[:,1].max()+pad)
ax.set_zlim(-0.05, all_pts[:,2].max()+pad)

ax.set_xlabel("X (m)", color="#888888", fontsize=8)
ax.set_ylabel("Y (m)", color="#888888", fontsize=8)
ax.set_zlabel("Z (m)", color="#888888", fontsize=8)
ax.tick_params(colors="#555555", labelsize=7)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor("#222222")
ax.yaxis.pane.set_edgecolor("#222222")
ax.zaxis.pane.set_edgecolor("#222222")
ax.grid(True, color="#1a1a1a", linewidth=0.5)

title = ax.set_title("Front Axle Kinematics — h_bump = 0.0 mm",
                      color="#e0e0e0", fontsize=11, fontweight="bold", pad=10)

# Colour palette
c_lower  = "#00c8ff"   # lower wishbone
c_upper  = "#ff6b00"   # upper wishbone
c_upright= "#ffffff"   # upright / knuckle
c_tie    = "#ff2d55"   # tie rod
c_push   = "#a8ff3e"   # pushrod
c_rocker = "#ffdd00"   # rocker
c_ground = "#444444"   # ground plane

# ── Ground plane ─────────────────────────────────────────────────────────────
gx = np.linspace(all_pts[:,0].min()-pad, all_pts[:,0].max()+pad, 2)
gy = np.linspace(all_pts[:,1].min()-pad, all_pts[:,1].max()+pad, 2)
gx, gy = np.meshgrid(gx, gy)
gz = np.zeros_like(gx)
ax.plot_surface(gx, gy, gz, alpha=0.08, color=c_ground, zorder=0)

# ── Contact patch trace ───────────────────────────────────────────────────────
cp_trace_L = np.array([f["contact_patch"] for f in frames_L])
cp_trace_R = np.array([f["contact_patch"] for f in frames_R])
ax.plot(cp_trace_L[:,0], cp_trace_L[:,1], cp_trace_L[:,2],
        color=c_upright, linewidth=0.8, alpha=0.3, linestyle="--")
ax.plot(cp_trace_R[:,0], cp_trace_R[:,1], cp_trace_R[:,2],
        color=c_upright, linewidth=0.8, alpha=0.3, linestyle="--")

# ── Line objects (to be updated each frame) ───────────────────────────────────
def make_line(color, lw=2, alpha=1.0):
    line, = ax.plot([], [], [], color=color, linewidth=lw, alpha=alpha)
    return line

def make_dot(color, size=40):
    dot = ax.scatter([], [], [], color=color, s=size, zorder=5)
    return dot

# Per side: [lower_wb_fore, lower_wb_aft, upper_wb_fore, upper_wb_aft,
#            knuckle, tie_rod, pushrod, rocker_fore, rocker_aft,
#            upright_top, upright_fore, upright_aft, wheel_rim]
def make_side_lines():
    return {
        "lw_fore"     : make_line(c_lower,   lw=2.5),
        "lw_aft"      : make_line(c_lower,   lw=2.5),
        "lw_spread"   : make_line(c_lower,   lw=1,   alpha=0.4),
        "uw_fore"     : make_line(c_upper,   lw=2.5),
        "uw_aft"      : make_line(c_upper,   lw=2.5),
        "uw_spread"   : make_line(c_upper,   lw=1,   alpha=0.4),
        "knuckle"     : make_line(c_upright, lw=2),
        "tie_rod"     : make_line(c_tie,     lw=2),
        "pushrod"     : make_line(c_push,    lw=1.5),
        "rocker_fore" : make_line(c_rocker,  lw=2),
        "rocker_aft"  : make_line(c_rocker,  lw=2),
        "rocker_damp" : make_line(c_rocker,  lw=1,   alpha=0.5),
        "upright_fore": make_line(c_upright, lw=1,   alpha=0.5),
        "upright_aft" : make_line(c_upright, lw=1,   alpha=0.5),
        "upright_top" : make_line(c_upright, lw=1,   alpha=0.5),
        "cp_dot"      : make_line(c_upright, lw=3),
    }

lines_L = make_side_lines()
lines_R = make_side_lines()

# Heave damper spanning left to right
heave_line, = ax.plot([], [], [], color=c_rocker, linewidth=2.5, linestyle="-")
roll_line,  = ax.plot([], [], [], color="#ff9500", linewidth=2.5, linestyle="-")

def seg(line, p1, p2):
    """Update a line object to draw segment from p1 to p2."""
    line.set_data_3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])

def draw_side(lines, frame, ch):
    r = frame
    seg(lines["lw_fore"],    ch["lw_fore"],       r["LBJ"])
    seg(lines["lw_aft"],     ch["lw_aft"],        r["LBJ"])
    seg(lines["lw_spread"],  ch["lw_fore"],       ch["lw_aft"])
    seg(lines["uw_fore"],    ch["uw_fore"],       r["UBJ"])
    seg(lines["uw_aft"],     ch["uw_aft"],        r["UBJ"])
    seg(lines["uw_spread"],  ch["uw_fore"],       ch["uw_aft"])
    seg(lines["knuckle"],    r["LBJ"],            r["UBJ"])
    seg(lines["tie_rod"],    ch["tr_chas"],       r["TRO"])
    seg(lines["pushrod"],    r["pushrod_outboard"], r["pushrod_inboard"])
    seg(lines["rocker_fore"],ch["bc_fore"],       r["pushrod_inboard"])
    seg(lines["rocker_aft"], ch["bc_aft"],        r["pushrod_inboard"])
    seg(lines["rocker_damp"],r["pushrod_inboard"],r["heave_damper"])
    seg(lines["upright_fore"],r["LBJ"],           r["fore_wheel"])
    seg(lines["upright_aft"], r["LBJ"],           r["aft_wheel"])
    seg(lines["upright_top"], r["LBJ"],           r["upper_wheel"])
    seg(lines["cp_dot"],      r["contact_patch"], r["contact_patch"] + np.array([0,0,0.001]))

def animate(i):
    h = bump_range[i]
    ch_L = chassis_points(FL, h)
    ch_R = chassis_points(FR, h)

    draw_side(lines_L, frames_L[i], ch_L)
    draw_side(lines_R, frames_R[i], ch_R)

    seg(heave_line, frames_L[i]["heave_damper"], frames_R[i]["heave_damper"])
    seg(roll_line,  frames_L[i]["roll_damper"],  frames_R[i]["roll_damper"])

    title.set_text(f"Front Axle Kinematics — h_bump = {h*1000:+.1f} mm")
    return list(lines_L.values()) + list(lines_R.values()) + [heave_line, roll_line, title]

anim = animation.FuncAnimation(
    fig, animate,
    frames=len(bump_range),
    interval=50,
    blit=False
)

plt.tight_layout()

print("Saving animation...")
anim.save("suspension_animation.gif",
          writer="pillow",
          fps=20,
          dpi=120)
print("Saved suspension_animation.gif")
plt.close()
