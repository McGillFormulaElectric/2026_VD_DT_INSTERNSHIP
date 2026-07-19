# Author: Ludih
# Summary: 3D animation of all four corners through any imposed motion.
#
# HOW TO USE:
#   Every animation in the menu at the BOTTOM is one animate_motion(...) call.
#   Comment out the ones you don't want. Any combination of roll/heave/rack
#   works (the march co-ramps them).
#
#   view      : "iso" | "front" | "side" | "top"
#   symmetric : -target -> +target through zero (else 0 -> target)
#   pingpong  : sweep out and back so the gif loops smoothly
#   preview   : open an interactive window instead of saving a gif

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Suspension_Geometry import FL, FR, RL, RR, F_len, R_len
import Geometric_Functions as gf
from Kinematic_Solver import march, chassis_frame, solve_corner, static_ref


def animate_motion(outfile, roll_deg=0.0, heave_mm=0.0, rack_mm=0.0,
                   steps=60, symmetric=True, pingpong=True, view="iso",
                   fps=20, dpi=120, figsize=(16, 10), preview=False):
    """Solve the motion and render one animation. All units human:
    degrees and millimetres."""
    ROLL, HEAVE, RACK = np.radians(roll_deg), heave_mm/1000.0, rack_mm/1000.0
    def static_row():
        corners = {"FL": FL, "FR": FR, "RL": RL, "RR": RR}
        lens    = {"FL": F_len, "FR": F_len, "RL": R_len, "RR": R_len}
        frames  = {k: chassis_frame(c, gf.build_lbj_circle(c)) for k, c in corners.items()}
        results = {k: solve_corner(frames[k], lens[k], static_ref(c)) for k, c in corners.items()}
        return {"roll": 0.0, "h_bump": 0.0, "rack": 0.0,
                "corners": results, "frames": frames}

    print(f"Solving motion for {outfile}...")
    if symmetric:
        neg = march(FL, FR, RL, RR, F_len, R_len,
                    roll_end=-ROLL, heave_end=-HEAVE, rack_end=-RACK, steps=steps)
        pos = march(FL, FR, RL, RR, F_len, R_len,
                    roll_end= ROLL, heave_end= HEAVE, rack_end= RACK, steps=steps)
        history = list(reversed(neg)) + [static_row()] + pos
    else:
        history = [static_row()] + march(FL, FR, RL, RR, F_len, R_len,
                                         roll_end=ROLL, heave_end=HEAVE,
                                         rack_end=RACK, steps=steps)
    if pingpong:
        history = history + history[-2:0:-1]

    # ════════════════════ FIGURE SETUP ═══════════════════════
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("#0f0f0f")
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0f0f0f")

    all_corners = [FL, FR, RL, RR]
    all_pts = np.array([
        pt for c in all_corners for pt in [
            c.lower_wishbone_fore, c.lower_wishbone_aft, c.lower_ball_joint,
            c.upper_wishbone_fore, c.upper_wishbone_aft, c.upper_ball_joint,
            c.tie_rod_chassis,     c.tie_rod_outer,
            c.upper_wheel_pt,      c.contact_patch,
            c.fore_wheel_pt,       c.aft_wheel_pt,
            c.pushrod_outboard,    c.pushrod_inboard,
            c.bc_fore_mount,       c.bc_aft_mount,
            c.heave_damper,        c.roll_damper,
        ]
    ])

    pad = 0.06
    ax.set_xlim(all_pts[:,0].min()-pad, all_pts[:,0].max()+pad)
    ax.set_ylim(all_pts[:,1].min()-pad, all_pts[:,1].max()+pad)
    ax.set_zlim(-0.05, all_pts[:,2].max()+pad)

    ax.set_xlabel("X (m)", color="#888888", fontsize=8)
    ax.set_ylabel("Y (m)", color="#888888", fontsize=8)
    ax.set_zlabel("Z (m)", color="#888888", fontsize=8)
    ax.tick_params(colors="#555555", labelsize=7)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#222222")
    ax.grid(True, color="#1a1a1a", linewidth=0.5)

    views = {"iso": (20, -60), "front": (0, 0), "side": (0, -90), "top": (90, -90)}
    ax.view_init(*views[view])

    title = ax.set_title("", color="#e0e0e0", fontsize=12, fontweight="bold", pad=12)

    # ── Colours ──────────────────────────────────────────────
    c_lower   = "#00c8ff"
    c_upper   = "#ff6b00"
    c_upright = "#ffffff"
    c_tie     = "#ff2d55"
    c_push    = "#a8ff3e"
    c_rocker  = "#ffdd00"
    c_ground  = "#444444"
    c_heave   = "#ffdd00"
    c_roll    = "#ff9500"
    c_chassis = "#9b6dff"
    c_wheel   = "#dddddd"

    # ── Ground plane ─────────────────────────────────────────
    gx = np.linspace(all_pts[:,0].min()-pad, all_pts[:,0].max()+pad, 2)
    gy = np.linspace(all_pts[:,1].min()-pad, all_pts[:,1].max()+pad, 2)
    gx, gy = np.meshgrid(gx, gy)
    ax.plot_surface(gx, gy, np.zeros_like(gx), alpha=0.06, color=c_ground, zorder=0)

    # ── Contact patch traces ─────────────────────────────────
    for key, color in [("FL", "#00c8ff"), ("FR", "#0072ff"),
                       ("RL", "#ff6b00"), ("RR", "#ff2d55")]:
        cp = np.array([h["corners"][key]["contact_patch"] for h in history])
        ax.plot(cp[:,0], cp[:,1], cp[:,2], color=color,
                linewidth=0.8, alpha=0.25, linestyle="--")

    # ── Line helpers ─────────────────────────────────────────
    def make_line(color, lw=2, alpha=1.0):
        line, = ax.plot([], [], [], color=color, linewidth=lw, alpha=alpha)
        return line

    def seg(line, p1, p2):
        line.set_data_3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])

    def make_side_lines():
        return {
            "lw_fore"     : make_line(c_lower,   lw=2.5),
            "lw_aft"      : make_line(c_lower,   lw=2.5),
            "lw_spread"   : make_line(c_lower,   lw=1,   alpha=0.35),
            "uw_fore"     : make_line(c_upper,   lw=2.5),
            "uw_aft"      : make_line(c_upper,   lw=2.5),
            "uw_spread"   : make_line(c_upper,   lw=1,   alpha=0.35),
            "knuckle"     : make_line(c_upright, lw=2),
            "tie_rod"     : make_line(c_tie,     lw=2),
            "pushrod"     : make_line(c_push,    lw=1.5),
            "rocker_fore" : make_line(c_rocker,  lw=2),
            "rocker_aft"  : make_line(c_rocker,  lw=2),
            "rocker_damp" : make_line(c_rocker,  lw=1,   alpha=0.5),
            "upright_fore": make_line(c_upright, lw=1,   alpha=0.4),
            "upright_aft" : make_line(c_upright, lw=1,   alpha=0.4),
            "upright_top" : make_line(c_upright, lw=1,   alpha=0.4),
            "wheel"       : make_line(c_wheel,   lw=1.5, alpha=0.9),
            "cp_mark"     : make_line(c_wheel,   lw=2,   alpha=0.9),
        }

    lines = {k: make_side_lines() for k in ("FL", "FR", "RL", "RR")}

    front_heave, = ax.plot([], [], [], color=c_heave, linewidth=2.5)
    front_roll,  = ax.plot([], [], [], color=c_roll,  linewidth=2.5)
    rear_heave,  = ax.plot([], [], [], color=c_heave, linewidth=2.5)
    rear_roll,   = ax.plot([], [], [], color=c_roll,  linewidth=2.5)
    rack_line,   = ax.plot([], [], [], color=c_tie,   linewidth=2.5)

    # ── Chassis box: REAL hardpoints only (wishbone inboard mounts) ──
    # side rails per corner + lateral ties at each axle + longitudinal ties.
    CHASSIS_SEGS = [
        # verticals at each mount pair (fore and aft, each corner)
        ("FL", "uw_fore", "FL", "lw_fore"), ("FL", "uw_aft", "FL", "lw_aft"),
        ("FR", "uw_fore", "FR", "lw_fore"), ("FR", "uw_aft", "FR", "lw_aft"),
        ("RL", "uw_fore", "RL", "lw_fore"), ("RL", "uw_aft", "RL", "lw_aft"),
        ("RR", "uw_fore", "RR", "lw_fore"), ("RR", "uw_aft", "RR", "lw_aft"),
        # lateral ties, front axle
        ("FL", "uw_fore", "FR", "uw_fore"), ("FL", "lw_fore", "FR", "lw_fore"),
        ("FL", "uw_aft",  "FR", "uw_aft"),  ("FL", "lw_aft",  "FR", "lw_aft"),
        # lateral ties, rear axle
        ("RL", "uw_fore", "RR", "uw_fore"), ("RL", "lw_fore", "RR", "lw_fore"),
        ("RL", "uw_aft",  "RR", "uw_aft"),  ("RL", "lw_aft",  "RR", "lw_aft"),
        # longitudinal ties (front aft mounts -> rear fore mounts)
        ("FL", "uw_aft", "RL", "uw_fore"), ("FL", "lw_aft", "RL", "lw_fore"),
        ("FR", "uw_aft", "RR", "uw_fore"), ("FR", "lw_aft", "RR", "lw_fore"),
    ]
    chassis_lines = [make_line(c_chassis, lw=1.2, alpha=0.55) for _ in CHASSIS_SEGS]

    # ── Wheel circle from the four solved wheel-plane points ─────────
    _wheel_theta = np.linspace(0, 2*np.pi, 33)

    def wheel_circle(res):
        center = 0.5 * (res["fore_wheel"] + res["aft_wheel"])
        e1     = res["fore_wheel"] - center
        r      = np.linalg.norm(e1)
        e1     = e1 / r
        n      = np.cross(res["upper_wheel"] - res["contact_patch"], e1)
        n      = n / np.linalg.norm(n)
        e2     = np.cross(n, e1)
        return (center[None, :]
                + r * np.cos(_wheel_theta)[:, None] * e1[None, :]
                + r * np.sin(_wheel_theta)[:, None] * e2[None, :])

    def draw_side(L, r, ch):
        seg(L["lw_fore"],     ch["lw_fore"],          r["LBJ"])
        seg(L["lw_aft"],      ch["lw_aft"],           r["LBJ"])
        seg(L["lw_spread"],   ch["lw_fore"],          ch["lw_aft"])
        seg(L["uw_fore"],     ch["uw_fore"],          r["UBJ"])
        seg(L["uw_aft"],      ch["uw_aft"],           r["UBJ"])
        seg(L["uw_spread"],   ch["uw_fore"],          ch["uw_aft"])
        seg(L["knuckle"],     r["LBJ"],               r["UBJ"])
        seg(L["tie_rod"],     ch["tr_chas"],          r["TRO"])
        seg(L["pushrod"],     r["pushrod_outboard"],  r["pushrod_inboard"])
        seg(L["rocker_fore"], ch["bc_fore"],          r["pushrod_inboard"])
        seg(L["rocker_aft"],  ch["bc_aft"],           r["pushrod_inboard"])
        seg(L["rocker_damp"], r["pushrod_inboard"],   r["heave_damper"])
        seg(L["upright_fore"],r["LBJ"],               r["fore_wheel"])
        seg(L["upright_aft"], r["LBJ"],               r["aft_wheel"])
        seg(L["upright_top"], r["LBJ"],               r["upper_wheel"])
        w = wheel_circle(r)
        L["wheel"].set_data_3d(w[:, 0], w[:, 1], w[:, 2])
        cp = r["contact_patch"]
        L["cp_mark"].set_data_3d([cp[0]-0.03, cp[0]+0.03], [cp[1], cp[1]], [0, 0])

    def animate(i):
        h = history[i]
        for k in lines:
            draw_side(lines[k], h["corners"][k], h["frames"][k])

        for line, (kA, pA, kB, pB) in zip(chassis_lines, CHASSIS_SEGS):
            seg(line, h["frames"][kA][pA], h["frames"][kB][pB])

        seg(rack_line, h["frames"]["FL"]["tr_chas"], h["frames"]["FR"]["tr_chas"])

        seg(front_heave, h["corners"]["FL"]["heave_damper"], h["corners"]["FR"]["heave_damper"])
        seg(front_roll,  h["corners"]["FL"]["roll_damper"],  h["corners"]["FR"]["roll_damper"])
        seg(rear_heave,  h["corners"]["RL"]["heave_damper"], h["corners"]["RR"]["heave_damper"])
        seg(rear_roll,   h["corners"]["RL"]["roll_damper"],  h["corners"]["RR"]["roll_damper"])

        title.set_text(f"Full Car Kinematics — "
                       f"roll {np.degrees(h['roll']):+.2f} deg   "
                       f"heave {h['h_bump']*1000:+.1f} mm   "
                       f"rack {h['rack']*1000:+.1f} mm")

        return ([l for side in lines.values() for l in side.values()] +
                chassis_lines +
                [rack_line, front_heave, front_roll, rear_heave, rear_roll, title])

    anim = animation.FuncAnimation(fig, animate, frames=len(history),
                                   interval=1000 / fps, blit=False)


    plt.tight_layout()
    if preview:
        plt.show()
    else:
        print("Saving...")
        anim.save(outfile, writer="pillow", fps=fps, dpi=dpi)
        print(f"Saved {outfile}")
        plt.close(fig)
    return anim


# ════════ THE ANIMATIONS — comment out what you don't need ════════

animate_motion("heave_animation.gif",    heave_mm=25)
animate_motion("roll_animation.gif",     roll_deg=2.0, view="front")
animate_motion("steer_animation.gif",    rack_mm=20,   view="top")
#animate_motion("combined_animation.gif", roll_deg=1.5, heave_mm=10, rack_mm=10)