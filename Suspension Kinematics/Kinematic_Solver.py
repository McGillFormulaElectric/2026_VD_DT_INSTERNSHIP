# Author: Ludih
# Summary: This code solves the kinematics of the suspension.

import numpy as np
from scipy.optimize import brentq
import Three_sphere_intersection as tsi
from Kinematic_Outputs import compute_outputs

# ── Circle parametrization ───────────────────────────────────────────────────



# ── Circle parametrization ───────────────────────────────────────────────────

def build_lbj_circle(c):
    """
    Compute the circle that LBJ sweeps as the lower wishbone
    rotates about its pivot axis. Called once per corner at startup.
    """
    fore = c.lower_wishbone_fore
    aft  = c.lower_wishbone_aft
    lbj  = c.lower_ball_joint

    axis   = aft - fore
    axis   = axis / np.linalg.norm(axis) #Unit vector along the axis of rotation

    v      = lbj - fore
    center = fore + np.dot(v, axis) * axis

    radius = np.linalg.norm(lbj - center)
    u      = (lbj - center) / radius
    v      = np.cross(axis, u)

    return {"center": center, "radius": radius, "u": u, "v": v}

def lbj_at(theta, circle):
    """Return LBJ position at rotation angle theta."""
    return (circle["center"]
            + circle["radius"] * (np.cos(theta) * circle["u"]
                                + np.sin(theta) * circle["v"]))

# ── Helpers ──────────────────────────────────────────────────────────────────

def heave_chassis(c, h_bump):
    """Shift all chassis-fixed points down by h_bump."""
    heave = np.array([0, 0, h_bump])
    return {
        "lw_fore" : c.lower_wishbone_fore - heave,
        "lw_aft"  : c.lower_wishbone_aft  - heave,
        "uw_fore" : c.upper_wishbone_fore  - heave,
        "uw_aft"  : c.upper_wishbone_aft   - heave,
        "tr_chas" : c.tie_rod_chassis      - heave,
        "bc_fore" : c.bc_fore_mount        - heave,
        "bc_aft"  : c.bc_aft_mount         - heave,
    }

def heave_circle(circle, h_bump):
    """Shift the LBJ circle center down by h_bump."""
    heave = np.array([0, 0, h_bump])
    return {
        "center": circle["center"] - heave,
        "radius": circle["radius"],
        "u"     : circle["u"],
        "v"     : circle["v"],
    }

def pick_root(candidates, reference):
    """Pick the trilateration candidate closest to a reference point."""
    d1 = np.linalg.norm(reference - candidates[0])
    d2 = np.linalg.norm(reference - candidates[1])
    return candidates[0] if d1 < d2 else candidates[1]

# ── Residual ─────────────────────────────────────────────────────────────────

def residual(theta, circle, c, L, h_bump):
    """Returns contact patch z — brentq drives this to zero."""
    ch  = heave_chassis(c, h_bump)
    hc  = heave_circle(circle, h_bump)
    LBJ = lbj_at(theta, hc)

    UBJ = pick_root(
        tsi.three_sphere_intersection(
            LBJ,       ch["uw_fore"], ch["uw_aft"],
            L.knuckle, L.upper_fore,  L.upper_aft),
        c.upper_ball_joint)

    TRO = pick_root(
        tsi.three_sphere_intersection(
            ch["tr_chas"],      UBJ,                  LBJ,
            L.tie_rod,          L.upper_steering_arm,  L.lower_steering_arm),
        c.tie_rod_outer)

    CP = pick_root(
        tsi.three_sphere_intersection(
            TRO,                      UBJ,               LBJ,
            L.out_tierod2lower_wheel, L.ubj2lower_wheel,  L.lbj2lower_wheel),
        c.contact_patch)

    return CP[2]

# ── Full solve chain ─────────────────────────────────────────────────────────

def solve_corner(c, L, circle, h_bump):
    """
    Solve all suspension points at a given chassis heave.
    Returns dict of all point positions.
    """
    if h_bump == 0:
        theta = 0.0
    else:
        theta = brentq(
            residual,
            a=-np.pi/4,
            b= np.pi/4,
            args=(circle, c, L, h_bump),
            xtol=1e-10
        )

    ch  = heave_chassis(c, h_bump)
    hc  = heave_circle(circle, h_bump)
    LBJ = lbj_at(theta, hc)

    UBJ = pick_root(
        tsi.three_sphere_intersection(
            LBJ,       ch["uw_fore"], ch["uw_aft"],
            L.knuckle, L.upper_fore,  L.upper_aft),
        c.upper_ball_joint)

    TRO = pick_root(
        tsi.three_sphere_intersection(
            ch["tr_chas"],      UBJ,                  LBJ,
            L.tie_rod,          L.upper_steering_arm,  L.lower_steering_arm),
        c.tie_rod_outer)

    CP = pick_root(
        tsi.three_sphere_intersection(
            TRO,                      UBJ,               LBJ,
            L.out_tierod2lower_wheel, L.ubj2lower_wheel,  L.lbj2lower_wheel),
        c.contact_patch)

    upper_wheel = pick_root(
        tsi.three_sphere_intersection(
            TRO,                      UBJ,               LBJ,
            L.out_tierod2upper_wheel, L.ubj2upper_wheel,  L.lbj2upper_wheel),
        c.upper_wheel_pt)

    fore_wheel = pick_root(
        tsi.three_sphere_intersection(
            TRO,                     UBJ,                    LBJ,
            L.out_tierod2fore_wheel, L.upper_out2fore_wheel,  L.lower_out2fore_wheel),
        c.fore_wheel_pt)

    aft_wheel = pick_root(
        tsi.three_sphere_intersection(
            TRO,                    UBJ,                   LBJ,
            L.out_tierod2aft_wheel, L.upper_out2aft_wheel,  L.lower_out2aft_wheel),
        c.aft_wheel_pt)

    pushrod_outboard = pick_root(
        tsi.three_sphere_intersection(
            UBJ,                 ch["uw_fore"],                ch["uw_aft"],
            L.ubj2lower_pushrod, L.fore_wishbone2lower_pushrod, L.aft_wishbone2lower_pushrod),
        c.pushrod_outboard)

    pushrod_inboard = pick_root(
        tsi.three_sphere_intersection(
            pushrod_outboard, ch["bc_fore"],           ch["bc_aft"],
            L.pushrod,        L.bc_fore2upper_pushrod,  L.bc_aft2upper_pushrod),
        c.pushrod_inboard)

    heave_damper = pick_root(
        tsi.three_sphere_intersection(
            pushrod_inboard,              ch["bc_fore"],          ch["bc_aft"],
            L.upper_pushrod2heave_damper, L.bc_fore2heave_damper,  L.bc_aft2heave_damper),
        c.heave_damper)

    roll_damper = pick_root(
        tsi.three_sphere_intersection(
            pushrod_inboard,             ch["bc_fore"],         ch["bc_aft"],
            L.upper_pushrod2roll_damper, L.bc_fore2roll_damper,  L.bc_aft2roll_damper),
        c.roll_damper)

    return {
        "LBJ"             : LBJ,
        "UBJ"             : UBJ,
        "TRO"             : TRO,
        "contact_patch"   : CP,
        "upper_wheel"     : upper_wheel,
        "fore_wheel"      : fore_wheel,
        "aft_wheel"       : aft_wheel,
        "pushrod_outboard": pushrod_outboard,
        "pushrod_inboard" : pushrod_inboard,
        "heave_damper"    : heave_damper,
        "roll_damper"     : roll_damper,
    }

# ── Sweep ────────────────────────────────────────────────────────────────────

def sweep_heave(c_L, c_R, L, bump_range=np.linspace(-0.0125, 0.040, 50)):
    """
    Sweep chassis heave over a range for a full axle.

    c_L, c_R = left and right corner instances
    L        = Lengths instance (same for both sides)
    """
    circle_L = build_lbj_circle(c_L)
    circle_R = build_lbj_circle(c_R)

    result_L_static = solve_corner(c_L, L, circle_L, h_bump=0)
    result_R_static = solve_corner(c_R, L, circle_R, h_bump=0)

    results = []
    for h_bump in bump_range:
        result_L = solve_corner(c_L, L, circle_L, h_bump)
        result_R = solve_corner(c_R, L, circle_R, h_bump)
        outputs  = compute_outputs(result_L, result_R,
                                   result_L_static, result_R_static,
                                   c_L, c_R, h_bump)
        outputs["h_bump"] = h_bump
        results.append(outputs)

    return results