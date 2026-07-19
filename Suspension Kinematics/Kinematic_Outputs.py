# Author: Ludih
# Summary: This script defines the suspension key metrics.

import numpy as np
from numpy.linalg import norm
import Geometric_Functions as gf

# ── Instant center / roll center ─────────────────────────────────────────────

def instant_center_front(result, fr):
    """
    Front-view instant center: intersection of the upper wishbone plane,
    the lower wishbone plane, and the vertical fore-aft slice plane through
    the contact patch (constant x).

    result = solved corner (needs UBJ, LBJ, contact_patch)
    fr     = chassis frame (needs uw_fore/uw_aft/lw_fore/lw_aft)
    """
    n1, d1 = gf.equation_of_plane(result["UBJ"], fr["uw_fore"], fr["uw_aft"])
    n2, d2 = gf.equation_of_plane(result["LBJ"], fr["lw_fore"], fr["lw_aft"])
    n3, d3 = np.array([1.0, 0.0, 0.0]), result["contact_patch"][0]
    A = np.vstack([n1, n2, n3])
    B = np.array([d1, d2, d3])
    return np.linalg.solve(A, B)

def roll_center(result_L, fr_L, result_R, fr_R):
    """
    Roll center of one axle: intersect the two contact-patch -> instant-center
    lines in the y-z plane. x is taken as the mean of the two contact patches.
    """
    IC_L = instant_center_front(result_L, fr_L)
    IC_R = instant_center_front(result_R, fr_R)
    CP_L = result_L["contact_patch"]
    CP_R = result_R["contact_patch"]

    # CP_L + t*(IC_L-CP_L) = CP_R + s*(IC_R-CP_R), solved in (y, z)
    dL = (IC_L - CP_L)[1:3]
    dR = (IC_R - CP_R)[1:3]
    A  = np.column_stack([dL, -dR])
    B  = (CP_R - CP_L)[1:3]
    t, _ = np.linalg.solve(A, B)

    yz = CP_L[1:3] + t * dL
    x  = 0.5 * (CP_L[0] + CP_R[0])
    return np.array([x, yz[0], yz[1]])

# ── Axle outputs ─────────────────────────────────────────────────────────────

def compute_outputs(result_L, result_R, result_L_static, result_R_static,
                    c_L, c_R, h_bump, fr_L=None, fr_R=None):
    """
    Compute suspension parameters from solved points for a full axle.

    result_L / result_R               = current step (left and right)
    result_L_static / result_R_static = ride height reference
    c_L / c_R                         = corner instances
    h_bump                            = chassis heave (m), positive = bump
    fr_L / fr_R                       = chassis frames (optional). If given,
                                        the axle roll center is included.
    """

    def corner_outputs(result, result_static, c):

        # ── Wheel plane ──────────────────────────────────────────
        # Four upright points define the wheel plane.
        # fore-aft and top-bottom vectors span the plane.
        # Their cross product gives the wheel spin axis (normal).
        fore_aft     = result["fore_wheel"] - result["aft_wheel"]
        top_bot      = result["upper_wheel"] - result["contact_patch"]
        wheel_normal = np.cross(top_bot, fore_aft)
        wheel_normal = wheel_normal / norm(wheel_normal)

        # Flip the right side so the normal points outboard on BOTH sides.
        right = c.lower_ball_joint[1] < 0
        if right:
            wheel_normal = -wheel_normal

        # ── Camber ───────────────────────────────────────────────
        # Angle between wheel spin axis and horizontal plane.
        # Positive = top of wheel leaning outboard.
        camber = -np.degrees(np.arcsin(wheel_normal[2]))

        # ── Toe ──────────────────────────────────────────────────
        # Angle of the wheel plane vs straight ahead, from the outboard
        # normal. Using |n_y| makes the convention side-symmetric:
        # positive = toe in, on BOTH sides (no 180-deg wrap on the right).
        toe = np.degrees(np.arctan2(wheel_normal[0], abs(wheel_normal[1])))

        # ── Wheel (steer) angle ──────────────────────────────────
        # Same magnitude as toe but signed consistently across the car:
        # positive = wheel pointing toward +y (left). Both wheels read the
        # same sign when steered together — this is what Ackermann uses.
        wheel_angle = -toe if not right else toe

        # ── Kingpin axis ─────────────────────────────────────────
        kingpin_vec = result["UBJ"] - result["LBJ"]

        # ── Caster ───────────────────────────────────────────────
        # Angle of kingpin axis in the XZ plane.
        # Positive = top of kingpin leaning rearward.
        caster = -np.degrees(np.arctan2(kingpin_vec[0], kingpin_vec[2]))

        # ── KPI (Kingpin Inclination) ────────────────────────────
        # Angle of kingpin axis in the YZ plane, signed so that the top of
        # the kingpin leaning INBOARD is positive on both sides.
        kpi = -np.degrees(np.arctan2(kingpin_vec[1], kingpin_vec[2]))
        if right:
            kpi = -kpi

        # ── Kingpin ground intersection ──────────────────────────
        t              = -result["LBJ"][2] / kingpin_vec[2]
        kingpin_ground = result["LBJ"] + t * kingpin_vec

        # ── Scrub radius ─────────────────────────────────────────
        # Lateral distance between contact patch and kingpin axis at the
        # ground, signed so contact patch OUTBOARD of the kingpin is
        # positive on both sides.
        scrub = (result["contact_patch"][1] - kingpin_ground[1]) * 1000
        if right:
            scrub = -scrub

        # ── Mechanical trail ─────────────────────────────────────
        # Longitudinal distance between the kingpin ground intersection and
        # the contact patch. Positive = kingpin hits the ground AHEAD of
        # the contact patch (stabilising).
        trail = (kingpin_ground[0] - result["contact_patch"][0]) * 1000

        # ── Contact patch migration ──────────────────────────────
        cp_x = (result["contact_patch"][0] - result_static["contact_patch"][0]) * 1000
        cp_y = (result["contact_patch"][1] - result_static["contact_patch"][1]) * 1000

        return {
            "camber"     : camber,
            "toe"        : toe,
            "wheel_angle": wheel_angle,
            "caster"     : caster,
            "kpi"        : kpi,
            "scrub"      : scrub,
            "trail"      : trail,
            "cp_x"       : cp_x,
            "cp_y"       : cp_y,
        }

    out_L = corner_outputs(result_L, result_L_static, c_L)
    out_R = corner_outputs(result_R, result_R_static, c_R)

    # ── Damper lengths ───────────────────────────────────────────
    # Dampers span across the axle between left and right rocker arms.
    # Only the lengths are computed here; the motion ratios are
    # INSTANTANEOUS (step-to-step) quantities, so they are computed in
    # the sweeps (sweep_heave / sweep_roll), where consecutive solved
    # poses are available.
    heave_len = norm(result_L["heave_damper"] - result_R["heave_damper"])
    roll_len  = norm(result_L["roll_damper"]  - result_R["roll_damper"])

    out = {
        "L"           : out_L,
        "R"           : out_R,
        "wheel_travel": h_bump,
        "heave_damper_len": heave_len,
        "roll_damper_len" : roll_len,
    }

    # ── Roll center (needs chassis frames) ───────────────────────
    if fr_L is not None and fr_R is not None:
        rc = roll_center(result_L, fr_L, result_R, fr_R)
        out["rc"]         = rc
        out["rc_height"]  = rc[2] * 1000   # mm
        out["rc_lateral"] = rc[1] * 1000   # mm

    return out

# ── Ackermann ────────────────────────────────────────────────────────────────

def ackermann_percentage(wheel_angle_L, wheel_angle_R, wheelbase, track):
    """
    Ackermann percentage for one steer state, from the two front wheel
    angles (deg, positive = wheel pointing toward +y / left).

      100 %  = perfect Ackermann (inner wheel at the ideal angle for a
               common turn centre on the rear axle line)
        0 %  = parallel steer (both wheels at the same angle)
      < 0 %  = anti-Ackermann (inner wheel turns LESS than the outer)

    Ideal inner angle from the outer:  cot(d_in) = cot(d_out) - track/wheelbase
    Percentage = (d_in - d_out) / (d_in_ideal - d_out) * 100

    Returns np.nan below ~0.1 deg of mean steer, where the ratio is
    numerically meaningless.
    """
    mean = 0.5 * (wheel_angle_L + wheel_angle_R)
    if abs(mean) < 0.1:
        return np.nan

    # Turning left (+) -> left wheel is inner; turning right -> right is inner.
    if mean > 0:
        d_in, d_out = abs(wheel_angle_L), abs(wheel_angle_R)
    else:
        d_in, d_out = abs(wheel_angle_R), abs(wheel_angle_L)

    d_out_rad   = np.radians(d_out)
    d_in_ideal  = np.degrees(np.arctan(1.0 / (1.0 / np.tan(d_out_rad) - track / wheelbase)))

    denom = d_in_ideal - d_out
    if abs(denom) < 1e-12:
        return np.nan
    return (d_in - d_out) / denom * 100.0