# Author: Ludih
# Summary: This script defines the suspension key metrics.

import numpy as np
from numpy.linalg import norm


def compute_outputs(result_L, result_R, result_L_static, result_R_static, c_L, c_R, h_bump):
    """
    Compute suspension parameters from solved points for a full axle.

    result_L / result_R               = current step (left and right)
    result_L_static / result_R_static = ride height reference
    c_L / c_R                         = corner instances
    h_bump                            = chassis heave (m), positive = bump
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

        # Right side normal points inboard — flip so convention is consistent
        if c.lower_ball_joint[1] < 0:
            wheel_normal = -wheel_normal

        # ── Camber ───────────────────────────────────────────────
        # Angle between wheel spin axis and horizontal plane.
        # Positive = top of wheel leaning outboard.
        camber = np.degrees(np.arcsin(wheel_normal[2]))

        # ── Toe ──────────────────────────────────────────────────
        # Angle of wheel normal projected onto XY plane vs lateral axis.
        # Positive = toe in.
        toe = np.degrees(np.arctan2(wheel_normal[0], wheel_normal[1]))

        # ── Kingpin axis ─────────────────────────────────────────
        kingpin_vec = result["UBJ"] - result["LBJ"]

        # ── Caster ───────────────────────────────────────────────
        # Angle of kingpin axis in the XZ plane.
        caster = np.degrees(np.arctan2(kingpin_vec[0], kingpin_vec[2]))

        # ── KPI (Kingpin Inclination) ────────────────────────────
        # Angle of kingpin axis in the YZ plane.
        kpi = np.degrees(np.arctan2(kingpin_vec[1], kingpin_vec[2]))

        # ── Scrub radius ─────────────────────────────────────────
        # Lateral distance between contact patch and kingpin axis
        # projected to the ground plane (z=0).
        t              = -result["LBJ"][2] / kingpin_vec[2]
        kingpin_ground = result["LBJ"] + t * kingpin_vec
        scrub          = result["contact_patch"][1] - kingpin_ground[1]

        # ── Contact patch migration ──────────────────────────────
        cp_x = result["contact_patch"][0] - result_static["contact_patch"][0]
        cp_y = result["contact_patch"][1] - result_static["contact_patch"][1]

        return {
            "camber": camber,
            "toe"   : toe,
            "caster": caster,
            "kpi"   : kpi,
            "scrub" : scrub,
            "cp_x"  : cp_x,
            "cp_y"  : cp_y,
        }

    out_L = corner_outputs(result_L, result_L_static, c_L)
    out_R = corner_outputs(result_R, result_R_static, c_R)

    # ── Motion ratios ────────────────────────────────────────────
    # Dampers span across the axle between left and right rocker arms.
    # MR = change in damper length / wheel travel.
    heave_len        = norm(result_L["heave_damper"] - result_R["heave_damper"])
    roll_len         = norm(result_L["roll_damper"]  - result_R["roll_damper"])
    heave_len_static = norm(result_L_static["heave_damper"] - result_R_static["heave_damper"])
    roll_len_static  = norm(result_L_static["roll_damper"]  - result_R_static["roll_damper"])

    if h_bump != 0:
        heave_MR = (heave_len - heave_len_static) / h_bump
        roll_MR  = (roll_len  - roll_len_static)  / h_bump
    else:
        heave_MR = 0.0
        roll_MR  = 0.0

    return {
        "L"           : out_L,
        "R"           : out_R,
        "heave_MR"    : heave_MR,
        "roll_MR"     : roll_MR,
        "wheel_travel": h_bump,
    }