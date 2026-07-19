# Author: Ludih
# Summary: This code solves the kinematics of the suspension.

import numpy as np
from scipy.optimize import brentq
import Geometric_Functions as gf
from Kinematic_Outputs import compute_outputs, roll_center, ackermann_percentage

POINT_KEYS = ("lw_fore", "lw_aft", "uw_fore", "uw_aft",
              "tr_chas", "bc_fore", "bc_aft", "lbj_center") #Used to easily sweep through the points of chassis
DIR_KEYS   = ("u", "v")   # LBJ circle in-plane axes

###------------Build Chassis Frame------------------###
def chassis_frame(c, circle):
    """
    Build the static chassis frame for a corner.

    Collects all chassis-fixed geometry into one dict, copying each
    array so later heave/roll/steer transformations can modify the
    frame without mutating the original corner data. This is the
    baseline that heave_chassis / roll / steer operate on.

    Args:
        c: corner points instance (FL, FR, RL, RR) with the chassis-side
           mount points as numpy.ndarray, shape (3,).
        circle (dict): LBJ circle from build_lbj_circle(), providing
            center, radius, u, v.

    Returns:
        dict:
            "lw_fore", "lw_aft" (numpy.ndarray, shape (3,)): lower
                wishbone fore/aft pivots
            "uw_fore", "uw_aft" (numpy.ndarray, shape (3,)): upper
                wishbone fore/aft pivots
            "tr_chas" (numpy.ndarray, shape (3,)): tie rod chassis mount
            "bc_fore", "bc_aft" (numpy.ndarray, shape (3,)): bellcrank mounts
            "lbj_center" (numpy.ndarray, shape (3,)): LBJ circle center
            "u", "v" (numpy.ndarray, shape (3,)): LBJ circle in-plane
                unit vectors
            "radius" (float): LBJ circle radius
    """
    # .copy() gives each frame its own arrays; without it, transforming
    # this frame in place would silently corrupt the original geometry.
    # radius is a plain float, so no copy needed (floats are immutable).
    return {
        "lw_fore"   : c.lower_wishbone_fore.copy(),
        "lw_aft"    : c.lower_wishbone_aft.copy(),
        "uw_fore"   : c.upper_wishbone_fore.copy(),
        "uw_aft"    : c.upper_wishbone_aft.copy(),
        "tr_chas"   : c.tie_rod_chassis.copy(),
        "bc_fore"   : c.bc_fore_mount.copy(),
        "bc_aft"    : c.bc_aft_mount.copy(),
        "lbj_center": circle["center"].copy(),
        "u"         : circle["u"].copy(),
        "v"         : circle["v"].copy(),
        "radius"    : circle["radius"],
    }

###------------------Chassis Motions------------------------###
def heave_frame(frame, h_bump):
    """
    Translate all chassis-fixed points down by h_bump to simulate heave.

    In bump the wheel moves up relative to the chassis; equivalently the
    chassis moves down while the ground stays fixed. Only position keys
    (POINT_KEYS) are translated. The LBJ circle directions u, v and the
    radius are unaffected by translation and are carried over as-is.

    Args:
        frame (dict): chassis frame from chassis_frame() (or an already
            posed frame from another mover).
        h_bump (float): heave displacement in metres; positive = bump
            (chassis translated down by h_bump).

    Returns:
        dict: a NEW frame with translated points. The input frame is
        not modified (shallow copy + reassignment of point keys only).
    """
    # Vertical translation vector; subtracting moves points down (-z)
    d = np.array([0.0, 0.0, h_bump])
    # Shallow copy: new dict, original untouched. u/v/radius carry over.
    out = dict(frame)
    # frame[k] - d allocates a new array each time, so the original
    # frame's arrays are never mutated
    for k in POINT_KEYS:
        out[k] = frame[k] - d
    return out

def steer_frame(frame, rack_travel):
    """
    Translate the tie rod's chassis-side (inner) point to simulate steering.

    The inner tie rod end rides on the steering rack, which slides
    laterally along y. The same world-frame offset applies to BOTH sides
    of the car (the rack is one rigid body), so left and right frames
    get the identical +y translation — one side toes in while the other
    toes out, which is what steers the car.

    Args:
        frame (dict): chassis frame from chassis_frame() (or an already
            posed frame from another mover — movers compose).
        rack_travel (float): lateral rack displacement in metres along +y;
            positive = rack moves toward +y (left).

    Returns:
        dict: a NEW frame with the translated tr_chas point; all other
        keys carried over unchanged. The input frame is not modified.
    """
    # Shallow copy, then repoint only tr_chas at a newly allocated array
    out = dict(frame)
    out["tr_chas"] = frame["tr_chas"] + np.array([0.0, rack_travel, 0.0])
    return out

def roll_frame(frame, axis_start, axis_end, angle):
    """
    Rotate the chassis frame about the roll axis to simulate roll.

    All chassis-fixed POINTS rotate about the actual axis line
    (axis_start -> axis_end, e.g. front RC -> rear RC). The LBJ circle
    axes u and v are DIRECTIONS, not positions: they must pick up the
    same rotation but no translation, so they are rotated about a
    parallel axis through the origin instead. The radius is a scalar
    and carries over unchanged.

    Args:
        frame (dict): chassis frame from chassis_frame() (or an already
            posed frame from another mover — movers compose).
        axis_start (array_like, shape (3,)): a point on the roll axis
            (e.g. front roll center).
        axis_end (array_like, shape (3,)): a second point on the axis
            (e.g. rear roll center).
        angle (float): roll angle in radians, right-hand rule about the
            axis_start -> axis_end direction.

    Returns:
        dict: a NEW frame with rotated points and directions. The input
        frame is not modified.
    """
    out = dict(frame)

    # Stack the 8 points into an (8, 3) array and rotate them all in
    # one call (rotation_about_axis handles (N, 3) input)
    pts = np.array([frame[k] for k in POINT_KEYS])
    pts = gf.rotation_about_axis(pts, axis_start, axis_end, angle)
    for i, k in enumerate(POINT_KEYS):
        out[k] = pts[i]

    # Directions: same rotation, but about an axis through the origin.
    # Passing (origin, axis_dir) reuses rotation_about_axis as a pure
    # rotation with no translate/untranslate offset.
    axis_dir = np.asarray(axis_end, float) - np.asarray(axis_start, float)
    for k in DIR_KEYS:
        out[k] = gf.rotation_about_axis(frame[k], np.zeros(3), axis_dir, angle)
    return out

# ── Root picking ─────────────────────────────────────────────────────────────

def pick_root(candidates, reference):
    """
    Pick the physically correct trilateration solution.

    three_sphere_intersection always returns TWO points, mirrored across
    the plane of the three sphere centers. Only one corresponds to the
    real mechanism; the other is the "elbow bent the wrong way" solution.
    The real one is identified as whichever lies closest to a reference
    point: the static position for a direct solve, or the previous step's
    solved position during a march (continuity).

    Args:
        candidates (tuple of two numpy.ndarray, shape (3,)): the two
            intersection points from three_sphere_intersection().
        reference (numpy.ndarray, shape (3,)): known-good nearby position
            of the same point.

    Returns:
        numpy.ndarray, shape (3,): the candidate closest to reference.
    """
    # Euclidean distance from the reference to each candidate;
    # keep the nearer one
    d1 = np.linalg.norm(reference - candidates[0])
    d2 = np.linalg.norm(reference - candidates[1])
    return candidates[0] if d1 < d2 else candidates[1]

def static_ref(c):
    """
    Build the reference-point dict used by pick_root, from static geometry.

    Maps each solved point's name to its design-position location. When
    solve_corner trilaterates a point and gets two candidates, it uses
    the entry here with the matching key to pick the physical one.

    During a march, pass the PREVIOUS step's solved result dict INSTEAD OF THIS FUNCTION,
    it has the same keys, so it drops in as a replacement. This
    continuity seeding matters because at large motion the wrong
    mirror-image root can end up closer to the *static* position than
    the correct root, flipping the mechanism inside-out mid-sweep. The
    previous step's pose is always close to the current one (small
    increments), so it can't branch-flip.

    Args:
        c: corner points instance (FL, FR, RL, RR) with the static
           outboard point positions as numpy.ndarray, shape (3,).

    Returns:
        dict of numpy.ndarray (shape (3,)): reference position for every
        point solve_corner produces, keyed by the same names as
        solve_corner's result dict (UBJ, TRO, contact_patch, upper_wheel,
        fore_wheel, aft_wheel, pushrod_outboard, pushrod_inboard,
        heave_damper, roll_damper).
    """
    return {
        "UBJ"             : c.upper_ball_joint,
        "TRO"             : c.tie_rod_outer,
        "contact_patch"   : c.contact_patch,
        "upper_wheel"     : c.upper_wheel_pt,
        "fore_wheel"      : c.fore_wheel_pt,
        "aft_wheel"       : c.aft_wheel_pt,
        "pushrod_outboard": c.pushrod_outboard,
        "pushrod_inboard" : c.pushrod_inboard,
        "heave_damper"    : c.heave_damper,
        "roll_damper"     : c.roll_damper,
    }

# ── Solvers ──────────────────────────────────────────────────────────────

def _chain(theta, fr, L, ref):
    """
    Solve the core kinematic chain LBJ -> UBJ -> TRO -> CP for one
    candidate wishbone angle theta on a posed chassis frame.

    Each point is found from three already-known points and three fixed
    rigid-body distances (trilateration), in dependency order:

      1. LBJ — directly on its circle at angle theta (1 DOF of the
         mechanism; this is the unknown brentq searches over).
      2. UBJ — knuckle length from LBJ, plus the two upper wishbone
         arm lengths from the chassis pivots.
      3. TRO — tie rod length from the (posed) rack point, plus the
         two steering-arm lengths from UBJ and LBJ. This is where
         steering enters: moving tr_chas drags TRO around the kingpin.
      4. CP  — fixed upright distances from TRO, UBJ, LBJ. Rigidly
         carried by the upright; its height is the residual driven
         to zero.

    Shared by residual() and solve_corner() so the root-finder and the
    final solve can never use different equations.

    Args:
        theta (float): lower wishbone rotation angle (radians).
        fr (dict): posed chassis frame (from chassis_frame + movers).
        L: Lengths instance for this axle (rigid-body distances).
        ref (dict): reference points for pick_root (static_ref or the
            previous step's solved result).

    Returns:
        tuple of numpy.ndarray (shape (3,)): (LBJ, UBJ, TRO, CP).
    """
    # 1 DOF in: theta fixes the LBJ on its circle about the LW pivot axis
    LBJ = gf.lbj_at(theta, fr["lbj_center"], fr["radius"], fr["u"], fr["v"])

    # UBJ: distance L.knuckle from LBJ (rigid upright), L.upper_fore /
    # L.upper_aft from the upper wishbone chassis pivots (rigid arms)
    UBJ = pick_root(
        gf.three_sphere_intersection(
            LBJ,       fr["uw_fore"], fr["uw_aft"],
            L.knuckle, L.upper_fore,  L.upper_aft),
        ref["UBJ"])

    # TRO: tie rod length from the rack point, steering-arm distances
    # from UBJ and LBJ (TRO is rigidly part of the upright)
    TRO = pick_root(
        gf.three_sphere_intersection(
            fr["tr_chas"], UBJ,                  LBJ,
            L.tie_rod,     L.upper_steering_arm, L.lower_steering_arm),
        ref["TRO"])

    # CP: rigid upright distances from the three solved upright points.
    # Its z-height is what the outer root-finder drives to zero.
    CP = pick_root(
        gf.three_sphere_intersection(
            TRO,                      UBJ,               LBJ,
            L.out_tierod2lower_wheel, L.ubj2lower_wheel, L.lbj2lower_wheel),
        ref["contact_patch"])

    return LBJ, UBJ, TRO, CP

def residual(theta, fr, L, ref):
    """
    Constraint function for the root-finder: contact patch height.

    For an arbitrary wishbone angle theta, the solved contact patch
    generally floats above or dips below the ground plane (z = 0).
    The physically correct pose is the theta where it sits exactly ON
    the ground. brentq varies theta to drive this return value to zero.

    Sign behaviour: theta too low -> wheel hangs (CP z > 0 or < 0
    depending on geometry); the function is monotonic in theta over the
    bracket, which is what lets brentq bisect on a sign change.

    Args:
        theta (float): candidate lower wishbone angle (radians).
        fr (dict): posed chassis frame.
        L: Lengths instance for this axle.
        ref (dict): reference points for pick_root.

    Returns:
        float: z-coordinate of the contact patch (m); 0 means the tire
        is exactly on the ground.
    """
    # _chain returns (LBJ, UBJ, TRO, CP); [3] is CP, [2] is its z
    return _chain(theta, fr, L, ref)[3][2]

def solve_corner(fr, L, ref, bracket=np.pi/4):
    """
    Solve the complete suspension pose for one corner on a posed
    chassis frame.

    Two phases:
      1. Find the mechanism's single DOF: brentq searches the wishbone
         angle theta until the contact patch sits exactly on the ground
         (residual = CP height = 0). This yields the core chain
         LBJ, UBJ, TRO, CP.
      2. Forward-propagate the remaining points by trilateration, each
         from three already-solved points and rigid lengths:
         wheel-plane points (from the upright), then pushrod outboard
         (rides on the upper wishbone), pushrod inboard (on the
         bellcrank), and the two damper points (also on the bellcrank).

    Args:
        fr (dict): posed chassis frame (chassis_frame + any movers).
        L: Lengths instance for this axle.
        ref (dict): reference points for pick_root — static_ref(c) for
            a direct solve, or the previous step's result during a march.
        bracket (float): half-width of the theta search interval
            (radians). residual(-bracket) and residual(+bracket) must
            have opposite signs for brentq to converge.

    Returns:
        dict: solved pose —
            "theta" (float): wishbone angle at the solution (radians);
            all other keys (numpy.ndarray, shape (3,)): LBJ, UBJ, TRO,
            contact_patch, upper_wheel, fore_wheel, aft_wheel,
            pushrod_outboard, pushrod_inboard, heave_damper, roll_damper.
            Keys match static_ref/pick_root so results chain in a march.
    """
    # ── Phase 1: solve the 1 DOF ─────────────────────────────────
    # Find theta where the contact patch height crosses zero.
    theta = brentq(residual, a=-bracket, b=bracket,
                   args=(fr, L, ref), xtol=1e-10)

    # Re-evaluate the core chain at the converged theta
    LBJ, UBJ, TRO, CP = _chain(theta, fr, L, ref)

    # ── Phase 2: propagate the rest of the mechanism ─────────────

    # Wheel-plane points: each rigidly fixed to the upright, located
    # by its three stored distances from TRO, UBJ, LBJ
    upper_wheel = pick_root(
        gf.three_sphere_intersection(
            TRO,                      UBJ,               LBJ,
            L.out_tierod2upper_wheel, L.ubj2upper_wheel, L.lbj2upper_wheel),
        ref["upper_wheel"])

    fore_wheel = pick_root(
        gf.three_sphere_intersection(
            TRO,                     UBJ,                    LBJ,
            L.out_tierod2fore_wheel, L.upper_out2fore_wheel, L.lower_out2fore_wheel),
        ref["fore_wheel"])

    aft_wheel = pick_root(
        gf.three_sphere_intersection(
            TRO,                    UBJ,                   LBJ,
            L.out_tierod2aft_wheel, L.upper_out2aft_wheel, L.lower_out2aft_wheel),
        ref["aft_wheel"])

    # Pushrod outboard end: mounted on the UPPER wishbone, so it is
    # rigid relative to the UBJ and the two (posed) chassis pivots
    pushrod_outboard = pick_root(
        gf.three_sphere_intersection(
            UBJ,                 fr["uw_fore"],                 fr["uw_aft"],
            L.ubj2lower_pushrod, L.fore_wishbone2lower_pushrod, L.aft_wishbone2lower_pushrod),
        ref["pushrod_outboard"])

    # Pushrod inboard end: pushrod length from its outboard end, and
    # rigid bellcrank distances from the two (posed) bellcrank mounts
    pushrod_inboard = pick_root(
        gf.three_sphere_intersection(
            pushrod_outboard, fr["bc_fore"],           fr["bc_aft"],
            L.pushrod,        L.bc_fore2upper_pushrod, L.bc_aft2upper_pushrod),
        ref["pushrod_inboard"])

    # Damper attachment points: both rigid on the bellcrank, located
    # from the solved pushrod inboard point and the bellcrank mounts
    heave_damper = pick_root(
        gf.three_sphere_intersection(
            pushrod_inboard,              fr["bc_fore"],          fr["bc_aft"],
            L.upper_pushrod2heave_damper, L.bc_fore2heave_damper, L.bc_aft2heave_damper),
        ref["heave_damper"])

    roll_damper = pick_root(
        gf.three_sphere_intersection(
            pushrod_inboard,             fr["bc_fore"],         fr["bc_aft"],
            L.upper_pushrod2roll_damper, L.bc_fore2roll_damper, L.bc_aft2roll_damper),
        ref["roll_damper"])

    # Solved pose; keys deliberately match the ref dict so this result
    # can seed the next step's root picking in a march
    return {
        "theta"           : theta,
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

# ── Independent sweeps (direct solves, no march) ─────────────────────────────

def _solve_range(fr0, pose, values, L, c):
    """
    Solve one corner over a range of pose values (rack travels, heaves)
    with continuity seeding. 

    The values are sorted and walked OUTWARD from the one closest to
    zero, in two passes (toward +max, then toward -max). Each solve is
    seeded with the previous solve's result, so pick_root always
    compares against a nearby pose and the trilateration roots cannot
    branch-flip at large motion — the same continuity idea as march(),
    but for independent single-corner sweeps.

    Args:
        fr0 (dict): static chassis frame for this corner.
        pose (function): mover with signature pose(fr0, v) -> posed
            frame, e.g. steer_frame or heave_frame.
        values (array_like): pose values to solve (any order, may span
            zero, e.g. np.linspace(-0.025, 0.025, 50)).
        L: Lengths instance for this axle.
        c: corner points instance (for the static reference seed).

    Returns:
        list of dict: solve_corner results, in the ORIGINAL order of
        `values` (not sorted order), so results[i] corresponds to
        values[i].
    """
    #Sorting the values in ascending order and finding the index of the one closest to 0
    values  = np.asarray(values, float)
    order   = np.argsort(values)        # indices that would sort values
    sorted_ = values[order]             # the values in ascending order
    zero_i  = int(np.argmin(np.abs(sorted_)))   # index (in sorted order)
                                                # of the value nearest 0
                                                # is the start point of both walks

    results = [None] * len(values)      # empty list of size values
    ref0    = static_ref(c)             # both walks start seeded from static

    # Walk from ~zero up to +max, re-seeding each step from the last
    ref = ref0
    for j in range(zero_i, len(sorted_)):           # 0 -> +max
        res = solve_corner(pose(fr0, sorted_[j]), L, ref)
        results[order[j]] = res         # order[j] maps back to the
                                        # position in the ORIGINAL values
        ref = res                       # continuity seed for next step

    # Walk from just below zero down to -max, again seeded from static
    ref = ref0
    for j in range(zero_i - 1, -1, -1):             # 0 -> -max
        res = solve_corner(pose(fr0, sorted_[j]), L, ref)
        results[order[j]] = res
        ref = res

    return results

def sweep_heave(c_L, c_R, L, bump_range=np.linspace(-0.0125, 0.040, 50)):
    """
    Sweep chassis heave for one axle (both corners) and compute outputs.

    Heave is path-independent, so every pose is solved DIRECTLY from
    the static frame at its absolute h_bump (no marching). Root
    continuity is protected by _solve_range's outward-from-zero
    seeding. After the sweep, the instantaneous heave motion ratio is
    computed numerically from neighbouring poses.

    Args:
        c_L, c_R: left and right corner instances for this axle.
        L: Lengths instance for this axle.
        bump_range (array_like): heave values in metres, positive =
            bump. Default: -12.5 mm droop to +40 mm bump, 50 steps.

    Returns:
        list of dict: one row per bump value — the compute_outputs dict
        (per-corner metrics under "L"/"R", damper lengths, roll center)
        plus "h_bump" (m) and "heave_MR" (wheel travel / damper travel).
    """
    # Static frames, references, and the ride-height baseline solve
    fr_L0 = chassis_frame(c_L, gf.build_lbj_circle(c_L))
    fr_R0 = chassis_frame(c_R, gf.build_lbj_circle(c_R))
    ref_L, ref_R = static_ref(c_L), static_ref(c_R)

    result_L_static = solve_corner(fr_L0, L, ref_L)
    result_R_static = solve_corner(fr_R0, L, ref_R)

    # Solve every heave pose for each corner (continuity-seeded)
    res_L = _solve_range(fr_L0, heave_frame, bump_range, L, c_L)
    res_R = _solve_range(fr_R0, heave_frame, bump_range, L, c_R)

    # One output row per pose; re-pose frames for the RC construction
    results = []
    for i, h_bump in enumerate(bump_range):
        fr_L, fr_R = heave_frame(fr_L0, h_bump), heave_frame(fr_R0, h_bump)
        outputs = compute_outputs(res_L[i], res_R[i],
                                  result_L_static, result_R_static,
                                  c_L, c_R, h_bump, fr_L, fr_R)
        outputs["h_bump"] = h_bump
        results.append(outputs)

    # ── Instantaneous heave motion ratio ─────────────────────────
    # MR = wheel travel / damper travel. This is a DERIVATIVE (rate of
    # change between neighbouring poses), not a ratio of absolutes, so
    # it can only be computed here in the sweep where consecutive
    # solved poses exist — not inside compute_outputs on a single pose.

    # Gather the sweep as two aligned arrays: x = wheel travel (h_bump,
    # since in pure heave the wheel moves h_bump relative to chassis),
    # y = heave damper length at that travel
    hb  = np.array([r["h_bump"] for r in results]) # wheel travel (x)
    ln  = np.array([r["heave_damper_len"] for r in results]) # heave damper length (y)

    # np.gradient assumes x is monotonic; results are in bump_range
    # order, which may not be sorted. Sort for the derivative, then
    # scatter the answers back to original positions via the same
    # argsort-index trick as _solve_range.
    idx = np.argsort(hb)                     # gradient needs monotonic x
    dl_dw = np.empty_like(hb)                # uninitialized output slots
    dl_dw[idx] = np.gradient(ln[idx], hb[idx])   # d(damper len)/d(wheel travel) , central differences

    # dl_dw is damper/wheel; MR convention is wheel/damper, so invert.
    # Minus sign: damper LENGTH decreases in bump (compression), so
    # dl_dw < 0 there; negating makes compression read as positive MR.
    # Guard against a flat spot (derivative ~ 0 would blow up to inf).
    for r, d in zip(results, dl_dw):
        r["heave_MR"] = -1.0 / d if abs(d) > 1e-12 else np.nan
    return results

def sweep_steer(c_L, c_R, L, rack_range=np.linspace(-0.025, 0.025, 50),
                wheelbase=None, track=None):
    """
    Sweep steering rack travel for one axle at static ride height.

    Structurally identical to sweep_heave — direct solves via
    _solve_range with continuity seeding — but the injected mover is
    steer_frame, so only the inner tie-rod point moves between poses.
    The same world-frame rack offset is applied to BOTH sides (one
    rigid rack), which is what makes one wheel toe in while the other
    toes out. No motion-ratio block: the dampers barely move under
    steer, so there is no meaningful steer MR to compute.

    Args:
        c_L, c_R: left and right corner instances (FL, FR — this is a
            front-axle sweep).
        L: Lengths instance for the axle.
        rack_range (array_like): rack displacements in metres along +y.
            Default: ±25 mm in 50 steps.
        wheelbase (float, optional): wheelbase in metres. Required
            together with track for Ackermann.
        track (float, optional): front track width in metres.

    Returns:
        list of dict: one row per rack value — compute_outputs dict
        plus "rack_travel" (m) and, if wheelbase and track were given,
        "ackermann" (%, np.nan near zero steer).
    """
    # Static frames, references, ride-height baseline (same preamble
    # as sweep_heave)
    fr_L0 = chassis_frame(c_L, gf.build_lbj_circle(c_L))
    fr_R0 = chassis_frame(c_R, gf.build_lbj_circle(c_R))
    ref_L, ref_R = static_ref(c_L), static_ref(c_R)

    result_L_static = solve_corner(fr_L0, L, ref_L)
    result_R_static = solve_corner(fr_R0, L, ref_R)

    # Solve every rack pose per corner; steer_frame is the mover
    res_L = _solve_range(fr_L0, steer_frame, rack_range, L, c_L)
    res_R = _solve_range(fr_R0, steer_frame, rack_range, L, c_R)

    results = []
    for i, rack in enumerate(rack_range):
        # Re-pose frames for the roll-center construction inside
        # compute_outputs (needs the posed chassis pivots)
        fr_L, fr_R = steer_frame(fr_L0, rack), steer_frame(fr_R0, rack)

        # h_bump argument is 0.0: the chassis is at ride height, only
        # the rack has moved
        outputs  = compute_outputs(res_L[i], res_R[i],
                                   result_L_static, result_R_static,
                                   c_L, c_R, 0.0, fr_L, fr_R)
        outputs["rack_travel"] = rack   # x-axis tag for plotting

        # Ackermann needs both wheel angles plus car dimensions; only
        # computed if the caller supplied wheelbase and track
        if wheelbase is not None and track is not None:
            outputs["ackermann"] = ackermann_percentage(
                outputs["L"]["wheel_angle"], outputs["R"]["wheel_angle"],
                wheelbase, track)
        results.append(outputs)
    return results

# ── Marching solver (roll / combined) ────────────────────────────────────────

def march(FL, FR, RL, RR, F_len, R_len,
          roll_end=0.0, heave_end=0.0, rack_end=0.0, steps=50,
          steer_front_only=True):
    """
    March all four corners from static to the target pose in `steps` equal
    increments of roll, heave and rack applied together (co-ramped).

    Each increment:
      1. front & rear roll centers are computed from the PREVIOUS step's
         solved geometry (migrating roll axis, one step stale),
      2. every chassis frame is rotated by d_roll about the RC_F -> RC_R
         line, heaved by d_heave, and the tie rod steered by d_rack,
      3. each corner is re-solved (contact patch to ground), root-picking
         seeded from the previous step's solution.

    Pure roll:  march(..., roll_end=np.radians(2))
    Combined:   march(..., roll_end=..., heave_end=..., rack_end=...)

    rack_end is applied to the FRONT corners only by default
    (steer_front_only=True): the rear tie rod is a fixed toe link, not a
    rack. Set steer_front_only=False only for a rear-steer car.

    Positive roll follows the right-hand rule about the RC_F -> RC_R
    direction. With +x forward, positive roll drops the right (-y) side.

    Returns a list of per-step dicts with solved corners, roll centers,
    and the applied inputs.
    """
    corners = {"FL": FL, "FR": FR, "RL": RL, "RR": RR}
    lens    = {"FL": F_len, "FR": F_len, "RL": R_len, "RR": R_len}

    frames  = {k: chassis_frame(c, gf.build_lbj_circle(c))  #Makes a dict of chassis points
               for k, c in corners.items()}
    results = {k: solve_corner(frames[k], lens[k], static_ref(c)) #Solves for static outboard points
               for k, c in corners.items()}

    d_roll  = roll_end  / steps
    d_heave = heave_end / steps
    d_rack  = rack_end  / steps

    # roll axis for the first increment, from the static geometry
    rc_F = roll_center(results["FL"], frames["FL"], results["FR"], frames["FR"])
    rc_R = roll_center(results["RL"], frames["RL"], results["RR"], frames["RR"])

    history = []
    for n in range(1, steps + 1):
        # 1. pose all frames by one increment about the current roll axis
        for k in frames:
            fr = frames[k]
            if d_roll != 0.0:  #Roll the chassis
                fr = roll_frame(fr, rc_F, rc_R, d_roll)
            if d_heave != 0.0:      #Then heave the chassis
                fr = heave_frame(fr, d_heave)
            if d_rack != 0.0 and (k in ("FL", "FR") or not steer_front_only):   #Then steer, error here since rack only moves in y
                fr = steer_frame(fr, d_rack)
            frames[k] = fr

        # 2. re-solve, seeded from last step
        results = {k: solve_corner(frames[k], lens[k], results[k])
                   for k in frames}

        # 3. roll centers of the NEW geometry: stored with this step,
        #    and used as the axis for the next increment (migrating RC).
        rc_F = roll_center(results["FL"], frames["FL"], results["FR"], frames["FR"])
        rc_R = roll_center(results["RL"], frames["RL"], results["RR"], frames["RR"])

        history.append({
            "roll"   : n * d_roll,
            "h_bump" : n * d_heave,
            "rack"   : n * d_rack,
            "rc_F"   : rc_F,
            "rc_R"   : rc_R,
            "corners": {k: results[k] for k in results},
            "frames" : {k: frames[k] for k in frames},
        })
    return history

def sweep_roll(FL, FR, RL, RR, F_len, R_len,
               roll_range=(np.radians(-2.0), np.radians(2.0)), steps=50):
    """
    Roll sweep over roll_range = (min, max) in RADIANS, using the
    migrating-roll-axis march.

    Unlike heave/steer, roll poses are PATH DEPENDENT: the rotation axis
    is recomputed from the geometry every increment, so a pose at -1 deg
    can only be reached by marching 0 -> -1, never by jumping there or
    by continuing from +2 back down through zero. A range spanning zero
    is therefore run as TWO independent marches (0 -> lo and 0 -> hi)
    stitched around an exactly-static zero row. `steps` is the total
    budget, split between the branches in proportion to their spans.

    Args:
        FL, FR, RL, RR: the four corner instances.
        F_len, R_len: Lengths instances for the front and rear axles.
        roll_range (tuple of float): (min, max) roll in RADIANS.
            May span zero, or be entirely one-sided.
        steps (int): total number of rows across the whole range.

    Returns:
        list of dict: one row per pose, ascending in roll —
            "roll_deg" (float), and "front"/"rear": full axle output
            dicts (compute_outputs) each also carrying "roll_MR".
        Use march() directly instead if you want raw solved points.
    """
    lo, hi = roll_range                     # tuple unpacking
    if lo > hi:
        raise ValueError("roll_range must be (min, max)")

    # ── Static solve: outputs baseline + the zero-roll row ───────
    st, fr0 = {}, {}
    for k, c, L in (("FL", FL, F_len), ("FR", FR, F_len),
                    ("RL", RL, R_len), ("RR", RR, R_len)):
        fr0[k] = chassis_frame(c, gf.build_lbj_circle(c))
        st[k]  = solve_corner(fr0[k], L, static_ref(c))

    # ── Convert march history to output rows ─────────────────────
    # Nested helper: turns each raw march step (solved points + frames)
    # into per-axle metrics. Closes over st/FL/FR/RL/RR from the
    # enclosing scope, so it needs no arguments beyond the history.
    def rows_from(history):
        rows = []
        for h in history:
            co, fr = h["corners"], h["frames"]
            front = compute_outputs(co["FL"], co["FR"], st["FL"], st["FR"],
                                    FL, FR, 0.0, fr["FL"], fr["FR"])
            rear  = compute_outputs(co["RL"], co["RR"], st["RL"], st["RR"],
                                    RL, RR, 0.0, fr["RL"], fr["RR"])
            rows.append({"roll_deg": np.degrees(h["roll"]),
                         "front": front, "rear": rear})
        return rows

    # ── Split the step budget between the two branches ───────────
    # Each branch gets steps proportional to its share of the span
    # (e.g. (-1, +3): 25% neg, 75% pos); max(1, ...) guards against a
    # branch rounding down to zero steps.
    span    = hi - lo
    n_neg   = max(1, round(steps * (0.0 - min(lo, 0.0)) / span)) if lo < 0 else 0
    n_pos   = max(1, round(steps * (max(hi, 0.0) - 0.0) / span)) if hi > 0 else 0

    rows = []

    # Negative branch: march 0 -> lo (increasingly negative), then
    # REVERSE it so the assembled rows run ascending (-|lo| ... -step)
    if lo < 0:
        neg = rows_from(march(FL, FR, RL, RR, F_len, R_len,
                              roll_end=lo, steps=n_neg))
        rows.extend(reversed(neg))                    # -|lo| ... -step

    # Zero row: the static solve compared against itself (baseline
    # deltas read 0), with the un-posed frames for the roll centers.
    # march() doesn't emit a step-zero row, so it's inserted manually.
    if lo <= 0.0 <= hi:
        zero_front = compute_outputs(st["FL"], st["FR"], st["FL"], st["FR"],
                                     FL, FR, 0.0, fr0["FL"], fr0["FR"])
        zero_rear  = compute_outputs(st["RL"], st["RR"], st["RL"], st["RR"],
                                     RL, RR, 0.0, fr0["RL"], fr0["RR"])
        rows.append({"roll_deg": 0.0,
                     "front": zero_front, "rear": zero_rear})

    # Positive branch: march 0 -> hi, already ascending
    if hi > 0:
        rows.extend(rows_from(march(FL, FR, RL, RR, F_len, R_len,
                                    roll_end=hi, steps=n_pos)))

    # One-sided range (e.g. (1deg, 2deg)): the march had to start from
    # zero anyway (path dependence), so it produced 0 -> bound; trim to
    # the requested window. The 1e-12 tolerances absorb radian/degree
    # round-trip float error at the window edges.
    if lo > 0 or hi < 0:
        rows = [r for r in rows
                if np.radians(r["roll_deg"]) >= lo - 1e-12
                and np.radians(r["roll_deg"]) <= hi + 1e-12]

    # ── Instantaneous roll motion ratio ──────────────────────────
    # Same np.gradient pattern as sweep_heave, but the "wheel travel"
    # side is derived: in pure roll each wheel moves (track/2)·dφ
    # relative to the chassis, so
    #   MR = d(wheel)/d(damper) = (t/2)·dφ / d(len) = (t/2) / (d len/dφ).
    # Minus sign: compression reads positive. A uniformly negative MR
    # just means the roll element EXTENDS under positive roll — use the
    # magnitude when carrying it into a roll-stiffness calc.
    t_F = abs(FL.contact_patch[1] - FR.contact_patch[1])   # front track (m)
    t_R = abs(RL.contact_patch[1] - RR.contact_patch[1])   # rear track (m)
    phi = np.radians([r["roll_deg"] for r in rows])        # x for gradient (rad)
    if len(rows) > 1:                    # gradient needs >= 2 samples
        for axle, t in (("front", t_F), ("rear", t_R)):
            ln    = np.array([r[axle]["roll_damper_len"] for r in rows])
            dl_dp = np.gradient(ln, phi)          # d(damper len)/d(roll rad)
            for r, d in zip(rows, dl_dp):
                r[axle]["roll_MR"] = (-(t / 2.0) / d
                                      if abs(d) > 1e-12 else np.nan)
    return rows