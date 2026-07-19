# Author: Ludih
# Summary: Pure geometry primitives used by the kinematics solver.

import numpy as np
from numpy import sqrt, dot, cross
from numpy.linalg import norm

# ── Trilateration ────────────────────────────────────────────────────────────

def three_sphere_intersection(P1, P2, P3, r1, r2, r3):
    """
    Trilateration: find the two intersection points of three spheres.

    Args:
        P1, P2, P3 (numpy.ndarray, shape (3,)): sphere centers
        r1, r2, r3 (float): sphere radii

    Returns:
        tuple of two numpy.ndarray (shape (3,)): the intersection points,
        mirrored across the plane of the three centers.

    Raises:
        Exception: if the spheres do not intersect.
    """
    # Build local frame: e_x along P1->P2, e_y in the centers' plane, e_z normal to it
    temp1 = P2 - P1
    e_x = temp1 / norm(temp1)              # local x-axis
    temp2 = P3 - P1
    i = dot(e_x, temp2)                    # x-coord of P3 in local frame
    temp3 = temp2 - i * e_x
    e_y = temp3 / norm(temp3)              # local y-axis
    e_z = cross(e_x, e_y)                  # local z-axis

    d = norm(P2 - P1)                      # distance P1 to P2
    j = dot(e_y, temp2)                    # y-coord of P3 in local frame

    # Solve sphere equations for local coordinates of the intersection
    x = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    y = (r1 * r1 - r3 * r3 - 2 * i * x + i * i + j * j) / (2 * j)
    temp4 = r1 * r1 - x * x - y * y        # z², negative means no intersection
    if temp4 < 0:
        raise Exception("The three spheres do not intersect!")
    z = sqrt(temp4)

    # Convert back to world coordinates; ±z gives the two mirrored points
    p_12_a = P1 + x * e_x + y * e_y + z * e_z
    p_12_b = P1 + x * e_x + y * e_y - z * e_z
    return p_12_a, p_12_b

# ── LBJ circle parametrization ───────────────────────────────────────────────
def build_lbj_circle(c):
    """
    Compute the circle swept by the LBJ as the lower wishbone rotates
    about its fore-aft pivot axis.

    Args:
        c: corner points instance (FL, FR, RL, RR) with attributes
           lower_wishbone_fore, lower_wishbone_aft, lower_ball_joint
           (each a numpy.ndarray, shape (3,)).

    Returns:
        dict with keys:
            "center" (numpy.ndarray, shape (3,)): center of the circle,
                the projection of the LBJ onto the pivot axis
            "radius" (float): distance from the LBJ to the axis
            "u" (numpy.ndarray, shape (3,)): unit vector from center to LBJ, like i hat with vectors
            "v" (numpy.ndarray, shape (3,)): unit vector perpendicular to u, like j hat with vectors
                and the axis (so points on the circle are
                center + r*cos(θ)*u + r*sin(θ)*v)
    """
    fore = c.lower_wishbone_fore
    aft  = c.lower_wishbone_aft
    lbj  = c.lower_ball_joint

    # Unit vector along the wishbone pivot axis (fore -> aft)
    axis = aft - fore
    axis = axis / np.linalg.norm(axis)

    # Project the LBJ onto the axis to find the circle's center
    v = lbj - fore
    center = fore + np.dot(v, axis) * axis

    # Radius is the perpendicular distance from LBJ to the axis
    radius = np.linalg.norm(lbj - center)

    # In-plane basis vectors: u points at the LBJ, v completes the frame
    u = (lbj - center) / radius
    v = np.cross(axis, u)

    return {"center": center, "radius": radius, "u": u, "v": v}

def lbj_at(theta, center, radius, u, v):
    """
    Compute the LBJ position at a given wishbone rotation angle.

    Standard parametric circle equation in 3D:
    P(θ) = center + r·cos(θ)·u + r·sin(θ)·v

    The center is passed in explicitly (rather than read from the circle dict)
    so the caller can supply the "moved" center after heave/roll while keeping
    the constant radius/u/v from the original circle.

    Args:
        theta (float): wishbone rotation angle in radians; theta = 0
            corresponds to the original LBJ position (along u).
        center (numpy.ndarray, shape (3,)): center of the circle.
        radius (float): circle radius.
        u (numpy.ndarray, shape (3,)): in-plane unit vector toward the
            original LBJ position.
        v (numpy.ndarray, shape (3,)): in-plane unit vector perpendicular
            to u (direction of increasing theta).

    Returns:
        numpy.ndarray, shape (3,): LBJ position at angle theta.
    """
    # Parametric point on the circle: start at u (θ=0), rotate toward v
    return center + radius * (np.cos(theta) * u + np.sin(theta) * v)


# ── Rotation about an arbitrary axis ─────────────────────────────────────────
def rotation_about_axis(points, axis_start, axis_end, angle):
    """
    Rotate points about an arbitrary axis in 3D (Rodrigues' rotation formula).
    Used to rotate the chassis about its roll axis. 

    Builds the rotation matrix R = cos(θ)·I + sin(θ)·K + (1-cos(θ))·aaᵀ,
    where K is the skew-symmetric cross-product matrix of the unit axis a.
    Rotation follows the right-hand rule about the axis_start → axis_end
    direction.

    Args:
        points (array_like, shape (3,) or (N, 3)): point(s) to rotate.
        axis_start (array_like, shape (3,)): a point on the rotation axis.
        axis_end (array_like, shape (3,)): a second point defining the axis.
        angle (float): rotation angle in radians (right-hand rule).

    Returns:
        numpy.ndarray, same shape as points: the rotated point(s).
    """
    # Unit vector along the rotation axis
    axis = np.asarray(axis_end, float) - np.asarray(axis_start, float)  #Turning the axis points into np arrays and subtracting them to get the vector along the axis
    axis = axis / np.linalg.norm(axis)

    c = np.cos(angle)
    s = np.sin(angle)
    # Skew-symmetric matrix K, such that K @ p == cross(axis, p)
    K = np.array([[0.0,      -axis[2],  axis[1]],
                  [axis[2],   0.0,     -axis[0]],
                  [-axis[1],  axis[0],  0.0]])
    # Rodrigues' formula assembled as a 3x3 rotation matrix
    R = c * np.eye(3) + s * K + (1.0 - c) * np.outer(axis, axis)

    pts = np.asarray(points, float)
    # Translate so the axis passes through the origin, rotate, translate back.
    # (pts - start) @ R.T + start works for both (3,) and (N,3) inputs.
    return (pts - np.asarray(axis_start, float)) @ R.T + np.asarray(axis_start, float)

# ── Plane from three points ──────────────────────────────────────────────────
def equation_of_plane(p1, p2, p3):
    """
    Compute the equation of the plane passing through three points.

    The plane is expressed in Hesse normal form:
        dot(normal, x) = d
    where `normal` is the unit normal vector and d is the (signed)
    distance from the origin to the plane. Used by the instant-centre
    construction (intersect wishbone planes with a contact-patch plane).

    Args:
        p1, p2, p3 (array_like, shape (3,)): three non-collinear points
            on the plane.

    Returns:
        tuple:
            n (numpy.ndarray, shape (3,)): unit normal vector. Its
                direction follows the right-hand rule from (p2-p1)
                to (p3-p1).
            d (float): plane offset, so that dot(n, x) = d for every
                point x on the plane.
    """
    # Two edge vectors lying in the plane; their cross product is
    # perpendicular to both, i.e. normal to the plane
    n = np.cross(np.asarray(p2, float) - np.asarray(p1, float),
                 np.asarray(p3, float) - np.asarray(p1, float))
    # Normalize so d becomes a true distance and comparisons are scale-free
    n = n / np.linalg.norm(n)

    # Any point on the plane satisfies dot(n, x) = d; use p1 to find d
    # dot(n, x - p1) = 0
    #dot(n, x) - dot(n, p1) = 0
    #dot(n, x) = dot(n, p1) = d
    
    d = np.dot(n, np.asarray(p1, float))
    return n, d