# Author: Ludih
# Summary: Shared solver functions for the Autocross/Endurance events.

import numpy as np
import physics as ph
import Limits


def corner_speed_ceiling(track, car, tire, n_grid=40):
    """v_max at every track point, from corner_speed(R = 1/|k|).

    corner_speed costs a bisection each call, so instead of calling it at every
    track point we call it on a small grid of curvatures once and interpolate:
    the same precompute-then-lookup pattern as the tire grip table."""
    k_abs = np.abs(track.k)
    k_lo = 1e-4                                   # 10 km radius, effectively straight
    k_hi = max(k_abs.max(), 1e-3)
    k_grid = np.geomspace(k_lo, k_hi, n_grid)
    v_grid = np.array([Limits.corner_speed(tire, car, 1.0 / kk)["v"] for kk in k_grid])
    return np.interp(np.clip(k_abs, k_lo, None), k_grid, v_grid)


def braking_decel(car, tire, v, ax_prev=0.0):
    """Peak deceleration (m/s^2, positive) available at speed v.
    Tire braking peaks at the current loads — including longitudinal load
    transfer from the previous step's deceleration (ax_prev, negative under
    braking, same lagged-coupling scheme as Acceleration.py) — plus drag and
    rolling resistance helping, capped by the car's max_decel."""
    DF = ph.downforce(car, v)
    FzF = max(ph.FL_Fz(car, DF, 0.0, ax_prev), 0.0)
    FzR = max(ph.RL_Fz(car, DF, 0.0, ax_prev), 0.0)
    F_tire = 2 * tire.peak(FzF)[1] + 2 * tire.peak(FzR)[1]
    F_resist = ph.drag(car, v) + ph.rolling_resistance(car, v, car.mass_total * car.g + DF)
    return min((F_tire + F_resist) / car.mass_total, car.max_decel)


def ellipse(v, v_max):
    """Friction ellipse scale factor for longitudinal force while cornering.
    Lateral demand/capacity = (v/v_max)^2, so the remaining longitudinal
    fraction is sqrt(1 - (v/v_max)^4)."""
    r = min(v / max(v_max, 1e-9), 1.0)
    return np.sqrt(max(0.0, 1.0 - r**4))


def forward_pass(track, car, tire, v_max, v0, n_loops=1, closed=True):
    """March forward: accelerate as hard as traction allows, capped by v_max.
    closed=True wraps around the lap (flying lap, start speed converges over
    n_loops). closed=False is a standing start: v begins at v0 and never wraps."""
    n = len(v_max)
    v = np.minimum(np.full(n, v0), v_max)
    for _ in range(n_loops):
        ax_prev = 0.0                              # load transfer, lagged one step
        for i in range(n - 1):
            vi = min(v[i], v_max[i])
            DF = ph.downforce(car, vi)
            FzF = max(ph.FL_Fz(car, DF, 0.0, ax_prev), 0.0)
            FzR = max(ph.RL_Fz(car, DF, 0.0, ax_prev), 0.0)
            r = Limits.tractive_force_4wd(tire, car, FzF, FzR, max(vi, 0.1))
            F_trac = r["Fx_total"] * ellipse(vi, v_max[i])
            F_net = F_trac - ph.drag(car, vi) - ph.rolling_resistance(
                car, vi, car.mass_total * car.g + DF)
            ax = F_net / car.mass_total
            ax_prev = ax
            v[i + 1] = min(np.sqrt(max(vi**2 + 2 * ax * track.ds, 0.01)), v_max[i + 1])
        if closed:
            v[0] = v[-1]                           # wrap for flying laps
        else:
            v[0] = v0                              # standing start stays a standing start
    return v


def backward_pass(track, car, tire, v_max, n_loops=1, closed=True):
    """March backward: at each point, the fastest speed from which the car can
    still brake down to the next point's speed. closed=False leaves the finish
    line unconstrained (cross it flat out, no braking after)."""
    n = len(v_max)
    v = v_max.copy()
    for _ in range(n_loops):
        dec_prev = 0.0                             # load transfer, lagged one step
        for i in range(n - 1, 0, -1):
            vi = min(v[i], v_max[i])
            dec = braking_decel(car, tire, vi, ax_prev=-dec_prev) * ellipse(vi, v_max[i])
            dec_prev = dec
            v[i - 1] = min(np.sqrt(vi**2 + 2 * dec * track.ds), v_max[i - 1])
        if closed:
            v[-1] = v[0]                           # wrap for flying laps
    return v


def energy_and_time(track, car, tire, v):
    """Integrate lap time and battery energy over the final speed profile."""
    n = len(v)
    dt = track.ds / np.maximum(v, 0.1)
    t = np.concatenate(([0.0], np.cumsum(dt[:-1])))

    P_pack = np.zeros(n)
    for i in range(n - 1):
        vi = max(v[i], 0.1)
        DF = ph.downforce(car, vi)
        ax = (v[i + 1]**2 - v[i]**2) / (2 * track.ds)
        F_resist = ph.drag(car, vi) + ph.rolling_resistance(
            car, vi, car.mass_total * car.g + DF)
        F_prop = car.mass_total * ax + F_resist    # force the powertrain must supply
        if F_prop > 0:                             # driving (braking energy: regen TODO)
            T = ph.motor_torque(car, F_prop / 4)   # per motor
            rpm = ph.motor_rpm(car, vi)
            eta = ph.motor_eff(car, rpm, T)
            P_pack[i] = ph.pack_power(car, [T] * 4, [eta] * 4, rpm)
    energy_J = float(np.sum(P_pack[:-1] * dt[:-1]))
    return t, P_pack, energy_J



def lap_profile(track, car, tire, v_max, standing_start):
    """One lap's speed profile. standing_start=True: v0=0, open ends.
    standing_start=False: flying lap, wrapped until converged."""
    if standing_start:
        v_fwd = forward_pass(track, car, tire, v_max, v0=0.1, n_loops=1, closed=False)
        v_bwd = backward_pass(track, car, tire, v_max, n_loops=1, closed=False)
    else:
        v_fwd = forward_pass(track, car, tire, v_max, v0=v_max[0], n_loops=2, closed=True)
        v_bwd = backward_pass(track, car, tire, v_max, n_loops=2, closed=True)
    return np.minimum(v_fwd, v_bwd), v_fwd, v_bwd