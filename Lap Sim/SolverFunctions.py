import numpy as np
import Physics as ph
import Limits

def corner_speed_ceiling(track, car, tire, n_grid=40):
    """v_max at every track point (m/s). corner_speed is slow, so it runs on a
    small grid of curvatures and the track is interpolated from it."""
    k_abs = np.abs(track.k)                       # 1/m
    k_lo = 1e-4                                   # 1/m, 10 km radius, effectively straight
    k_hi = max(k_abs.max(), 1e-3)                 # 1/m
    k_grid = np.geomspace(k_lo, k_hi, n_grid)
    v_grid = np.array([Limits.corner_speed(tire, car, 1.0 / kk)["v"] for kk in k_grid])   # m/s
    return np.interp(np.clip(k_abs, k_lo, None), k_grid, v_grid)


def braking_decel(car, tire, v, v_max, k=0.0, ax_prev=0.0):
    """Peak deceleration (m/s^2, positive) at speed v. Tyre braking at the current
    loads (load transfer from the previous step, ax_prev negative under braking)
    plus drag and rolling resistance, capped by car.max_decel."""
    DF = ph.downforce(car, v)                     # N
    ay = v**2 * abs(k)                            # m/s^2
    utilisation = (v / max(v_max, 1e-9))**2       # -, lateral grip used
    FzFL = max(ph.FL_Fz(car, DF, ay, ax_prev), 0.0)   # N
    FzFR = max(ph.FR_Fz(car, DF, ay, ax_prev), 0.0)
    FzRL = max(ph.RL_Fz(car, DF, ay, ax_prev), 0.0)
    FzRR = max(ph.RR_Fz(car, DF, ay, ax_prev), 0.0)
    front_grip = tire.peak(FzFL)[1] + tire.peak(FzFR)[1]   # N
    rear_grip  = tire.peak(FzRL)[1] + tire.peak(FzRR)[1]   # N
    F_tire = ph.brake_force_limit(front_grip, rear_grip, car.brake_bias_front)   # N, limited by brake bias
    F_resist = (ph.drag(car, v)
                + ph.rolling_resistance(car, v, car.mass_total*car.g + DF)
                + ph.cornering_drag(car, ay, utilisation, tire.slip_peak))   # N, all help braking
    return min((F_tire + F_resist) / car.mass_effective, car.max_decel)


def ellipse(v, v_max):
    """Fraction of longitudinal grip left while cornering. Lateral use is
    (v/v_max)^2, so what is left is sqrt(1 - (v/v_max)^4)."""
    r = min(v / max(v_max, 1e-9), 1.0)
    return np.sqrt(max(0.0, 1.0 - r**4))


def forward_pass(track, car, tire, v_max, v0, n_loops=1, closed=True, torque_profile=None):
    """Accelerate as hard as traction allows, capped by v_max and the rev limit.
    closed=True wraps the lap (flying lap). closed=False is a standing start from v0."""
    n = len(v_max)
    v = np.minimum(np.full(n, v0), v_max)          # m/s
    v_rev = ph.vehicle_speed(car, car.rpm_cap)     # m/s, top speed at the rev limit
    for _ in range(n_loops):
        ax_prev = 0.0                              # m/s^2, load transfer lagged one step
        for i in range(n - 1):
            vi = min(v[i], v_max[i], v_rev)        # m/s
            DF = ph.downforce(car, vi)             # N
            ay = vi**2 * abs(track.k[i])           # m/s^2
            utilisation = (vi / max(v_max[i], 1e-9))**2
            FzFL = max(ph.FL_Fz(car, DF, ay, ax_prev), 0.0)   # N
            FzFR = max(ph.FR_Fz(car, DF, ay, ax_prev), 0.0)
            FzRL = max(ph.RL_Fz(car, DF, ay, ax_prev), 0.0)
            FzRR = max(ph.RR_Fz(car, DF, ay, ax_prev), 0.0)
            r = Limits.tractive_force_4wd(tire, car, FzFL, FzFR, FzRL, FzRR, max(vi, 0.1))
            e = ellipse(vi, v_max[i]) if v_max[i] < 0.999 * v_rev else 1.0   # v_max at the rev limit is not lateral grip
            F_trac = r["Fx_total"] * e   # N
            if torque_profile is not None:
                F_trac = min(F_trac, ph.wheel_force(car, torque_profile[i]))   # N, driver limited, measured torque
            F_net = (F_trac
                     - ph.drag(car, vi)
                     - ph.rolling_resistance(car, vi, car.mass_total*car.g + DF)
                     - ph.cornering_drag(car, ay, utilisation, tire.slip_peak))   # N
            ax = F_net / car.mass_effective        # m/s^2
            ax_prev = ax
            v[i + 1] = min(np.sqrt(max(vi**2 + 2 * ax * track.ds, 0.01)), v_max[i + 1], v_rev)
        if closed:
            v[0] = v[-1]                           # flying lap, start speed = end speed
        else:
            v[0] = v0                              # standing start stays a standing start
    return v


def backward_pass(track, car, tire, v_max, n_loops=1, closed=True):
    """Fastest speed at each point from which the car can still brake to the next
    point. closed=False leaves the finish line flat out, no braking after it."""
    n = len(v_max)
    v = v_max.copy()                               # m/s
    v_rev = ph.vehicle_speed(car, car.rpm_cap)     # m/s, top speed at the rev limit
    for _ in range(n_loops):
        dec_prev = 0.0                             # m/s^2, load transfer lagged one step
        for i in range(n - 1, 0, -1):
            vi = min(v[i], v_max[i])
            e = ellipse(vi, v_max[i]) if v_max[i] < 0.999 * v_rev else 1.0   # v_max at the rev limit is not lateral grip
            dec = braking_decel(car, tire, vi, v_max[i], k=track.k[i], ax_prev=-dec_prev) * e   # m/s^2
            dec_prev = dec
            v[i - 1] = min(np.sqrt(vi**2 + 2 * dec * track.ds), v_max[i - 1])
        if closed:
            v[-1] = v[0]                           # flying lap, end speed = start speed
    return v


def energy_and_time(track, car, tire, v, v_max):
    """Lap time (s), pack power (W) and pack energy (J) over the final speed profile."""
    n = len(v)
    dt = track.ds / np.maximum(v, 0.1)             # s per step
    t = np.concatenate(([0.0], np.cumsum(dt[:-1])))

    P_pack = np.zeros(n)                           # W
    for i in range(n - 1):
        vi = max(v[i], 0.1)
        DF = ph.downforce(car, vi)                 # N
        ax = (v[i + 1]**2 - v[i]**2) / (2 * track.ds)   # m/s^2
        ay = vi**2 * abs(track.k[i])               # m/s^2
        util = (vi / max(v_max[i], 1e-9))**2       # -, lateral grip used, same as forward_pass
        F_resist = (ph.drag(car, vi)
                    + ph.rolling_resistance(car, vi, car.mass_total*car.g + DF)
                    + ph.cornering_drag(car, ay, util, tire.slip_peak))   # N
        F_prop = car.mass_effective * ax + F_resist    # N, force the motors must supply
        if F_prop > 0:                                 # driving only, regen TODO
            T = ph.motor_torque(car, F_prop / 4)       # Nm per motor
            rpm = ph.motor_rpm(car, vi)
            eta = ph.motor_eff(car, rpm, T) * car.efficiency_scale   # same correction as tractive_force_4wd
            P_pack[i] = ph.pack_power(car, [T] * 4, [eta] * 4, rpm)
        # Regen block
        elif F_prop < 0 and car.regen:                 # braking: motors take what they can, friction brakes do the rest
            F_brake = -F_prop                          # N, braking force at the tyres beyond drag and rolling
            rpm = ph.motor_rpm(car, vi)
            T_F = min(ph.motor_torque(car, F_brake * car.brake_bias_front / 2), car.max_regen_torque)         # Nm per front motor
            T_R = min(ph.motor_torque(car, F_brake * (1 - car.brake_bias_front) / 2), car.max_regen_torque)   # Nm per rear motor
            T = [T_F, T_F, T_R, T_R]
            eta = [ph.motor_eff(car, rpm, t) * car.efficiency_scale for t in T]
            P_pack[i] = ph.regen_pack_power(car, T, eta, rpm)   # W, negative = energy back in the pack
    energy_J = float(np.sum(P_pack[:-1] * dt[:-1]))    # J
    return t, P_pack, energy_J


def lap_profile(track, car, tire, v_max, standing_start, torque_profile=None):
    """One lap speed profile (m/s). standing_start=True starts from rest with open
    ends, False is a flying lap wrapped twice to converge."""
    if standing_start:
        v_fwd = forward_pass(track, car, tire, v_max, v0=0.1, n_loops=1, closed=False, torque_profile=torque_profile)
        v_bwd = backward_pass(track, car, tire, v_max, n_loops=1, closed=False)
    else:
        v_fwd = forward_pass(track, car, tire, v_max, v0=v_max[0], n_loops=2, closed=True, torque_profile=torque_profile)
        v_bwd = backward_pass(track, car, tire, v_max, n_loops=2, closed=True)
    return np.minimum(v_fwd, v_bwd), v_fwd, v_bwd