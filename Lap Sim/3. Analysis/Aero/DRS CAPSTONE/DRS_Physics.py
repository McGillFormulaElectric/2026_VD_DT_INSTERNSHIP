"""DRS (drag reduction system) for the QSS lap sim.

Two aero states, closed and open, each a CarProperties-like object with its own
CLA, CDA and aero_balance (and the DRS mass on both). The wing is only ever open
under power, never under braking, so:

  corner ceiling  -> closed aero (lateral grip)
  backward pass   -> closed aero (braking)   = SolverFunctions.backward_pass, untouched
  forward pass    -> state chosen per point by the controller (below)
  energy_and_time -> drag and loads follow the per-point state

Controller modes
  "controller"  open when NOT grip limited: with the wing OPEN the longitudinal
                tyre grip left by the friction ellipse still exceeds what the
                motors / 80 kW cap can deliver, AND lateral utilisation with the
                open-wing corner ceiling is below lat_margin. This is what the
                car's controller will do (power limited -> open).
  "ideal"       open wherever the open state gives more net force at that point.
                The upper bound in the capstone notebook ("whichever is faster").

Latency t_act [s]
  opening: the wing counts as closed until the car has travelled v*t_act after
           the open request (conservative, no partial-open aero)
  closing: the wing is forced closed for a distance v*t_act ahead of every
           braking zone so it is shut before the brakes go on. Braking zones
           come from the first profile, so the lap is solved twice.
"""

import copy
import numpy as np
import Physics as ph
import Limits
from SolverFunctions import corner_speed_ceiling, backward_pass, ellipse


# Add DRS to a car object, returning two Car Properties; one for DRS open and one for closed, describing the two car states
def make_states(car_base, CLA_closed, CDA_closed, bal_closed,
                CLA_open, CDA_open, bal_open, drs_mass=0.0):
    """Returns (car_closed, car_open). Both carry the DRS mass."""
    closed = copy.deepcopy(car_base)
    closed.mass_no_driver = car_base.mass_no_driver + drs_mass
    closed.CLA, closed.CDA, closed.aero_balance = CLA_closed, CDA_closed, bal_closed
    open = copy.deepcopy(closed)
    open.CLA, open.CDA, open.aero_balance = CLA_open, CDA_open, bal_open
    return closed, open #The two cars (two states, one with DRS open, one with DRS closed)


# ── grip-only corner ceiling (no rev / power limit) ──────────────────────
def grip_ceiling(track, car, tire, n_grid=40):
    """Lateral-grip speed limit (m/s) at every point, ignoring the rev and power
    limits. Same bisect as Limits.corner_speed.grip_fits. Used by the controller so
    'at the ceiling' on a rev-limited straight is not mistaken for 'at the grip limit'."""
    def fits(v, R):
        Ay = v**2 / R
        DF = ph.downforce(car, v)
        Fz = np.clip([ph.FL_Fz(car, DF, Ay, 0.0), ph.FR_Fz(car, DF, Ay, 0.0),
                      ph.RL_Fz(car, DF, Ay, 0.0), ph.RR_Fz(car, DF, Ay, 0.0)], 0.0, None)
        return sum(tire.peak(f)[2] for f in Fz) >= car.mass_total * Ay
    def vgrip(R):
        lo, hi = 0.1, 120.0
        if fits(hi, R):
            return hi
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if fits(mid, R):
                lo = mid
            else:
                hi = mid
        return lo
    k_abs = np.abs(track.k)
    k_lo, k_hi = 1e-4, max(k_abs.max(), 1e-3)
    k_grid = np.geomspace(k_lo, k_hi, n_grid)
    v_grid = np.array([vgrip(1.0 / kk) for kk in k_grid])
    return np.interp(np.clip(k_abs, k_lo, None), k_grid, v_grid)


# ── one point, one aero state ────────────────────────────────────────────
def _step(car, tire, vi, k, v_max_i, v_rev, ax_prev):
    """Net longitudinal force (N) at speed vi for one aero state, plus whether the
    tyres (after the friction ellipse) are what limits the drive force."""
    DF = ph.downforce(car, vi)
    ay = vi**2 * abs(k)
    util = (vi / max(v_max_i, 1e-9))**2
    Fz = [max(ph.FL_Fz(car, DF, ay, ax_prev), 0.0), max(ph.FR_Fz(car, DF, ay, ax_prev), 0.0),
          max(ph.RL_Fz(car, DF, ay, ax_prev), 0.0), max(ph.RR_Fz(car, DF, ay, ax_prev), 0.0)]
    r = Limits.tractive_force_4wd(tire, car, *Fz, max(vi, 0.1))
    e = ellipse(vi, v_max_i) if v_max_i < 0.999 * v_rev else 1.0
    F_trac = r["Fx_total"] * e

    # motor / power ceiling with unlimited grip: same call with the tyre table saturated
    big = tire.Fz_max
    F_pwr = Limits.tractive_force_4wd(tire, car, big, big, big, big, max(vi, 0.1))["Fx_total"]
    F_grip = sum(tire.peak_drive(f) for f in Fz) * e
    grip_limited = F_grip < F_pwr

    F_net = (F_trac - ph.drag(car, vi)
             - ph.rolling_resistance(car, vi, car.mass_total * car.g + DF)
             - ph.cornering_drag(car, ay, util, tire.slip_peak))
    return F_net, grip_limited, r["PackPower"]


# ── forward pass with DRS ────────────────────────────────────────────────
def forward_pass_drs(track, car_c, car_o, tire, vmax_c, vmax_o, v0, n_loops, closed,
                     mode="controller", t_act=0.0, lat_margin=0.95, close_mask=None):
    n = len(vmax_c)
    v = np.minimum(np.full(n, v0), vmax_c)
    state = np.zeros(n, dtype=bool)
    v_rev = ph.vehicle_speed(car_c, car_c.rpm_cap)
    for _ in range(n_loops):
        ax_prev = 0.0
        pending = 0.0                      # m travelled since the open request
        for i in range(n - 1):
            vi = min(v[i], vmax_c[i], v_rev)
            Fc, _, _ = _step(car_c, tire, vi, track.k[i], vmax_c[i], v_rev, ax_prev)
            Fo, grip_lim_o, _ = _step(car_o, tire, vi, track.k[i], vmax_c[i], v_rev, ax_prev)

            if mode == "ideal":
                want = Fo > Fc
            elif mode == "controller":
                want = (not grip_lim_o) and (vi < lat_margin * vmax_o[i])
            elif mode == "closed":
                want = False
            else:
                raise ValueError(mode)
            if close_mask is not None and close_mask[i]:
                want = False

            if want:
                pending += track.ds
                is_open = pending >= vi * t_act
            else:
                pending = 0.0
                is_open = False
            state[i] = is_open

            F = Fo if is_open else Fc
            car = car_o if is_open else car_c
            ax = F / car.mass_effective
            ax_prev = ax
            v[i + 1] = min(np.sqrt(max(vi**2 + 2 * ax * track.ds, 0.01)), vmax_c[i + 1], v_rev)
        if closed:
            v[0] = v[-1]
        else:
            v[0] = v0
    state[-1] = state[-2]
    return v, state


def braking_close_mask(track, v, v_fwd, t_act):
    """True where the wing must be shut: inside braking zones (final speed below the
    free forward speed) and for a distance v*t_act ahead of each zone."""
    n = len(v)
    braking = v < v_fwd - 1e-3
    mask = braking.copy()
    idx = np.where(braking[1:] & ~braking[:-1])[0] + 1       # zone onsets
    for i0 in idx:
        lead = v[i0] * t_act
        j = i0
        while j > 0 and (track.s[i0] - track.s[j]) < lead:
            j -= 1
            mask[j] = True
    return mask


def lap_profile_drs(track, car_c, car_o, tire, vmax_c, vmax_o, standing_start,
                    mode="controller", t_act=0.0, lat_margin=0.95):
    if standing_start:
        kw = dict(v0=0.1, n_loops=1, closed=False)
        v_bwd = backward_pass(track, car_c, tire, vmax_c, n_loops=1, closed=False)
    else:
        kw = dict(v0=vmax_c[0], n_loops=2, closed=True)
        v_bwd = backward_pass(track, car_c, tire, vmax_c, n_loops=2, closed=True)
    # pass 1: find braking zones
    v_fwd, state = forward_pass_drs(track, car_c, car_o, tire, vmax_c, vmax_o,
                                    mode=mode, t_act=t_act, lat_margin=lat_margin, **kw)
    v = np.minimum(v_fwd, v_bwd)
    mask = braking_close_mask(track, v, v_fwd, t_act)
    # pass 2: wing shut ahead of the brakes
    v_fwd, state = forward_pass_drs(track, car_c, car_o, tire, vmax_c, vmax_o,
                                    mode=mode, t_act=t_act, lat_margin=lat_margin,
                                    close_mask=mask, **kw)
    v = np.minimum(v_fwd, v_bwd)
    state = state & ~(v < v_fwd - 1e-3)          # never open while braking
    return v, state, v_fwd, v_bwd


# ── energy with per-point state ──────────────────────────────────────────
def energy_and_time_drs(track, car_c, car_o, tire, v, vmax_c, state):
    n = len(v)
    dt = track.ds / np.maximum(v, 0.1)
    t = np.concatenate(([0.0], np.cumsum(dt[:-1])))
    P = np.zeros(n)
    for i in range(n - 1):
        car = car_o if state[i] else car_c
        vi = max(v[i], 0.1)
        DF = ph.downforce(car, vi)
        ax = (v[i + 1]**2 - v[i]**2) / (2 * track.ds)
        ay = vi**2 * abs(track.k[i])
        util = (vi / max(vmax_c[i], 1e-9))**2
        F_res = (ph.drag(car, vi) + ph.rolling_resistance(car, vi, car.mass_total * car.g + DF)
                 + ph.cornering_drag(car, ay, util, tire.slip_peak))
        F_prop = car.mass_effective * ax + F_res
        rpm = ph.motor_rpm(car, vi)
        if F_prop > 0:
            T = ph.motor_torque(car, F_prop / 4)
            eta = ph.motor_eff(car, rpm, T) * car.efficiency_scale
            P[i] = ph.pack_power(car, [T] * 4, [eta] * 4, rpm)
        elif F_prop < 0 and car.regen:
            F_b = -F_prop
            T_F = min(ph.motor_torque(car, F_b * car.brake_bias_front / 2), car.max_regen_torque)
            T_R = min(ph.motor_torque(car, F_b * (1 - car.brake_bias_front) / 2), car.max_regen_torque)
            T = [T_F, T_F, T_R, T_R]
            eta = [ph.motor_eff(car, rpm, tt) * car.efficiency_scale for tt in T]
            P[i] = ph.regen_pack_power(car, T, eta, rpm)
    E = float(np.sum(P[:-1] * dt[:-1]))
    return t, P, E


# ── events ───────────────────────────────────────────────────────────────
def solve_track_event(tire, car_c, car_o, track, standing_start, mode="controller",
                      t_act=0.0, lat_margin=0.95):
    vmax_c = corner_speed_ceiling(track, car_c, tire)
    # open-wing LATERAL GRIP ceiling only: the lateral-margin check must not fire on a
    # rev- or power-limited straight, where v sits at the ceiling without using grip
    vmax_o = grip_ceiling(track, car_o, tire) if mode != "closed" else vmax_c
    v, state, v_fwd, v_bwd = lap_profile_drs(track, car_c, car_o, tire, vmax_c, vmax_o,
                                             standing_start, mode, t_act, lat_margin)
    t, P, E = energy_and_time_drs(track, car_c, car_o, tire, v, vmax_c, state)
    lap_time = t[-1] + track.ds / max(v[-1], 0.1)
    return {"s": track.s, "v": v, "state": state, "v_max": vmax_c, "v_max_open": vmax_o,
            "v_fwd": v_fwd, "v_bwd": v_bwd, "t": t, "P_pack": P,
            "lap_time": lap_time, "energy_J": E,
            "open_fraction": float(np.mean(state))}


def solve_accel(tire, car_c, car_o, mode="controller", t_act=0.0,
                track_length=75.0, run_up=0.3):
    """Acceleration.py loop with the DRS state chosen each step. No lateral term."""
    def step_size(k):
        return 1e-4 if k < 1000 else 1e-3 if k < 2000 else 1e-2
    v, s, ax_prev, k = 0.0, -run_up, 0.0, 0
    pending = 0.0
    log_s, log_v, log_state, log_P, log_dt = [], [], [], [], []
    while s < track_length and k <= 200_000:
        ds = step_size(k)
        res = {}
        for name, car in (("c", car_c), ("o", car_o)):
            DF = ph.downforce(car, v)
            FzF = max(ph.FL_Fz(car, DF, 0.0, ax_prev), 0.0)
            FzR = max(ph.RL_Fz(car, DF, 0.0, ax_prev), 0.0)
            r = Limits.tractive_force_4wd(tire, car, FzF, FzF, FzR, FzR, max(v, 0.1))
            big = tire.Fz_max
            F_pwr = Limits.tractive_force_4wd(tire, car, big, big, big, big, max(v, 0.1))["Fx_total"]
            F_grip = 2 * tire.peak_drive(FzF) + 2 * tire.peak_drive(FzR)
            F_net = r["Fx_total"] - ph.drag(car, v) - ph.rolling_resistance(car, v, 2 * FzF + 2 * FzR)
            res[name] = (F_net, F_grip < F_pwr, r["PackPower"], r["v"])
        if mode == "ideal":
            want = res["o"][0] > res["c"][0]
        elif mode == "controller":
            want = not res["o"][1]
        else:
            want = False
        if want:
            pending += ds
            is_open = pending >= v * t_act
        else:
            pending, is_open = 0.0, False
        F_net, _, P, v_clamp = res["o"] if is_open else res["c"]
        car = car_o if is_open else car_c
        v = v_clamp
        ax = F_net / car.mass_effective
        v_next = np.sqrt(max(v**2 + 2 * ax * ds, 0.0))
        v_bar = 0.5 * (v + v_next)
        if v_bar > 1e-6:
            dt = ds / v_bar
        elif ax > 0:
            dt = np.sqrt(2 * ds / ax)
        else:
            break
        log_s.append(s); log_v.append(v); log_state.append(is_open); log_P.append(P); log_dt.append(dt)
        ax_prev = ax
        v = v_next
        s += ds
        k += 1
    dt = np.asarray(log_dt)
    return {"s": np.asarray(log_s), "v": np.asarray(log_v), "state": np.asarray(log_state),
            "t": np.concatenate(([0.0], np.cumsum(dt)[:-1])), "P_pack": np.asarray(log_P),
            "lap_time": float(dt.sum()), "energy_J": float(np.sum(np.asarray(log_P) * dt)),
            "open_fraction": float(np.mean(log_state))}