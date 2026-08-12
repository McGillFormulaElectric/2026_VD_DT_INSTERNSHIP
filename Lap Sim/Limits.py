# Author: Anne-So
# Summary : Performance limits, corner_speed and tractive_force_4wd, 
#           each returning the lowest of three ceilings the car can reach at a given instant.

import numpy as np
import Physics as ph

def tractive_force_4wd(tire, car, FzFL, FzFR, FzRL, FzRR, v):
    """Returns the forward force (N) at the four contact patches, the lowest of the grip, motor torque and pack power ceilings.

    tire   : Tire object with .grip(Fz, camber) -> (drive, brake, lateral)
    car    : CarProperties
    FzFL    : vertical load on FRONT LEFT wheel (N)
    FzFR    : vertical load on FRONT RIGHT wheel (N)
    FzRL    : vertical load on REAR LEFT wheel (N)
    FzRR    : vertical load on REAR RIGHT wheel (N)
    v     : vehicle speed (m/s)

    Per-corner grip (valid for Ay != 0). For an Ay=0 call pass the front load as
    both FzFL/FzFR and the rear as both FzRL/FzRR — left==right falls out.
    """
    # Motor speed.
    rpm = ph.motor_rpm(car, v)
    if rpm > car.rpm_cap:
        rpm = car.rpm_cap
        v = ph.vehicle_speed(car, rpm)
    omega = rpm * 2*np.pi/60                                   # [rad/s]

    # Grip, per wheel (drive only — brake/lateral not needed here)
    g_FL = tire.peak_drive(FzFL)
    g_FR = tire.peak_drive(FzFR)
    g_RL = tire.peak_drive(FzRL)
    g_RR = tire.peak_drive(FzRR)

    # Motor torque
    T_max = min(np.interp(rpm, car.curve_rpm, car.curve_torque), car.torque_cap) * car.torque_scale   # [Nm] per motor,capped
    Fx_motor = ph.wheel_force(car, T_max)                      # [N] per wheel

    # Pack power
    b = car.torque_split
    norm = max(b, 1.0 - b)
    base_F = b/norm * Fx_motor                       # commanded force per front wheel at lam=1
    base_R = (1-b)/norm * Fx_motor                   # per rear wheel

    # efficiency cached by torque: identical torques share one lookup
    eta_cache = {}
    def eff(t):
        k = round(t, 2)
        e = eta_cache.get(k)
        if e is None:
            e = ph.motor_eff(car, rpm, t)
            eta_cache[k] = e
        return e

    def demand(lam, etas=None):
        FxFL = min(lam*base_F, g_FL)            # clip each wheel by ITS OWN grip
        FxFR = min(lam*base_F, g_FR)
        FxRL = min(lam*base_R, g_RL)
        FxRR = min(lam*base_R, g_RR)
        T = [ph.motor_torque(car, f) for f in (FxFL, FxFR, FxRL, FxRR)]
        eta = [eff(t) for t in T] if etas is None else etas   # reuse eta in back-off
        P_pack = ph.pack_power(car, T, eta, rpm)
        return [FxFL, FxFR, FxRL, FxRR], T, eta, P_pack

    Fx, T, eta, P_pack = demand(1.0)

    if P_pack > car.power_cap:                                 # too much, back it off
        lam = 1.0
        for _ in range(4):
            lam *= car.power_cap / P_pack          # nudge lam by the current error ratio
            Fx, T, eta, P_pack = demand(lam, etas=eta)   # eta ~constant as lam trims torque
            if abs(P_pack - car.power_cap) < 0.005 * car.power_cap:
                break

    return {
        "Fx_total": sum(Fx),              # Ax = (Fx_total - Drag - Frolling) / mass
        "FxFL": Fx[0], "FxFR": Fx[1], "FxRL": Fx[2], "FxRR": Fx[3],
        "Tmotor": T, "Efficiency": eta,
        "PackPower": P_pack, "rpm": rpm, "v": v,
        # legacy aliases (Ay=0 -> L==R)
        "FxF": Fx[0], "FxR": Fx[2],
        "TmotorF": T[0], "TmotorR": T[2],
        "EfficiencyF": eta[0], "EfficiencyR": eta[2],
        "P_F": T[0] * omega,      # per front motor [W]
        "P_R": T[2] * omega,      # per rear motor  [W]
    }

def corner_speed(tire, car, R, AxG=0.0):
    """Returns the maximum steady state speed (m/s) through a corner, the lowest of the grip, rev limit and pack power ceilings.

    tire : Tire object with .grip(Fz, camber) -> (drive, brake, lateral)
    car  : CarProperties
    R    : corner radius (m). Sign ignored.
    AxG  : simultaneous longitudinal acceleration (g). Zero for pure cornering,
           non zero when sweeping the friction ellipse for the g-g diagram.
    """
    R = max(abs(R), 1.0)

    # Rpm limit
    v_rpm = ph.vehicle_speed(car, car.rpm_cap)

    # Power to hold the speed
    def power_fits(v):
        DF = ph.downforce(car, v)
        Fx = ph.drag(car, v) + ph.rolling_resistance(car, v, car.mass_total * car.g + DF)
        T = ph.motor_torque(car, Fx/4)                # even split in a steady corner
        rpm = ph.motor_rpm(car, v)
        eta = ph.motor_eff(car, rpm, T)               # look up once
        return ph.pack_power(car, [T]*4, [eta]*4, rpm) <= car.power_cap   # reuse it

    # Grip 
    def grip_fits(v):
        Ay = v**2 / R                                # demand [m/s^2]
        DF = ph.downforce(car, v)
        Fz = np.clip([ph.FL_Fz(car, DF, Ay, AxG * car.g), ph.FR_Fz(car, DF, Ay, AxG*car.g), ph.RL_Fz(car, DF, Ay, AxG * car.g), ph.RR_Fz(car, DF, Ay, AxG * car.g)], 0.0, None)   # lifted wheel, no grip
        AyG_cap = sum(tire.peak(f)[2] for f in Fz) / (car.mass_total * car.g)   # [g]

        # Friction ellipse. 
        if AxG != 0.0:
            AxG_cap = sum(tire.peak(f)[0] for f in Fz) / (car.mass_total * car.g)
            inner = 1.0 - (AxG / AxG_cap)**2
            if inner <= 0.0:
                return False                          # all grip spent longitudinally
            AyG_cap *= np.sqrt(inner)

        return AyG_cap >= Ay/car.g                       # capacity >= demand

    def bisect(fits, hi):
        if fits(hi):
            return hi
        lo = 0.1
        for _ in range(40):
            mid = 0.5*(lo + hi)
            if fits(mid):
                lo = mid                              # fits, go faster
            else:
                hi = mid                              # does not fit, back off
        return lo

    v_power = bisect(power_fits, v_rpm)
    v_grip = bisect(grip_fits, v_rpm)

    v = min(v_grip, v_rpm, v_power)

    # Corner loads at the converged speed.
    Ay = v**2 / R
    DF = ph.downforce(car, v)
    Fz = np.clip([ph.FL_Fz(car, DF, Ay, AxG * car.g), ph.FR_Fz(car, DF, Ay, AxG * car.g), ph.RL_Fz(car, DF, Ay, AxG * car.g), ph.RR_Fz(car, DF, Ay, AxG * car.g)], 0.0, None)

    Fx  = ph.drag(car, v) + ph.rolling_resistance(car, v, car.mass_total * car.g + DF)
    T   = ph.motor_torque(car, Fx / 4)               # even split in a steady corner
    rpm = ph.motor_rpm(car, v)
    eta = ph.motor_eff(car, rpm, T)
    P_pack = ph.pack_power(car, [T]*4, [eta]*4, rpm)  # steady power to hold the corner [W]
    
    return {
        "v":    v,                                  # corner speed [m/s]
        "AyG":   v**2 / (R * car.g),                 # lateral acceleration [g]
        "limit": ("grip" if v == v_grip else "power" if v == v_power else "rev"),
        "P_pack":  P_pack,
        "FzFL":  Fz[0],
        "FzFR":  Fz[1],
        "FzRL":  Fz[2],
        "FzRR":  Fz[3],
    }