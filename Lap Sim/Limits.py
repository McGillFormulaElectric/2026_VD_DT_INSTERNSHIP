# Author: Anne-So
# Summary : Performance limits, corner_speed and tractive_force_4wd, 
#           each returning the lowest of three ceilings the car can reach at a given instant.

import numpy as np
import physics as ph

def tractive_force_4wd(tire, car, FzF, FzR, v, camber=0.0):
    """Returns the forward force (N) at the four contact patches, the lowest of the grip, motor torque and pack power ceilings.

    tire   : Tire object with .grip(Fz, camber) -> (drive, brake, lateral)
    car    : CarProperties
    FzF    : vertical load on ONE front wheel (N)
    FzR    : vertical load on ONE rear wheel (N)
    v     : vehicle speed (m/s)
    camber : inclination angle (deg)

    Valid for Ay = 0 only. Left and right are assumed equal, hence the 2* below.
    """
    # Motor speed.
    rpm = ph.motor_rpm(car, v)
    if rpm > car.rpm_cap:
        rpm = car.rpm_cap
        v = ph.vehicle_speed(car, rpm)
    omega = rpm * 2*np.pi/60                                   # [rad/s]

    # Grip 
    Fx_grip_F = tire.peak(FzF, camber)[0]                      # [N] per front wheel
    Fx_grip_R = tire.peak(FzR, camber)[0]                      # [N] per rear wheel

    # Motor torque
    T_max = min(np.interp(rpm, car.curve_rpm, car.curve_torque), car.torque_cap)   # [Nm] per motor,capped
    Fx_motor = ph.wheel_force(car, T_max)                      # [N] per wheel

    # Pack power
    b = car.torque_split
    norm = max(b, 1.0 - b)

    def demand(lam):
        FxF = min(lam * b/norm * Fx_motor, Fx_grip_F)          # demand, clipped by grip
        FxR = min(lam * (1-b)/norm * Fx_motor, Fx_grip_R)
        TF = ph.motor_torque(car, FxF)                         # back to motor torque [Nm]
        TR = ph.motor_torque(car, FxR)
        etaF = ph.motor_eff(car, rpm, TF)
        etaR = ph.motor_eff(car, rpm, TR)
        P_F = TF * omega
        P_R = TR * omega
        P_pack = ph.pack_power(car, [TF, TF, TR, TR], [etaF, etaF, etaR, etaR], rpm)
        return FxF, FxR, TF, TR, P_F, P_R, etaF, etaR, P_pack
    
    FxF, FxR, TF, TR, P_F, P_R, etaF, etaR, P_pack = demand(1.0)

    if P_pack > car.power_cap:                                 # too much, back it off
        lam = 1.0
        for _ in range(4):
            lam *= car.power_cap / P_pack          # nudge lam by the current error ratio
            FxF, FxR, TF, TR, P_F, P_R, etaF, etaR, P_pack = demand(lam)
            if abs(P_pack - car.power_cap) < 0.005 * car.power_cap:
                break

    return {
        "Fx_total":    2*FxF + 2*FxR,     # Ax = (Fx_total - Drag - Frolling) / mass
        "FxF":         FxF,               # per front wheel [N]
        "FxR":         FxR,               # per rear wheel  [N]
        "TmotorF":     TF,                # [Nm]
        "TmotorR":     TR,                # [Nm]
        "P_F":         P_F,               # per motor [W]
        "P_R":         P_R,               # per motor [W]
        "EfficiencyF": etaF,
        "EfficiencyR": etaR,
        "PackPower":   P_pack,            # [W]
        "rpm":         rpm,
        "v":          v,                # clipped to the rev limit
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

    return {
        "v":    v,                                  # corner speed [m/s]
        "AyG":   v**2 / (R * car.g),                 # lateral acceleration [g]
        "limit": ("grip" if v == v_grip else "power" if v == v_power else "rev"),
        "FzFL":  Fz[0],
        "FzFR":  Fz[1],
        "FzRL":  Fz[2],
        "FzRR":  Fz[3],
    }