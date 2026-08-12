# Author: Ludih and Anne-So
# Summary: Physics functions for the MFE27 Lap Simulator.
#          All functions are stateless — they take a car (MFE27) instance
#          and current state values, and return forces / accelerations / limits.
#          Import this module into your solver and call functions by name.

import numpy as np
from scipy.interpolate import interpn

# ── Tires ──────────────────────────────────────────────────

def rolling_resistance(car, Vx, Fz):
    """Returns the rolling resistance (N) at a given speed (m/s) and normal load (N)"""
    p_bar = car.tire_pressure * 0.0689476  # psi to bar
    c = 0.005 + (1/p_bar) * (0.01 + 0.0095 * ((Vx * 3.6) /100)**2)
    return Fz * c

# ── Load Transfer ──────────────────────────────────────────────────

def long_LT_total(car, Ax):
    """Returns the longitudinal load transfer (N) at a given acceleration (m/s^2)"""
    return car.mass_total * Ax * car.CG_height / car.wheelbase

def front_long_LT(long_LT_total):
    """Returns the front right vertical load transfer (N) at a given total longitudinal load transfer (N)"""
    return - long_LT_total / 2

def rear_long_LT(long_LT_total):
    """Returns the rear right vertical load transfer (N) at a given total longitudinal load transfer (N)"""
    return long_LT_total / 2

def LLT_total(car, Ay):
    """Returns the lateral load transfer (N) at a given acceleration (m/s^2)"""
    return car.mass_total * Ay * car.CG_height / car.track

def FR_LLT(LLT_total, LLTD):
    """Returns the front right vertical load transfer (N) at a given total lateral load transfer (N)"""
    return LLT_total * LLTD

def FL_LLT(LLT_total, LLTD):
    """Returns the front left vertical load transfer (N) at a given total lateral load transfer (N)"""
    return - LLT_total * LLTD

def RR_LLT(LLT_total, LLTD):
    """Returns the rear right vertical load transfer (N) at a given total lateral load transfer (N)"""
    return LLT_total * (1 - LLTD)

def RL_LLT(LLT_total, LLTD):
    """Returns the rear left vertical load transfer (N) at a given total lateral load transfer (N)"""
    return - LLT_total * (1 - LLTD)

# ── Wheels Load ───────────────────────────────────────────

def FL_Fz(car, downforce, Ay, Ax):
    """Returns the front left vertical load (N) at a given normal load (N) and accelerations (m/s^2)"""
    return (car.mass_total * car.g * car.CGx) / 2 + (downforce * car.aero_balance) / 2 + front_long_LT(long_LT_total(car, Ax)) + FL_LLT(LLT_total(car, Ay), car.lltd)

def FR_Fz(car, downforce, Ay, Ax):
    """Returns the front right vertical load (N) at a given normal load (N) and accelerations (m/s^2)"""
    return (car.mass_total * car.g * car.CGx) / 2 + (downforce * car.aero_balance) / 2 + front_long_LT(long_LT_total(car, Ax)) + FR_LLT(LLT_total(car, Ay), car.lltd)

def RL_Fz(car, downforce, Ay, Ax):
    """Returns the rear left vertical load (N) at a given normal load (N) and accelerations (m/s^2)"""
    return (car.mass_total * car.g * (1 - car.CGx)) / 2 + (downforce * (1 - car.aero_balance)) / 2 + rear_long_LT(long_LT_total(car, Ax)) + RL_LLT(LLT_total(car, Ay), car.lltd)

def RR_Fz(car, downforce, Ay, Ax):
    """Returns the rear right vertical load (N) at a given normal load (N) and accelerations (m/s^2)"""
    return (car.mass_total * car.g * (1 - car.CGx)) / 2 + (downforce * (1 - car.aero_balance)) / 2 + rear_long_LT(long_LT_total(car, Ax)) + RR_LLT(LLT_total(car, Ay), car.lltd)

# ── Aero ──────────────────────────────────────────────────

def downforce(car, v):
    """Returns the downforce (N) at a given speed (m/s)"""
    return 0.5 * car.air_density * car.CLA * v**2

def drag(car, v):
    """Returns the drag force (N) at a given speed (m/s)"""
    return 0.5 * car.air_density * car.CDA * v**2

#sideforce

# ── PowerTrain ──────────────────────────────────────────────────

def motor_eff(car, rpm, torque):
    """Motor efficiency (0..1), bilinear from the map, clamped to edges.
    Hand-rolled 2x2 lookup — avoids scipy per-call overhead in the solver loop."""
    Tg, Ng, M = car.eta_torque, car.eta_rpm, car.eta_map    # M[i,j] rows=torque, cols=rpm
    T = min(max(abs(torque), Tg[0]), Tg[-1])
    N = min(max(rpm,          Ng[0]), Ng[-1])

    i = np.searchsorted(Tg, T) - 1
    j = np.searchsorted(Ng, N) - 1
    i = min(max(i, 0), len(Tg) - 2)
    j = min(max(j, 0), len(Ng) - 2)

    tT = (T - Tg[i]) / (Tg[i+1] - Tg[i])
    tN = (N - Ng[j]) / (Ng[j+1] - Ng[j])

    m00, m01 = M[i,   j], M[i,   j+1]
    m10, m11 = M[i+1, j], M[i+1, j+1]
    return float((m00*(1-tT)*(1-tN) + m10*tT*(1-tN)
                + m01*(1-tT)*tN     + m11*tT*tN))
    
def pack_power(car, torques, etas, rpm):
    """Returns the battery pack power (W) drawn by a list of motor torques (Nm) at a given motor speed (rpm)"""
    omega = rpm * 2 * np.pi / 60
    return sum(T * omega / (eta * car.inverter_efficiency) for T, eta in zip(torques, etas))

def motor_rpm(car, Vx):
    """Returns the motor speed (rpm) at a given vehicle speed (m/s)"""
    return (Vx / car.tire_radius) * (60 / (2 * np.pi)) * car.gear_ratio

def vehicle_speed(car, rpm):
    """Returns the vehicle speed (m/s) at a given motor speed (rpm)"""
    return rpm * (2*np.pi/60) / car.gear_ratio * car.tire_radius

def motor_torque(car, Fx):
    """Returns the motor torque (Nm) required to produce a given force at one contact patch (N)"""
    return Fx * car.tire_radius / car.gear_ratio

def wheel_force(car, T):
    """Returns the force at one contact patch (N) produced by a given motor torque (Nm)"""
    return T * car.gear_ratio / car.tire_radius

# ── Cornering (induced) drag ───────────────────────────────

def cornering_drag(car, Ay, utilisation):
    """Returns the induced 'tyre' drag (N) from carrying lateral force Ay (m/s^2) at a slip angle that grows with lateral utilisation util (0 to 1)"""
    slip = np.radians(car.slip_peak) * min(max(utilisation, 0.0), 1.0)
    return abs(car.mass_total * Ay) * np.sin(slip)

# ── Braking ────────────────────────────────────────────────

def brake_force_limit(front_grip, rear_grip, brake_bias):
    """Returns the max total braking force (N) with a fixed front brake_bias (0 to 1), capped by whichever axle saturates first: min(front_grip/bias, rear_grip/(1-bias))"""
    b = min(max(brake_bias, 1e-6), 1.0 - 1e-6)
    return min(front_grip / b, rear_grip / (1.0 - b))