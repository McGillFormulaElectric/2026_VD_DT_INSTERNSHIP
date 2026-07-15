# Author: Ludih and Anne-So
# Summary: Physics functions for the MFE27 Lap Simulator.
#          All functions are stateless — they take a car (MFE27) instance
#          and current state values, and return forces / accelerations / limits.
#          Import this module into your solver and call functions by name.

import numpy as np

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
    return car.mass_total * Ay * car.CG_height / car.track_front

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
    """Motor efficiency, 0 to 1. Clamped to the map, never extrapolated."""
    T = np.clip(torque, car.eta_torque[0], car.eta_torque[-1])
    N = np.clip(rpm,    car.eta_rpm[0],    car.eta_rpm[-1])
    return float(car._eta((T, N)))

def pack_power(car, torques, rpm):
    """Returns the battery pack power (W) drawn by a list of motor torques (Nm) at a given motor speed (rpm)"""
    omega = rpm * 2 * np.pi / 60
    return sum(T * omega / (motor_eff(car, rpm, T) * car.inverter_efficiency) for T in torques)

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