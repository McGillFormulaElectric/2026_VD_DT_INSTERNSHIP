# Author: Ludih
# Summary: Physics functions for the MFE27 Lap Simulator.
#          All functions are stateless — they take a car (MFE27) instance
#          and current state values, and return forces / accelerations / limits.
#          Import this module into your solver and call functions by name.

import numpy as np

# ── Tires ──────────────────────────────────────────────────
def rolling_resistance(car, N):
    """Returns the rolling resistance force (N) at a given normal load (N) (m/s)"""
    return N * car.Crr

def traction_limit_lat(car, N):
    """Returns the lateral traction limit (N) at a given normal load (N)"""
    return N * car.mu_lat

def traction_limit_long(car, N):
    """Returns the longitudinal traction limit (N) at a given normal load (N)"""
    return N * car.mu_long

# ── Aero ──────────────────────────────────────────────────

def downforce(car, v):
    """Returns the downforce (N) at a given speed (m/s)"""
    return 0.5 * car.air_density * car.CLA * v**2

def drag(car, v):
    """Returns the drag force (N) at a given speed (m/s)"""
    return 0.5 * car.air_density * car.CDA * v**2

# ── Propulsion ────────────────────────────────────────────────

def F_power_limited(car, v):
    if v < 0.01:
        return float('inf')
    return car.peak_power/v

def F_torque_limited(car):
    return car.peak_motor_torque* 4 * car.gear_ratio / car.tire_radius

def propulsive_force(car, v, N):
    """Returns the propulsive force (N) at a given speed (m/s)"""
    return min(F_torque_limited(car), traction_limit_long(car, N), F_power_limited(car, v))

def rpm_to_speed(car, rpm):
    """Returns the speed of car (m/s) at a given motor RPM"""
    return rpm / car.gear_ratio * 2 * np.pi * car.tire_radius / 60

def speed_to_rpm(car, v):
    """Returns the motor RPM at a given speed of car (m/s)"""
    return v * car.gear_ratio / (2 * np.pi * car.tire_radius) * 60