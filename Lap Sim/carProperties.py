# Author: Ludih
# Summary: This script is used to define the properties of the car used as inputs in the Lap Sim.

from dataclasses import dataclass #Dataclasses allows us to sweep through paramters easily

@dataclass
class Car:

    # ── Mass ──────────────────────────────────────────────────
    mass_sprung:        float = 139.5   # kg  (chassis + aero + battery)
    mass_unsprung:      float = 16.0    # kg  (all 4 corners combined)
    mass_driver:        float = 70.0    # kg

    # ── Geometry ──────────────────────────────────────────────
    wheelbase:          float = 1.525   # m
    track_front:        float = 1.150   # m
    track_rear:         float = 1.150   # m
    cog_height:         float = 0.286   # m
    front_bias:         float = 0.44    # fraction of total weight on front axle

    # ── Aero ──────────────────────────────────────────────────
    CL:                 float = 3.20    # downforce coefficient (positive = down)
    CD:                 float = 1.50    # drag coefficient
    ref_area:           float = 1.10    # m²  frontal reference area
    aero_front_bias:    float = 0.45    # fraction of downforce on front axle

    # ── Grip ──────────────────────────────────────────────────
    mu_lat:             float = 1.50    # peak lateral friction coefficient
    mu_long:            float = 1.45    # peak longitudinal friction coefficient
    tire_radius:        float = 0.2032  # m  loaded radius

    # ── Powertrain ────────────────────────────────────────────
    peak_power:         float = 80_000  # W   total system peak power
    peak_torque_wheel:  float = 240.0   # Nm  total at all driven wheels
    drivetrain:         str   = "AWD"   # "AWD"  "RWD"  "FWD"
    torque_front_bias:  float = 0.50    # fraction of drive torque to front (AWD)

    # ── Brakes ────────────────────────────────────────────────
    max_decel:          float = 18.0    # m/s²  peak braking deceleration
    brake_bias_front:   float = 0.5    # fraction of brake force on front axle

    # ── Constants ─────────────────────────────────────────────
    air_density:        float = 1.225   # kg/m³
    g:                  float = 9.81    # m/s²


# ── Quick sanity check ────────────────────────────────────────
if __name__ == "__main__":
    car = Car()

    mass_total = car.mass_sprung + car.mass_unsprung + car.mass_driver

    print(f"Total mass     : {mass_total:.1f} kg")
    print(f"Weight         : {mass_total * car.g:.0f} N")
    print(f"Front bias     : {car.front_bias*100:.0f} %")
    print(f"Drivetrain     : {car.drivetrain}")
    print(f"Peak power     : {car.peak_power/1000:.0f} kW")
    print(f"Peak torque    : {car.peak_torque_wheel:.0f} Nm at wheels")
    print(f"CL / CD        : {car.CL} / {car.CD}")
    print(f"mu lat / long  : {car.mu_lat} / {car.mu_long}")
