# Author: Ludih
# Summary: This script is used to define the properties of the car used as inputs in the Lap Sim.

from dataclasses import dataclass #Dataclasses allows us to sweep through paramters easily
from scipy.io import loadmat
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path

DATA = Path(__file__).parent / "Powertrain_mat"

@dataclass
class MFE27:

    # ── Geometry ──────────────────────────────────────────────
    wheelbase:          float = 1.525   # m
    track:              float = 1.150   # m
    lltd:               float = 0.474   # m  lateral load transfer distribution (front)
    CG_height:          float = 0.286   # m
    CGx:                float = 0.44    # fraction of total weight on front axle

    # ── Mass ──────────────────────────────────────────────────
    mass_battery:       float = 46.0  # kg  battery pack mass 
    mass_driver:        float = 70.0  # kg  driver mass
    mass_aero:          float = 16.0  # kg  aerodynamic mass
    mass_unsprung_f:    float = 32.0  # kg  unsprung mass (front)
    mass_unsprung_r:    float = 32.0  # kg  unsprung mass (rear)
    mass_no_driver:     float = 203.0 # kg  total mass without driver

    @property
    def mass_total(self):  # kg  total mass of the car with driver
        return self.mass_no_driver + self.mass_driver

    @property
    def mass_sprung(self):  # kg  (chassis + aero + battery)
        return self.mass_total - self.mass_unsprung_f - self.mass_unsprung_r
    
    @property
    def mass_sprung_front(self):  # kg  (chassis + aero + battery)
        return self.mass_sprung * self.CGx_front_bias
    
    @property
    def mass_sprung_rear(self):  # kg  (chassis + aero + battery)
        return self.mass_sprung * (1-self.CGx_front_bias)

    # ── Tires ──────────────────────────────────────────────────
    mu_lat:             float = 1.50    # peak lateral friction coefficient
    mu_long:            float = 1.45    # peak longitudinal friction coefficient
    tire_radius:        float = 0.2032  # m  loaded radius
    tire_pressure:      float = 13      # psi tire pressure cold

    # ── Suspension ────────────────────────────────────────────
    RC_height_front:         float = 0.0866   # m  front roll center height
    RC_height_rear:          float = 0.13437  # m  rear roll center height
    RC_height_sprung:        float = 0.1100   # m  sprung roll center height

    # ── Aero ──────────────────────────────────────────────────
    CL:                 float = 4.26    # downforce coefficient (positive = down)
    CD:                 float = 1.77    # drag coefficient
    ref_area:           float = 1.10    # m²  frontal reference area
    aero_balance:       float = 0.45    # fraction of downforce on front axle

    @property
    def CLA(self): # downforce coefficient (positive = down)
        return self.CL * self.ref_area
    
    @property
    def CDA(self):
        return self.CD * self.ref_area


    # ── Powertrain ────────────────────────────────────────────
    max_power:            float = 80_000  # W   total system peak power
    peak_motor_torque:     float = 9        # Nm  at motor
    torque_split:          float = 0.50    # fraction of torque to front (AWD)
    inverter_efficiency:   float = 0.98    # fraction of power delivered to motor
    motor_rpm_max:         float = 18_000  # rpm  max motor speed

    drivetrain:            str   = "AWD"   # "AWD"  "RWD"  "FWD"
    gear_ratio:            float = 13.39     # final drive ratio
    regen:                 bool = True      # True if regen braking is enabled
    regen_torque:          float = 1        # Nm  at motor
    
    def __post_init__(self):
        # motor efficiency map, 2D
        m = loadmat(DATA / "AMK_Efficiency.mat")
        self.eta_torque = np.squeeze(m["Torque"]).astype(float)      # (12,)   [Nm]
        self.eta_rpm    = np.squeeze(m["RPM"]).astype(float)         # (11,)   [rpm]
        self.eta_map    = np.asarray(m["Efficiency"], dtype=float)   # (12,11) rows=torque, cols=rpm
        self.eta_interp = RegularGridInterpolator((self.eta_torque, self.eta_rpm), self.eta_map, method="linear")

        # motor torque envelope, 1D
        c = loadmat(DATA / "AMK_MotorCurve.mat")
        self.curve_rpm    = np.squeeze(c["RPM"]).astype(float)       # (11,)   [rpm]
        self.curve_torque = np.squeeze(c["Tmotor"]).astype(float)    # (11,)   [Nm]

    # ── Brakes ────────────────────────────────────────────────
    max_decel:          float = 18.0    # m/s²  peak braking deceleration
    brake_bias_front:   float = 0.5    # fraction of brake force on front axle

    
    # ── Constants ─────────────────────────────────────────────
    air_density:        float = 1.225   # kg/m³
    g:                  float = 9.81    # m/s²
