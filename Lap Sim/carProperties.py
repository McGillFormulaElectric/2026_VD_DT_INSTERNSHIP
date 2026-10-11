from dataclasses import dataclass
from scipy.io import loadmat
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path

DATA = Path(__file__).parent / "1. Data" / "PowerTrain_mat" 

@dataclass
class MFE26:
    # ── Geometry ──────────────────────────────────────────────
    wheelbase:          float = 1.525   # m
    track:              float = 1.150   # m
    lltd:               float = 0.474   # m  lateral load transfer distribution (front)
    CG_height:          float = 0.286   # m
    CGx:                float = 0.4437    # fraction of total weight on front axle
    # ── Mass ──────────────────────────────────────────────────
    mass_battery:       float = 46.0  # kg  battery pack mass 
    mass_driver:        float = 70.0  # kg  driver mass
    mass_aero:          float = 16.0  # kg  aerodynamic mass
    mass_unsprung_f:    float = 32.0  # kg  unsprung mass (front)
    mass_unsprung_r:    float = 32.0  # kg  unsprung mass (rear)
    mass_no_driver:     float = 203.0 # kg  total mass without driver
    J_motor:            float = 0.000274  # kg m²  rotor inertia per motor, check the AMK datasheet
    J_wheel:            float = 0.10      # kg m²  wheel + tyre + hub per corner, estimate or measure
    
    @property
    def mass_total(self):  # kg  total mass of the car with driver
        return self.mass_no_driver + self.mass_driver

    @property
    def mass_sprung(self):  # kg  (chassis + aero + battery)
        return self.mass_total - self.mass_unsprung_f - self.mass_unsprung_r
    
    @property
    def mass_sprung_front(self):  # kg  (chassis + aero + battery)
        return self.mass_sprung * self.CGx
    
    @property
    def mass_sprung_rear(self):  # kg  (chassis + aero + battery)
        return self.mass_sprung * (1-self.CGx)
    
    @property
    def mass_effective(self):  # kg  mass + spinning parts seen at the wheels, for longitudinal accel only
        return self.mass_total + 4 * (self.J_motor * self.gear_ratio**2 + self.J_wheel) / self.tire_radius**2
    
    # ── Tires ──────────────────────────────────────────────────
    camber_front:       float = 0    # deg  front camber angle
    camber_rear:        float = 0    # deg  rear camber angle
    tire_radius:        float = 0.201  # m  loaded radius
    tire_pressure:      float = 13      # psi tire pressure hot
    # ── Suspension ────────────────────────────────────────────
    RC_height_front:         float = 0.0866   # m  front roll center height
    RC_height_rear:          float = 0.13437  # m  rear roll center height
    RC_height_sprung:        float = 0.1100   # m  sprung roll center height
    # ── Aero ──────────────────────────────────────────────────
    CLA:                float = 4.44   # downforce coefficient (positive = down)
    CDA:                float = 1.5    #1.85   # drag coefficient
    CS:                 float = 0.08    # sideforce coefficient
    ref_area:           float = 1.10    # m²  frontal reference area
    aero_balance:       float = 0.45    # fraction of downforce on front axle
    
    @property
    def CSA(self):
        return self.CS * self.ref_area

    # ── Powertrain ────────────────────────────────────────────
    motor_type:            str   = "AMK"    # "AMK" or "Fisher"
    power_cap:             float = 80_000   # W   total system peak power
    torque_cap:            float = 21       # torque cap of motor [Nm]
    torque_split:          float = 0.50     # fraction of torque to front (AWD)
    inverter_efficiency:   float = 0.98     # fraction of power delivered to motor
    rpm_cap:               float = 19_000   # rpm  cap of the motor 
    torque_scale:          float = 1.0      # fraction of rated torque actually delivered (correlation)
    efficiency_scale:      float = 0.95     # pack to shaft correction from accel.mat (0.82 measured vs 0.873 sim)
    drivetrain:            str   = "AWD"    # "AWD"  "RWD"  "FWD"
    gear_ratio:            float = 13.39    # final drive ratio
    regen:                 bool = True      # True if regen braking is enabled
    max_regen_torque:      float = 10       # Nm  at motor
    
    def __post_init__(self):
        # motor efficiency map, 2D, efficiency as a FRACTION
        m = loadmat(DATA/ self.motor_type / (self.motor_type + "_Efficiency.mat"))
        self.eta_torque = np.squeeze(m["Torque"]).astype(float)      # (12,)   [Nm]
        self.eta_rpm    = np.squeeze(m["RPM"]).astype(float)         # (11,)   [rpm]
        self.eta_map    = np.asarray(m["Efficiency"], dtype=float)   # (12,11) rows=torque, cols=rpm
        self.eta_interp = RegularGridInterpolator((self.eta_torque, self.eta_rpm), self.eta_map, method="linear") #Might have to add a clipping function to not interpolate out of range and crash

        # motor torque envelope, 1D
        c = loadmat(DATA/ self.motor_type / (self.motor_type + "_MotorCurve.mat"))
        self.curve_rpm    = np.squeeze(c["RPM"]).astype(float)       # (11,)   [rpm]
        self.curve_torque = np.squeeze(c["Tmotor"]).astype(float)    # (11,)   [Nm] 

    # ── Brakes ────────────────────────────────────────────────
    max_decel:          float = 25.0    # m/s²  peak braking deceleration
    brake_bias_front:   float = 0.6    # fraction of brake force on front axle
    # ── Constants ─────────────────────────────────────────────
    air_density:        float = 1.225   # kg/m³
    g:                  float = 9.81    # m/s²
    
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
    mass_no_driver:     float = 185.0 # kg  total mass without driver
    J_motor:            float = 0.00033 # kg m²  rotor inertia per motor, check the AMK datasheet
    J_wheel:            float = 0.10      # kg m²  wheel + tyre + hub per corner, estimate or measure

    @property
    def mass_total(self):  # kg  total mass of the car with driver
        return self.mass_no_driver + self.mass_driver

    @property
    def mass_sprung(self):  # kg  (chassis + aero + battery)
        return self.mass_total - self.mass_unsprung_f - self.mass_unsprung_r
    
    @property
    def mass_sprung_front(self):  # kg  (chassis + aero + battery)
        return self.mass_sprung * self.CGx
    
    @property
    def mass_sprung_rear(self):  # kg  (chassis + aero + battery)
        return self.mass_sprung * (1-self.CGx)
    
    @property
    def mass_effective(self):  # kg  mass + spinning parts seen at the wheels, for longitudinal accel only
        return self.mass_total + 4 * (self.J_motor * self.gear_ratio**2 + self.J_wheel) / self.tire_radius**2

    # ── Tires ──────────────────────────────────────────────────
    camber_front:       float = 0    # deg  front camber angle
    camber_rear:        float = 0    # deg  rear camber angle
    tire_radius:        float = 0.198  # m  loaded radius
    tire_pressure:      float = 13      # psi tire pressure hot
    # ── Suspension ────────────────────────────────────────────
    RC_height_front:         float = 0.0866   # m  front roll center height
    RC_height_rear:          float = 0.13437  # m  rear roll center height
    RC_height_sprung:        float = 0.1100   # m  sprung roll center height
    # ── Aero ──────────────────────────────────────────────────
    CLA:                float = 4.44    # downforce coefficient (positive = down)
    CDA:                float = 1.85    # drag coefficient
    CS:                 float = 0.08    # sideforce coefficient
    ref_area:           float = 1.10    # m²  frontal reference area
    aero_balance:       float = 0.45    # fraction of downforce on front axle
    
    @property
    def CSA(self):
        return self.CS * self.ref_area

    # ── Powertrain ────────────────────────────────────────────
    motor_type:            str   = "Fisher"    # "AMK" or "Fisher"
    power_cap:             float = 80_000   # W   total system peak power
    torque_cap:            float = 29        # torque cap of motor [Nm]
    torque_split:          float = 0.50     # fraction of torque to front (AWD)
    inverter_efficiency:   float = 0.98     # fraction of power delivered to motor
    rpm_cap:               float = 19_000   # rpm  cap of the motor 
    torque_scale:          float = 1.0      # fraction of rated torque actually delivered (correlation)
    efficiency_scale:      float = 0.95     # pack to shaft correction from accel.mat (0.82 measured vs 0.873 sim)
    drivetrain:            str   = "AWD"    # "AWD"  "RWD"  "FWD"
    gear_ratio:            float = 12       # final drive ratio
    regen:                 bool = False     # True if regen braking is enabled
    max_regen_torque:      float = 10       # Nm  at motor
    
    def __post_init__(self):
        # motor efficiency map, 2D, efficiency as a FRACTION
        m = loadmat(DATA/ self.motor_type / (self.motor_type + "_Efficiency_Fraction.mat"))
        self.eta_torque = np.squeeze(m["Torque"]).astype(float)      # (12,)   [Nm]
        self.eta_rpm    = np.squeeze(m["RPM"]).astype(float)         # (11,)   [rpm]
        self.eta_map    = np.asarray(m["Efficiency"], dtype=float)   # (12,11) rows=torque, cols=rpm
        self.eta_interp = RegularGridInterpolator((self.eta_torque, self.eta_rpm), self.eta_map, method="linear") #Might have to add a clipping function to not interpolate out of range and crash

        # motor torque envelope, 1D
        c = loadmat(DATA/ self.motor_type / (self.motor_type + "_MotorCurve.mat"))
        self.curve_rpm    = np.squeeze(c["RPM"]).astype(float)       # (11,)   [rpm]
        self.curve_torque = np.squeeze(c["Tmotor"]).astype(float)    # (11,)   [Nm] 

    # ── Brakes ────────────────────────────────────────────────
    max_decel:          float = 28.0    # m/s²  peak braking deceleration
    brake_bias_front:   float = 0.5    # fraction of brake force on front axle
    # ── Constants ─────────────────────────────────────────────
    air_density:        float = 1.225   # kg/m³
    g:                  float = 9.81    # m/s²