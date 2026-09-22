# Author: Ludih
# Summary: This script is used to define the properties of the car used as inputs in the Lap Sim.

from dataclasses import dataclass #Dataclasses allows us to sweep through paramters easily
from scipy.io import loadmat
import numpy as np
from pathlib import Path

DATA = Path(__file__).parent / "Data" / "PowerTrain_mat" 

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
    mass_aero:          float = 16.0  # kg  aerodynamic mass
    mass_chassis_sprung: float = 77 #kg  everything except the items above (203 - 46 - 16 - 64), unsprung not included
    mass_unsprung_f:    float = 32.0 #kg
    mass_unsprung_r:    float = 32.0 #kg

    mass_driver:        float = 70.0  # kg  driver mass

    I_wheel:            float = 0.3 #kgm^2 rotational inertia of wheel

    @property
    def mass_total(self):  # kg  total mass of the car with driver
        return self.mass_battery + self.mass_aero + self.mass_chassis_sprung + self.mass_driver+ self.mass_unsprung_f + self.mass_unsprung_r

    @property
    def effective_mass(self): #Only for F=ma not load transfer
        return (self.mass_total
            + 4 * self.I_wheel / self.tire_radius**2) 
    
    @property
    def mass_sprung(self):  # kg  (chassis + aero + battery)
        return self.mass_total - self.mass_unsprung_f - self.mass_unsprung_r
    
    @property
    def mass_sprung_front(self):  # kg  (chassis + aero + battery+ driver)
        return self.mass_sprung * self.CGx
    
    @property
    def mass_sprung_rear(self):  # kg  (chassis + aero + battery)
        return self.mass_sprung * (1-self.CGx)

    # ── Tires ──────────────────────────────────────────────────
    
    #camber_front:       float = 0    # deg  front camber angle (Not in lapsim)
    #camber_rear:        float = 0    # deg  rear camber angle  (Not in lapsim)
    tire_radius:        float = 0.2032  # m  loaded radius
    tire_pressure:      float = 12      # psi tire pressure hot
    slip_peak:          float = 7.0     # deg  slip angle at peak lateral force (Fy)

    # ── Suspension ────────────────────────────────────────────
    #Not used currently
    RC_height_front:         float = 0.0866   # m  front roll center height
    RC_height_rear:          float = 0.13437  # m  rear roll center height
    RC_height_sprung:        float = 0.1100   # m  sprung roll center height

    # ── Aero ──────────────────────────────────────────────────
    CL:                 float = 4.26    # downforce coefficient (positive = down)
    CD:                 float = 1.77    # drag coefficient
    ref_area:           float = 1.10    # m²  frontal reference area
    aero_balance:       float = 0.45    # fraction of downforce on front axle
    cop_z:              float = 0.286    # m  center of pressure height, not used right now

    @property
    def CLA(self): # downforce coefficient (positive = down)
        return self.CL * self.ref_area
    
    @property
    def CDA(self):
        return self.CD * self.ref_area


    # ── Powertrain ────────────────────────────────────────────
    motor_type:            str   = "Fisher"    # "AMK" or "Fisher"
    power_cap:             float = 80_000   # W   total system peak power
    torque_cap:            float = 29        # torque cap of motor [Nm], if capped in software
    inverter_efficiency:   float = 0.98     # fraction of power delivered to motor
    rpm_cap:               float = 20_000   # rpm  cap of the motor 
    torque_scale:          float = 1.0      # fraction of rated torque actually delivered (correlation)
    torque_split:           float = 0.5    

    drivetrain:            str   = "AWD"    # "AWD"  "RWD"  "FWD", not used right now, always 4wd
    gear_ratio:            float = 15    # final drive ratio
    regen:                 bool = True      # True if regen braking is enabled
    max_regen_torque:      float = 5       # Nm  at motor
    max_regen_power:       float = 14000     #Max power delivered back to the battery

    pack_voltage:      float = 510  #V average pack voltage
    
    def __post_init__(self):
        # motor efficiency map, 2D
        m = loadmat(DATA/ self.motor_type / (self.motor_type + "_Efficiency.mat"))
        self.eta_torque = np.squeeze(m["Torque"]).astype(float)      
        self.eta_rpm    = np.squeeze(m["RPM"]).astype(float)         

        self.eta_map = np.asarray(m["Efficiency"], dtype=float)   # rows=torque, cols=rpm
        if self.eta_map.max() > 1.5:                              # percent map -> fraction
            self.eta_map /= 100
        self.eta_map = np.clip(self.eta_map, 0.05, 1.0)           # no zeros: pack_power divides by eta

        # motor torque envelope, 1D
        c = loadmat(DATA/ self.motor_type / (self.motor_type + "_MotorCurve.mat"))
        self.curve_rpm    = np.squeeze(c["RPM"]).astype(float)       # (11,)   [rpm]
        self.curve_torque = np.squeeze(c["Tmotor"]).astype(float)    # (11,)   [Nm] 

    # ── Brakes ────────────────────────────────────────────────
    max_decel:          float = 18.0    # m/s²  peak braking deceleration
    brake_bias_front:   float = 0.5    # fraction of brake force on front axle

    
    # ── Constants ─────────────────────────────────────────────
    air_density:        float = 1.225   # kg/m³
    g:                  float = 9.81    # m/s²
