import numpy as np, physics as ph
from carProperties import MFE27
car = MFE27()
print("eta map min/max:", car.eta_map.min(), car.eta_map.max())
print("torque axis:", car.eta_torque)
for T in (0, 0.5, 3, 12, 20, 24, 27, 29):
    print(f"T={T:5.1f}  eta={ph.motor_eff(car, 12000, T)}")