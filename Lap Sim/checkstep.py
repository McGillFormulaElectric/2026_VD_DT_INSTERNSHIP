import numpy as np, physics as ph, Limits as lim
from carProperties import MFE27
from Tire import Tire
from Acceleration import AccelSolver

car, tire = MFE27(), Tire("Data/Tire/MF61_Coefficients.csv")
res = AccelSolver().simulate(tire, car)

bad = np.where((res["limit"] == "power") & (res["PackPower"] < 0.9*car.power_cap))[0]
print("dip steps:", len(bad))
i = bad[np.argmin(res["PackPower"][bad])]           # the worst one
v, FzF, FzR = res["v"][i], res["FzF"][i], res["FzR"][i]
print(f"worst: s={res['s'][i]:.2f} m  v={v:.2f}  FzF={FzF:.0f}  FzR={FzR:.0f}  P={res['PackPower'][i]:.0f}")

rpm = ph.motor_rpm(car, v)
Tm  = min(np.interp(rpm, car.curve_rpm, car.curve_torque), car.torque_cap) * car.torque_scale
Fm  = ph.wheel_force(car, Tm)
gF, gR = tire.peak_drive(FzF), tire.peak_drive(FzR)
print(f"rpm={rpm:.0f}  T_max={Tm:.2f}  Fx_motor={Fm:.0f}  gripF={gF:.0f}  gripR={gR:.0f}")
print("curve_rpm:", car.curve_rpm)
print("curve_torque:", car.curve_torque)

for lam in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
    Fx = [min(lam*Fm, gF)]*2 + [min(lam*Fm, gR)]*2
    T  = [ph.motor_torque(car, f) for f in Fx]
    e  = [ph.motor_eff(car, rpm, t) for t in T]
    print(f"lam={lam:.1f}  Fx={np.round(Fx)}  T={np.round(T,2)}  eta={np.round(e,3)}  "
          f"P={ph.pack_power(car, T, e, rpm):.0f}")