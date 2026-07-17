import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from CarProperties import MFE27
from Tire import Tire
from Acceleration import AccelSolver
from SkidPad import SkidPadSolver

DATA = Path(__file__).parent

car = MFE27()
tire = Tire(DATA / "MF61_Coefficients.csv")

# ACCELERATION
print("=" * 50)
print("ACCELERATION")
print("=" * 50)

t0 = time.perf_counter()
accel = AccelSolver().simulate(tire, car)      # capture the result AND time it
solve_time = time.perf_counter() - t0

print(f"solve time  : {solve_time*1000:.0f} ms   ({accel['n_steps']} steps)")
print(f"75 m time   : {accel['time_total']:.3f} s      (expect 3.6 to 4.2)")
print(f"trap speed  : {accel['v_final']*3.6:.1f} km/h  (expect 100 to 110)")
print(f"pack energy : {accel['EnergyPack']/1000:.2f} kJ")
print(f"peak Ax     : {accel['Ax'].max()/9.81:.2f} g")
print(f"max rpm hit : {accel['rpm'].max():.0f}  (limit {car.motor_rpm_max})")

if accel["rpm"].max() >= car.motor_rpm_max * 0.999:
    print("  NOTE: motor hit the rev limit during the run.")


# --- plots -------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(11, 7))
fig.suptitle("Acceleration event")

ax[0, 0].plot(accel["dist"], accel["v"] * 3.6)
ax[0, 0].set_xlabel("distance [m]"); ax[0, 0].set_ylabel("speed [km/h]")
ax[0, 0].set_title("Speed")

ax[0, 1].plot(accel["dist"], accel["Ax"] / 9.81)
ax[0, 1].set_xlabel("distance [m]"); ax[0, 1].set_ylabel("Ax [g]")
ax[0, 1].set_title("Longitudinal acceleration")

ax[1, 0].plot(accel["dist"], accel["PackPower"] / 1000)
ax[1, 0].axhline(car.max_power / 1000, ls="--", color="r", label="80 kW limit")
ax[1, 0].set_xlabel("distance [m]"); ax[1, 0].set_ylabel("pack power [kW]")
ax[1, 0].set_title("Pack power"); ax[1, 0].legend()

ax[1, 1].plot(accel["dist"], accel["FzF"], label="front")
ax[1, 1].plot(accel["dist"], accel["FzR"], label="rear")
ax[1, 1].set_xlabel("distance [m]"); ax[1, 1].set_ylabel("Fz per wheel [N]")
ax[1, 1].set_title("Vertical load"); ax[1, 1].legend()

fig.tight_layout()

# SKIDPAD
print()
print("=" * 50)
print("SKIDPAD")
print("=" * 50)

t0 = time.perf_counter()
skid = SkidPadSolver().simulate(tire, car)
solve_time = time.perf_counter() - t0

print(f"solve time  : {solve_time*1000:.0f} ms")
print(f"lap time    : {skid['time']:.3f} s      (expect 4.8 to 5.3)")
print(f"speed       : {skid['v']*3.6:.1f} km/h")
print(f"lateral     : {skid['AyG']:.2f} g       (expect around 1.4)")
print(f"limited by  : {skid['limit']}           (expect grip)")
print(f"loads [N]   : FL {skid['FzFL']:.0f}  FR {skid['FzFR']:.0f}  "
      f"RL {skid['FzRL']:.0f}  RR {skid['FzRR']:.0f}")

if min(skid["FzFL"], skid["FzFR"], skid["FzRL"], skid["FzRR"]) <= 1.0:
    print("  NOTE: an inside wheel has lifted (load ~ 0).")
if skid["limit"] != "grip":
    print(f"  WARNING: skidpad should be grip limited, got '{skid['limit']}'.")


fig2, ax2 = plt.subplots(figsize=(5, 4))
corners = ["FL", "FR", "RL", "RR"]
loads = [skid["FzFL"], skid["FzFR"], skid["FzRL"], skid["FzRR"]]
ax2.bar(corners, loads)
ax2.set_ylabel("Fz [N]")
ax2.set_title(f"Skidpad corner loads   ({skid['time']:.2f} s, {skid['AyG']:.2f} g)")
fig2.tight_layout()

plt.show()