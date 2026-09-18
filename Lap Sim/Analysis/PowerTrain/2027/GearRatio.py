# Author: Anne-Sophie
# Summary: Sweep the final-drive gear ratio and score each event, to pick the
#          ratio that maximizes competition points.
#          Total = Skidpad + Accel + Endurance + Efficiency (Autocross hook below).
#          Endurance track is the Michigan .mat, cached to Data/Track as .npz.
import sys, pathlib, time
import numpy as np
import matplotlib.pyplot as plt

CORE = pathlib.Path(__file__).resolve().parents[3]   # 2027 -> PowerTrain -> Analysis -> Lap Sim
sys.path.insert(0, str(CORE))

from CarProperties import MFE27
from Tire import Tire
from Acceleration import AccelSolver
from SkidPad import SkidPadSolver
from Endurance import solve as endurance_solve
from TrackMap import loadTrack

DATA = CORE / "Data"

# sweep settings
GEAR_MIN, GEAR_MAX, GEAR_STEP = 9.0, 16.0, 0.5

# load car, tire, endurance track 
car  = MFE27()
tire = Tire(str(DATA / "Tire" / "MF61_Coefficients.csv"))

mat = DATA / "Track"
track_en = loadTrack(mat, "autocross")   # endurance track is the Michigan .mat, cached to Data/Track as .npz

# -- sweep ------------------------------------------------------------------
baseline = car.gear_ratio
ratios   = np.arange(GEAR_MIN, GEAR_MAX + GEAR_STEP/2, GEAR_STEP)   # 9.0 .. 16.0 step 0.5

print(f"baseline gear ratio : {baseline:.3f}")
print(f"sweeping {ratios[0]:.2f} -> {ratios[-1]:.2f} step {GEAR_STEP}  ({len(ratios)} points)\n")
print(f"{'gear':>6} {'accel':>7} {'skid':>6} {'endur':>7} {'eff':>7} {'TOTAL':>7}  "
      f"{'75m[s]':>7} {'skid[s]':>8} {'endur[s]':>9} {'trap':>7}")

rows = []
t0 = time.perf_counter()
for i, g in enumerate(ratios, 1):
    car.gear_ratio = g                          # live: motor_rpm/wheel_force read it

    accel = AccelSolver().simulate(tire, car)
    skid  = SkidPadSolver().simulate(tire, car)
    endur = endurance_solve(track_en, car, tire)

    r = {"accel_score": accel["comp_point"],
         "skid_score":  skid["comp_point"],
         "endur_score": endur["endurance_score"],
         "eff_score":   endur["efficiency_score"],
         "accel_time":  accel["lap_time"],
         "skid_time":   skid["lap_time"],
         "endur_time":  endur["total_time"],
         "trap_kmh":    accel["v_max"] * 3.6,
         "endur_kWh":   endur["total_energy_kWh"]}

    # Autocross hook: ax = autocross_solve(track_ax, car, tire); r["ax_score"] = ax["comp_point"]

    r["total"] = r["skid_score"] + r["accel_score"] + r["endur_score"] + r["eff_score"]
    rows.append(r)
    print(f"{g:6.3f} {r['accel_score']:7.1f} {r['skid_score']:6.1f} "
          f"{r['endur_score']:7.1f} {r['eff_score']:7.1f} {r['total']:7.1f}  "
          f"{r['accel_time']:7.3f} {r['skid_time']:8.3f} {r['endur_time']:9.1f} "
          f"{r['trap_kmh']:7.1f}   [{i}/{len(ratios)}]", flush=True)

car.gear_ratio = baseline                       # restore
print(f"\nswept in {time.perf_counter()-t0:.1f} s")

totals = np.array([r["total"] for r in rows])
i_opt  = int(np.argmax(totals))
g_opt  = ratios[i_opt]
i_base = int(np.argmin(abs(ratios - baseline)))
print(f"\noptimal gear ratio  : {g_opt:.3f}  ->  {totals[i_opt]:.1f} pts "
      f"(baseline ~{totals[i_base]:.1f}, gain {totals[i_opt]-totals[i_base]:+.1f})")

# -- figure -----------------------------------------------------------------
def col(k): return np.array([r[k] for r in rows])

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

a = ax[0, 0]
a.plot(ratios, totals, "o-", lw=2, color="k")
a.axvline(g_opt, color="green", ls="--", label=f"optimum {g_opt:.2f}")
a.axvline(baseline, color="gray", ls=":", label=f"baseline {baseline:.2f}")
a.scatter([g_opt], [totals.max()], s=120, color="green", zorder=5)
a.set_xlabel("gear ratio"); a.set_ylabel("competition points")
a.set_title("Total points vs gear ratio"); a.grid(alpha=0.3); a.legend()

a = ax[0, 1]
a.plot(ratios, col("accel_score"), "o-", label="Accel")
a.plot(ratios, col("skid_score"),  "s-", label="Skidpad")
a.plot(ratios, col("endur_score"), "^-", label="Endurance")
a.plot(ratios, col("eff_score"),   "d-", label="Efficiency")
a.axvline(g_opt, color="green", ls="--")
a.set_xlabel("gear ratio"); a.set_ylabel("event points")
a.set_title("Where the points come from"); a.grid(alpha=0.3); a.legend()

a = ax[1, 0]
a.plot(ratios, col("accel_time"), "o-", color="tab:blue")
a.set_xlabel("gear ratio"); a.set_ylabel("75 m time [s]", color="tab:blue")
a.tick_params(axis="y", labelcolor="tab:blue"); a.grid(alpha=0.3)
a2 = a.twinx()
a2.plot(ratios, col("trap_kmh"), "s-", color="tab:red")
a2.set_ylabel("trap speed [km/h]", color="tab:red"); a2.tick_params(axis="y", labelcolor="tab:red")
a.axvline(g_opt, color="green", ls="--")
a.set_title("Accel: shorter gear = more launch, less top end")

a = ax[1, 1]
a.plot(ratios, col("endur_time"), "o-", color="tab:blue")
a.set_ylabel("endurance time [s]", color="tab:blue"); a.tick_params(axis="y", labelcolor="tab:blue")
a2 = a.twinx()
a2.plot(ratios, col("endur_kWh"), "s-", color="tab:red")
a2.set_ylabel("energy [kWh]", color="tab:red"); a2.tick_params(axis="y", labelcolor="tab:red")
a.set_title("Endurance: time vs energy trade")
a.set_xlabel("gear ratio"); a.axvline(g_opt, color="green", ls="--"); a.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(CORE / "gear_sweep.png", dpi=120)
plt.show()