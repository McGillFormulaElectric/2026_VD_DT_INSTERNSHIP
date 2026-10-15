# Summary: Run the acceleration event on its own and look at every trace it
#          produces. Fill in the USER INPUTS block, then:  python RunAccel.py

import numpy as np
import matplotlib.pyplot as plt

from carProperties import MFE27
from Tire import Tire
from Acceleration import AccelSolver


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                              USER INPUTS                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# --- Car -------------------------------------------------------------------
# Anything not listed here keeps its default from carProperties.MFE27.
# Add / remove lines freely, e.g.  "power_cap": 70_000,  "CL": 3.5,  "mass_driver": 65,
CAR = {
    "motor_type":   "AMK",    # "Fisher" or "AMK"
    "gear_ratio":   13.39,        # final drive ratio
    "torque_split": 0.50,        # fraction of torque commanded to the front (0 = RWD, 1 = FWD)
    "power_cap":    80_000,      # W   pack power limit (rules: 80 kW)
}

# --- Tire ------------------------------------------------------------------
TIRE_FILE = "Data/Tire/MF61_Coefficients.csv"
TIRE = {
    "muxScale": 0.4,             # longitudinal grip correlation factor
    "muyScale": 0.9,             # lateral grip correlation factor (unused in accel)
}

# --- Event -----------------------------------------------------------------
TRACK_LENGTH = 75.0              # m   accel course
RUN_UP       = 0               # m   distance the car starts behind the start line

# --- Output ----------------------------------------------------------------
SHOW_PLOTS = True                # False -> summary only
SAVE_FIG   = None                # e.g. "accel.png" to save the figure, None to skip

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     nothing below needs editing                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝


# ── Build ─────────────────────────────────────────────────────────────────────
car  = MFE27(**CAR)
tire = Tire(TIRE_FILE, **TIRE)

# ── Run ───────────────────────────────────────────────────────────────────────
res = AccelSolver(track_length=TRACK_LENGTH, run_up=RUN_UP).simulate(tire, car)

s   = res["s"]
g   = car.g
lim = res["limit"]

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Acceleration event ===")
print(f"car: {CAR}")
print(f"time (from start line)  {res['lap_time']:7.3f} s    -> {res['comp_point']:.1f} pts")
print(f"trap speed              {res['v_max']*3.6:7.1f} km/h  ({res['v_max']:.1f} m/s)")
print(f"peak Ax                 {res['peak_Ax']/g:7.2f} g")
print(f"energy                  {res['energy_kWh']*1000:7.1f} Wh")
print(f"peak pack power         {res['PackPower'].max()/1000:7.1f} kW   (cap {car.power_cap/1000:.0f} kW)")
print(f"peak pack current       {res['I_pack'].max():7.0f} A    (at {car.pack_voltage:.0f} V)")
print(f"peak motor torque F/R   {res['TmotorF'].max():5.1f} / {res['TmotorR'].max():5.1f} Nm")
print(f"max rpm                 {res['rpm'].max():7.0f}     rev limited: {res['rev_limited']}")

print("\nactive ceiling along the run:")
for name in ("grip", "torque", "power", "rev"):
    m = lim == name
    if m.any():
        print(f"  {name:6s} {s[m].min():6.1f} m -> {s[m].max():6.1f} m   "
              f"({res['dt'][m].sum():.2f} s)")

# ── Plots ─────────────────────────────────────────────────────────────────────
if SHOW_PLOTS or SAVE_FIG:
    colors = {"grip": "tab:red", "torque": "tab:blue", "power": "tab:orange", "rev": "tab:gray"}

    fig, ax = plt.subplots(3, 2, sharex=True, figsize=(12, 8))
    ax = ax.ravel()

    for name, c in colors.items():                       # speed, coloured by active limit
        m = lim == name
        if m.any():
            ax[0].scatter(s[m], res["v"][m]*3.6, s=4, c=c, label=name)
    ax[0].set_ylabel("v [km/h]"); ax[0].legend(title="limit", fontsize=8)

    ax[1].plot(s, res["Ax"]/g)
    ax[1].set_ylabel("Ax [g]")

    ax[2].plot(s, res["PackPower"]/1000, label="pack power [kW]")
    ax[2].plot(s, res["I_pack"]/10, "--", label="pack current [A/10]")
    ax[2].axhline(car.power_cap/1000, color="k", lw=.5, ls=":")
    ax[2].set_ylabel("power / current"); ax[2].legend(fontsize=8)

    ax[3].plot(s, res["TmotorF"], label="front")
    ax[3].plot(s, res["TmotorR"], label="rear")
    ax[3].set_ylabel("motor torque [Nm]"); ax[3].legend(fontsize=8)

    ax[4].plot(s, res["FzF"], label="front (per wheel)")
    ax[4].plot(s, res["FzR"], label="rear (per wheel)")
    ax[4].plot(s, res["Downforce"]/4, ":", label="downforce / 4")
    ax[4].set_ylabel("Fz [N]"); ax[4].legend(fontsize=8)

    ax[5].plot(s, res["Fx_tractive"], label="tractive")
    ax[5].plot(s, res["Drag"], label="drag")
    ax[5].plot(s, res["Frolling"], label="rolling")
    ax[5].plot(s, res["Fx_net"], "k--", label="net")
    ax[5].set_ylabel("Fx [N]"); ax[5].legend(fontsize=8)

    for a in ax:
        a.grid(alpha=.3)
        a.axvline(0, color="k", lw=.5)                   # start line
    ax[4].set_xlabel("s [m]"); ax[5].set_xlabel("s [m]")
    fig.suptitle(f"Acceleration  {res['lap_time']:.3f} s   {res['comp_point']:.1f} pts   "
                 f"gear {car.gear_ratio}  split {car.torque_split}  {car.motor_type}")
    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig(SAVE_FIG, dpi=130)
    if SHOW_PLOTS:
        plt.show()