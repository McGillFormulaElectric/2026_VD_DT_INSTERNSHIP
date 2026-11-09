"""
EnergyBreakdown.py  -  where does the pack energy go over an endurance lap?
"""
import sys
from pathlib import Path
import csv
import numpy as np
import matplotlib
try:
    import tkinter                      # same backend as the other analysis scripts, if available
    matplotlib.use("TkAgg")
except ImportError:
    pass
import matplotlib.pyplot as plt

# ── find the Lap Sim root (folder that holds SolverFunctions.py), however deep this file sits
HERE = Path(__file__).resolve().parent
LAPSIM_ROOT = next(p for p in [HERE, *HERE.parents] if (p / "SolverFunctions.py").exists())
sys.path.insert(0, str(LAPSIM_ROOT))

import Physics as ph
from TrackMap import loadTrack
from carProperties import MFE26
from Tire import Tire
from SolverFunctions import corner_speed_ceiling, lap_profile, energy_and_time
from Endurance import N_LAPS

# ── settings ──────────────────────────────────────────────
TRACK = "Michigan2026_Endurance"

def drs_baseline():
    """MFE26 with the DRS capstone closed-wing aero (CFD tab, C0 baseline)."""
    car = MFE26()
    car.CLA = 4.43
    car.CDA = 1.92
    car.aero_balance = 0.41
    return car

CARS = [("MFE26 (DRS baseline aero)", drs_baseline)]     # add more (name, factory) pairs to compare cars
OUT_CSV = HERE / "energy_breakdown.csv"
OUT_PNG = HERE / "energy_breakdown.png"

TIRE_DIR = LAPSIM_ROOT / "1. Data" / "Tire"
TRACK_DIR = LAPSIM_ROOT / "1. Data" / "Track"

# ── energy bookkeeping, mirrors energy_and_time step by step ──
CATEGORIES = [
    "Aero drag",
    "Rolling resistance",
    "Tyre cornering drag",
    "Friction brakes",
    "Motor losses (drive)",
    "Inverter losses (drive)",
    "Unmodelled losses (efficiency_scale)",
    "Regen conversion losses",
    "Kinetic energy residual",
]

def energy_breakdown(track, car, tire, v, v_max):
    """Energy (J) over one lap, split by where it is dissipated. Sum of the
    categories = net pack energy (drive draw minus regen recovered)."""
    n = len(v)
    ds = track.ds
    E = {k: 0.0 for k in CATEGORIES}
    E_drive_pack = 0.0      # J, drawn from the pack while driving
    E_regen_pack = 0.0      # J, put back in the pack by regen (positive number)
    E_shaft_drive = 0.0     # J, mechanical work the motors deliver to the wheels
    E_regen_shaft = 0.0     # J, mechanical work the motors absorb under regen
    dKE = 0.0               # J, kinetic energy change over the lap (should be ~0, flying lap)

    for i in range(n - 1):
        vi = max(v[i], 0.1)
        DF = ph.downforce(car, vi)
        ax = (v[i + 1]**2 - v[i]**2) / (2 * ds)
        ay = vi**2 * abs(track.k[i])
        util = (vi / max(v_max[i], 1e-9))**2
        F_drag = ph.drag(car, vi)
        F_rr   = ph.rolling_resistance(car, vi, car.mass_total*car.g + DF)
        F_cd   = ph.cornering_drag(car, ay, util, tire.slip_peak)
        F_prop = car.mass_effective * ax + F_drag + F_rr + F_cd    # N

        # resistances are paid every step, by the pack when driving, by KE when coasting/braking
        E["Aero drag"]           += F_drag * ds
        E["Rolling resistance"]  += F_rr * ds
        E["Tyre cornering drag"] += F_cd * ds
        dKE += car.mass_effective * ax * ds

        rpm = ph.motor_rpm(car, vi)
        omega = rpm * 2 * np.pi / 60
        dt = ds / vi

        if F_prop > 0:                                   # driving
            T = ph.motor_torque(car, F_prop / 4)
            eta_map = ph.motor_eff(car, rpm, T)
            eta = eta_map * car.efficiency_scale
            P_shaft = 4 * T * omega                      # W, = F_prop * v
            P_motor_in = P_shaft / eta_map               # W, map losses only
            P_motor_in_corr = P_shaft / eta              # W, with the correlation correction
            P_pack = P_motor_in_corr / car.inverter_efficiency
            E_shaft_drive += P_shaft * dt
            E["Motor losses (drive)"]   += (P_motor_in - P_shaft) * dt
            E["Unmodelled losses (efficiency_scale)"] += (P_motor_in_corr - P_motor_in) * dt
            E["Inverter losses (drive)"] += (P_pack - P_motor_in_corr) * dt
            E_drive_pack += P_pack * dt

        elif F_prop < 0:                                 # braking
            F_brake = -F_prop                            # N, needed beyond drag + rolling + cornering
            if car.regen:
                T_F = min(ph.motor_torque(car, F_brake * car.brake_bias_front / 2), car.max_regen_torque)
                T_R = min(ph.motor_torque(car, F_brake * (1 - car.brake_bias_front) / 2), car.max_regen_torque)
                Ts = [T_F, T_F, T_R, T_R]
                P_shaft_regen = sum(T * omega for T in Ts)                                  # W absorbed by motors
                P_pack_regen = sum(T * omega * ph.motor_eff(car, rpm, T) * car.efficiency_scale
                                   * car.inverter_efficiency for T in Ts)                   # W back in the pack
                E_regen_shaft += P_shaft_regen * dt
                E_regen_pack += P_pack_regen * dt
                E["Regen conversion losses"] += (P_shaft_regen - P_pack_regen) * dt
                E["Friction brakes"] += F_brake * ds - P_shaft_regen * dt
            else:
                E["Friction brakes"] += F_brake * ds

    E["Kinetic energy residual"] = dKE
    info = {"E_drive_pack_J": E_drive_pack, "E_regen_pack_J": E_regen_pack,
            "E_shaft_drive_J": E_shaft_drive, "E_regen_shaft_J": E_regen_shaft,
            "E_net_pack_J": E_drive_pack - E_regen_pack}
    return E, info


# ── run ───────────────────────────────────────────────────
tire = Tire(TIRE_DIR / "MF61_Parameters.mat", TIRE_DIR / "FZ_Reference.mat")
track = loadTrack(TRACK_DIR, TRACK)

results = {}
for name, Car in CARS:
    car = Car()
    v_max = corner_speed_ceiling(track, car, tire)
    v, _, _ = lap_profile(track, car, tire, v_max, standing_start=False)
    t, P_pack, E_ref = energy_and_time(track, car, tire, v, v_max)     # the sim's own number, as a check
    E, info = energy_breakdown(track, car, tire, v, v_max)
    lap_time = t[-1] + track.ds / max(v[-1], 0.1)
    results[name] = (E, info, E_ref, lap_time)

    total = sum(E.values())
    print(f"\n{name}  -  {TRACK}, flying lap {lap_time:.2f} s, x{N_LAPS} laps")
    print(f"  {'source':<40}{'kWh/lap':>10}{'kWh/event':>12}{'share':>8}")
    for k in CATEGORIES:
        print(f"  {k:<40}{E[k]/3.6e6:>10.3f}{E[k]*N_LAPS/3.6e6:>12.2f}{100*E[k]/total:>7.1f}%")
    print(f"  {'NET PACK ENERGY (sum)':<40}{total/3.6e6:>10.3f}{total*N_LAPS/3.6e6:>12.2f}")
    print(f"  {'check: energy_and_time()':<40}{E_ref/3.6e6:>10.3f}{E_ref*N_LAPS/3.6e6:>12.2f}")
    print(f"  pack draw while driving {info['E_drive_pack_J']/3.6e6:.3f} kWh/lap, "
          f"regen recovered {info['E_regen_pack_J']/3.6e6:.3f} kWh/lap")

# ── csv (Excel friendly: one row per source, one block of columns per car) ──
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    header = ["Energy sink"]
    for name in results:
        header += [f"{name} kWh/lap", f"{name} kWh/event ({N_LAPS} laps)", f"{name} share %"]
    w.writerow(header)
    for k in CATEGORIES:
        row = [k]
        for name, (E, info, E_ref, lap_time) in results.items():
            total = sum(E.values())
            row += [f"{E[k]/3.6e6:.4f}", f"{E[k]*N_LAPS/3.6e6:.3f}", f"{100*E[k]/total:.2f}"]
        w.writerow(row)
    row = ["Net pack energy (sum)"]
    for name, (E, info, E_ref, lap_time) in results.items():
        total = sum(E.values())
        row += [f"{total/3.6e6:.4f}", f"{total*N_LAPS/3.6e6:.3f}", "100"]
    w.writerow(row)
    w.writerow([])
    w.writerow(["Info"] + [x for name in results for x in (name, "", "")])
    for label, key, scale in [("Pack draw while driving kWh/lap", "E_drive_pack_J", 1/3.6e6),
                              ("Regen recovered kWh/lap", "E_regen_pack_J", 1/3.6e6),
                              ("Shaft work delivered kWh/lap", "E_shaft_drive_J", 1/3.6e6)]:
        w.writerow([label] + [x for name, (E, info, _, _) in results.items()
                              for x in (f"{info[key]*scale:.4f}", "", "")])
    w.writerow(["Flying lap time s"] + [x for name, (_, _, _, lt) in results.items() for x in (f"{lt:.3f}", "", "")])
    w.writerow(["Track"] + [x for name in results for x in (TRACK, "", "")])
print(f"\ncsv  -> {OUT_CSV}")

# ── bar chart ─────────────────────────────────────────────
plot_cats = [c for c in CATEGORIES if c != "Kinetic energy residual"]
x = np.arange(len(plot_cats))
width = 0.8 / len(results)
fig, ax = plt.subplots(figsize=(13, 6))
for j, (name, (E, info, E_ref, lap_time)) in enumerate(results.items()):
    vals = [E[c] * N_LAPS / 3.6e6 for c in plot_cats]
    bars = ax.bar(x + (j - (len(results) - 1) / 2) * width, vals, width,
                  label=f"{name}  (net {sum(E.values())*N_LAPS/3.6e6:.2f} kWh)")
    ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
ax.set_xticks(x)
ax.set_xticklabels(plot_cats, rotation=25, ha="right")
ax.set_ylabel(f"energy [kWh] over endurance ({N_LAPS} laps)")
ax.set_title(f"Where the pack energy goes  -  {TRACK}")
ax.grid(True, axis="y", alpha=0.4)
ax.legend()
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f"plot -> {OUT_PNG}")
plt.show()