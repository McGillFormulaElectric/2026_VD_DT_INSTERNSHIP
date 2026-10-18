import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Path to the Lap Sim folder
LAPSIM_ROOT = Path(__file__).resolve().parents[3]   # folder that holds CarProperties.py, change the number if this file moves
sys.path.insert(0, str(LAPSIM_ROOT))

from TrackMap import loadTrack
from CarProperties import MFE26, MFE27
from Tire import Tire
from Acceleration import AccelSolver
from SkidPad import SkidPadSolver
from Autocross import solve
from Endurance import solve as endurance_solve
from SolverFunctions import corner_speed_ceiling, lap_profile, energy_and_time

TIRE_DIR = LAPSIM_ROOT / "1. Data" / "Tire"
TRACK_DIR = LAPSIM_ROOT / "1. Data" / "Track"

tire = Tire(TIRE_DIR / "MF61_Parameters.mat", TIRE_DIR / "FZ_Reference.mat")   # correlated defaults in Tire.py
track_autocross = loadTrack(TRACK_DIR, "Michigan2026_Autocross")
track_endurance = loadTrack(TRACK_DIR, "Michigan2026_Endurance")

# car name: (car, plot colour)
CARS = {
    "MFE26 AMK":    (MFE26(), "tab:blue"),
    "MFE27 Fisher": (MFE27(), "tab:orange"),
}

# top speed: v = rpm_cap / gear_ratio * 2 pi / 60 * tire_radius
MATCH_TOP_SPEED = True   # True = Fisher rev limit lowered so it has the same top speed as the AMK car
amk, fisher = CARS["MFE26 AMK"][0], CARS["MFE27 Fisher"][0]
v_top = lambda car: car.rpm_cap / car.gear_ratio * 2 * np.pi / 60 * car.tire_radius   # m/s
if MATCH_TOP_SPEED:
    fisher.rpm_cap = amk.rpm_cap * (fisher.gear_ratio / amk.gear_ratio) * (amk.tire_radius / fisher.tire_radius)   # rpm
print(f"top speed: AMK {v_top(amk) * 3.6:.1f} km/h, Fisher {v_top(fisher) * 3.6:.1f} km/h (rpm_cap {fisher.rpm_cap:.0f})")

# event name: (function that takes a car and returns the result dict, key for the event energy [kWh])
EVENTS = {
    "Accel":     (lambda car: AccelSolver().simulate(tire, car),               "energy_kWh"),
    "Skidpad":   (lambda car: SkidPadSolver().simulate(tire, car),             "energy_kWh"),
    "Autocross": (lambda car: solve(tire, car, track_autocross),               "energy_kWh"),
    "Endurance": (lambda car: endurance_solve(tire, car, track_endurance),     "total_energy_kWh"),
}

# run every event for every car
results = {}   # (car name, event) -> result dict
for car_name, (car, _) in CARS.items():
    for ev, (run, _) in EVENTS.items():
        results[(car_name, ev)] = run(car)
        print(f"\r{car_name}: {ev} done   ", end="")
print()

# table
names = list(CARS)
print(f"\n{'event':<10}" + "".join(f"{n:>16}" for n in names) + f"{'difference':>15}")
for ev, (_, key) in EVENTS.items():
    E = [results[(n, ev)][key] for n in names]   # kWh
    print(f"{ev:<10}" + "".join(f"{x:>12.3f} kWh" for x in E) + f"{E[1] - E[0]:>+11.3f} kWh")

# plots: energy used vs distance, both cars overlaid, event energy in the legend
fig, axs = plt.subplots(2, 2, figsize=(14, 9))
for ax, (ev, (_, key)) in zip(axs.flat, EVENTS.items()):
    energies = []
    trace = all(k in results[(n, ev)] for n in CARS for k in ("s", "t", "P_pack"))   # power trace returned by the solver
    for car_name, (_, colour) in CARS.items():
        r = results[(car_name, ev)]
        E = r[key]                                                   # kWh, whole event
        energies.append(E)
        if trace:
            t = np.asarray(r["t"])                                   # s
            P = np.asarray(r["P_pack"])                              # W
            E_cum = np.concatenate([[0.0], np.cumsum(0.5 * (P[1:] + P[:-1]) * np.diff(t))]) / 3.6e6   # kWh, used so far
            note = "" if abs(E_cum[-1] - E) < 0.05 * max(E, 1e-9) else "  (trace = one lap)"
            ax.plot(np.asarray(r["s"]), E_cum, color=colour, label=f"{car_name}  {E:.3f} kWh{note}")
        else:
            ax.bar(car_name, E, color=colour, label=f"{car_name}  {E:.3f} kWh")   # no power trace, show the total
    ax.set_title(f"{ev}: {energies[1] - energies[0]:+.3f} kWh ({(energies[1] / energies[0] - 1) * 100:+.1f} %)",
                 fontweight="bold")
    ax.set_ylabel("energy used [kWh]")
    if trace:
        ax.set_xlabel("distance [m]")
    else:
        ax.set_ylim(0.9 * min(energies), 1.05 * max(energies))      # zoom so the difference is visible
    ax.grid(True)
    ax.legend()
fig.suptitle("Energy used: MFE26 (AMK) vs MFE27 (Fisher)", fontweight="bold")
plt.tight_layout()

# ── endurance track map: one flying lap, coloured by energy used per metre ──
lap = {}   # car name -> (energy per metre [Wh/m], lap energy [Wh], lap time [s])
for car_name, (car, _) in CARS.items():
    v_max = np.asarray(corner_speed_ceiling(track_endurance, car, tire))                 # m/s, grip limit
    v, _, _ = lap_profile(track_endurance, car, tire, v_max, standing_start=False)       # m/s, flying lap
    v = np.asarray(v)
    t, P_pack, energy_J = energy_and_time(track_endurance, car, tire, v, v_max)
    t, P_pack = np.asarray(t), np.asarray(P_pack)                                        # s, W
    dt = np.append(np.diff(t), track_endurance.ds / max(v[-1], 0.1))                     # s, time spent on each step
    e = P_pack * dt / track_endurance.ds / 3600                                          # Wh/m, energy used per metre
    lap[car_name] = (e, energy_J / 3600, t[-1] + dt[-1])

print("\n── endurance, one lap ──")
for car_name, (e, E, T) in lap.items():
    print(f"{car_name:<14} {E:7.1f} Wh   {T:6.2f} s")

OFFSET = 3.0   # m, gap between the two lines: AMK on the track, Fisher moved this far inside

X, Y = np.asarray(track_endurance.y), np.asarray(track_endurance.x)   # m, east on the horizontal axis, north up
dX, dY = np.gradient(X), np.gradient(Y)
L = np.hypot(dX, dY)
nX, nY = -dY / L, dX / L                                              # unit normal, left of the driving direction
inside = np.sign(np.sum(X[:-1] * np.diff(Y) - Y[:-1] * np.diff(X)))   # +1 if the lap goes anticlockwise (left = inside)
lines = {}                                                            # car name -> (X, Y) of its line
for n_car, car_name in enumerate(lap):
    d = n_car * OFFSET * inside                                       # first car on the track, second one inside
    lines[car_name] = (X + d * nX, Y + d * nY)

e_max = np.percentile(np.concatenate([e for e, _, _ in lap.values()]), 99)   # same colour scale for both cars

fig2, ax2 = plt.subplots(figsize=(10, 8))
for car_name, (e, E, T) in lap.items():
    pts = np.column_stack(lines[car_name])[:, None, :]
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)                # one segment per track step
    lc = LineCollection(segs, cmap="inferno", linewidth=3)
    lc.set_array(e[:len(segs)])
    lc.set_clim(0, e_max)
    ax2.add_collection(lc)
d = 0.5 * OFFSET * inside                                             # halfway between the two lines
ax2.plot(X + d * nX, Y + d * nY, color="w", linewidth=1, zorder=3)   # thin white gap between the cars
ax2.autoscale()
ax2.set_aspect("equal")
ax2.set_xlabel("east [m]")
ax2.set_ylabel("north [m]")
names = list(lap)
ax2.set_title(f"outside line: {names[0]}  {lap[names[0]][1]:.1f} Wh, {lap[names[0]][2]:.2f} s\n"
              f"inside line: {names[1]}  {lap[names[1]][1]:.1f} Wh, {lap[names[1]][2]:.2f} s", fontsize=10)
fig2.colorbar(lc, ax=ax2, label="energy used [Wh/m]")
fig2.suptitle("Endurance, one lap: where the energy goes", fontweight="bold")
plt.show()

# same speed trace for both cars: only the motor efficiency is different
amk_car, fis_car = CARS["MFE26 AMK"][0], CARS["MFE27 Fisher"][0]
v_max = np.asarray(corner_speed_ceiling(track_endurance, amk_car, tire))         # m/s, same grip and aero for both
v_amk = np.asarray(lap_profile(track_endurance, amk_car, tire, v_max, standing_start=False)[0])
E_amk = energy_and_time(track_endurance, amk_car, tire, v_amk, v_max)[2] / 3600  # Wh
E_fis = energy_and_time(track_endurance, fis_car, tire, v_amk, v_max)[2] / 3600  # Wh
print(f"same speed as AMK: AMK {E_amk:.1f} Wh, Fisher {E_fis:.1f} Wh, AMK uses {(E_amk / E_fis - 1) * 100:+.1f} %")