from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from TrackMap import loadTrack
from CarProperties import MFE27
from Tire import Tire
from Acceleration import AccelSolver
from Endurance import solve as endurance_solve
from SolverFunctions import corner_speed_ceiling
import Physics as ph

RATIOS = np.arange(5, 20.01, 0.1)         # gear ratios to sweep
TRACK = "Endurance_Michigan_2024"         # endurance track
BATTERY_KWH = None                        # usable pack energy [kWh], set it to drop ratios that would not finish endurance

RUN_SENSITIVITY = True                    # robust choice: rerun with uncertain inputs changed (slow, about 2x the sweep)
SENS_RATIOS = np.arange(8, 15.01, 0.25)   # coarser sweep for the sensitivity, covers AMK and Fisher

# optimization weight sets: how much 1 % of each metric is worth (accel, endurance, energy)
WEIGHTS = {
    "lap time only":  (1.0, 1.0, 0.0),
    "balanced":       (1.0, 1.0, 0.5),
    "energy matters": (1.0, 1.0, 1.0),
}

LAPSIM_ROOT = Path(__file__).resolve().parents[0]
TIRE_DIR = LAPSIM_ROOT / "1. Data" / "Tire"
TRACK_DIR = LAPSIM_ROOT / "1. Data" / "Track"

car = MFE27()
tire = Tire(TIRE_DIR / "MF61_Parameters.mat", TIRE_DIR / "FZ_Reference.mat")   # correlated defaults in Tire.py
track = loadTrack(TRACK_DIR, TRACK)

# corner speed limits come from grip only, not the gear ratio, so compute them once
v_max = corner_speed_ceiling(track, car, tire)

baseline = car.gear_ratio
print(f"motor {car.motor_type}, torque_cap {car.torque_cap} Nm, rpm_cap {car.rpm_cap}, current ratio {baseline:.2f}")
print()

# ── main sweep ────────────────────────────────────────────────
accel_time = []
endurance_time = []
energy = []
efficiency_score = []
points = []
accel_trap = []         # km/h at the end of accel
accel_on_rev = []       # True if accel hit the rev limiter
v_rev_list = []         # km/h top speed at the rev limiter
v_top = []              # km/h fastest point of the endurance lap
on_rev = []             # % of the endurance lap on the rev limiter
avg_power = []          # kW average pack power over endurance
lap_time = []           # s endurance flying lap
energy_lap = []         # kWh per flying lap

for g in RATIOS:
    car.gear_ratio = g
    accel = AccelSolver().simulate(tire, car)
    en = endurance_solve(track, car, tire, v_max=v_max)

    accel_time.append(accel["lap_time"])
    endurance_time.append(en["total_time"])
    energy.append(en["total_energy_kWh"])
    efficiency_score.append(en["efficiency_score"])
    total = accel["comp_point"] + en["endurance_score"] + en["efficiency_score"]   # accel + endurance + efficiency
    points.append(total)

    v_rev = ph.vehicle_speed(car, car.rpm_cap)
    v_lap = np.asarray(en["v"])
    accel_trap.append(accel["v_max"] * 3.6)
    accel_on_rev.append(accel["rev_limited"])
    v_rev_list.append(v_rev * 3.6)
    v_top.append(v_lap.max() * 3.6)
    on_rev.append(np.mean(v_lap >= 0.999 * v_rev) * 100)
    avg_power.append(en["avg_power_kW"])
    lap_time.append(en["flying_lap_time"])
    energy_lap.append(en["energy_per_lap_kWh"])

    print(f"gear {g:5.2f}:  accel {accel['lap_time']:.3f} s   endurance {en['total_time']:.0f} s   "
          f"energy {en['total_energy_kWh']:.2f} kWh   efficiency {en['efficiency_score']:.1f} pts   "
          f"accel + endurance + efficiency {total:.1f} pts")

car.gear_ratio = baseline   # put the car back the way it was

acc = np.asarray(accel_time)
end = np.asarray(endurance_time)
en_kwh = np.asarray(energy)
i_now = int(np.argmin(np.abs(RATIOS - baseline)))

# ── best per event ────────────────────────────────────────────
print()
print("=" * 100)
print("BEST PER EVENT")
print("=" * 100)
for name, y in [("accel", acc), ("endurance", end)]:
    print(f"{name:10s} best {RATIOS[np.argmin(y)]:5.1f}: {y.min():.3f} s   at {RATIOS[i_now]:.1f}: {y[i_now]:.3f} s   "
          f"gain {y[i_now] - y.min():.3f} s ({(y[i_now] / y.min() - 1) * 100:.2f} %)")
for name, y in [("efficiency", np.asarray(efficiency_score)), ("points", np.asarray(points))]:
    print(f"{name:10s} best {RATIOS[np.argmax(y)]:5.1f}: {y.max():.1f} pts   at {RATIOS[i_now]:.1f}: {y[i_now]:.1f} pts   "
          f"gain {y.max() - y[i_now]:.1f} pts")

# ── optimization without competition points ──────────────────
# each metric becomes "% worse than the best ratio", then weighted and summed
acc_pct = (acc / acc.min() - 1) * 100        # % slower than the best accel
end_pct = (end / end.min() - 1) * 100        # % slower than the best endurance
en_pct = (en_kwh / en_kwh.min() - 1) * 100   # % more energy than the lowest

feasible = np.ones(len(RATIOS), dtype=bool) if BATTERY_KWH is None else en_kwh <= BATTERY_KWH

costs = {}
for name, (w_acc, w_end, w_en) in WEIGHTS.items():
    c = w_acc * acc_pct + w_end * end_pct + w_en * en_pct
    c[~feasible] = np.inf
    costs[name] = c

print()
print("=" * 100)
print("OPTIMIZATION, % worse than the best ratio for each metric (every 0.5 shown)")
print("=" * 100)
header = f"{'ratio':>6} {'accel [s]':>10} {'endur [s]':>10} {'energy':>8} | {'accel %':>8} {'endur %':>8} {'energy %':>9}"
for name in WEIGHTS:
    header += f" | {name[:14]:>14}"
print(header)
for i, g in enumerate(RATIOS):
    if not np.isclose(g * 2, round(g * 2)):       # print every 0.5 only, the sweep is finer
        continue
    line = (f"{g:>6.1f} {acc[i]:>10.3f} {end[i]:>10.0f} {en_kwh[i]:>8.2f} | "
            f"{acc_pct[i]:>8.2f} {end_pct[i]:>8.2f} {en_pct[i]:>9.2f}")
    for name in WEIGHTS:
        line += f" | {costs[name][i]:>14.2f}"
    if not feasible[i]:
        line += "   over battery"
    print(line)

print()
for name, c in costs.items():
    i = int(np.argmin(c))
    near = RATIOS[c <= c[i] + 0.5]            # ratios within 0.5 % of the best cost
    print(f"best for '{name}': {RATIOS[i]:.1f}   (within 0.5 %: {near.min():.1f} to {near.max():.1f})")

# pareto front: ratios where you cannot get a faster endurance without using more energy
pareto = []
for i in range(len(RATIOS)):
    dominated = np.any((end <= end[i]) & (en_kwh <= en_kwh[i]) & ((end < end[i]) | (en_kwh < en_kwh[i])))
    if not dominated:
        pareto.append(RATIOS[i])
pareto = np.asarray(pareto)
print(f"pareto ratios (endurance time vs energy): {pareto.min():.1f} to {pareto.max():.1f}")

# ── safe window: lap time within 0.5 % of the best, and accel not on the limiter ──
c_lap = costs["lap time only"]
i_lap = int(np.argmin(c_lap))
window = (c_lap <= c_lap[i_lap] + 0.5) & ~np.asarray(accel_on_rev)
if not window.any():
    print("note: every near-best ratio hits the rev limiter in accel, window uses lap time only")
    window = c_lap <= c_lap[i_lap] + 0.5
w_low, w_high = RATIOS[window].min(), RATIOS[window].max()
w_center = np.sqrt(w_low * w_high)            # geometric middle, same % margin on both sides

print()
print("=" * 100)
print("SAFE WINDOW")
print("=" * 100)
print(f"lap time within 0.5 % of best and accel not on the rev limiter: {w_low:.1f} to {w_high:.1f}")
print(f"center of the window: {w_center:.2f}   (margin {(w_center / w_low - 1) * 100:.0f} % on each side)")

# ── detail on the candidate ratios ───────────────────────────
candidates = sorted(set(np.round([w_low, w_center, RATIOS[i_lap], w_high, baseline], 2)))
print()
print("=" * 100)
print("CANDIDATES, compared to the current ratio")
print("=" * 100)
print(f"{'ratio':>6} {'v_rev':>7} {'trap':>6} {'margin':>7} {'accel':>7} {'d accel':>8} | {'lap':>7} {'endur':>7} {'d endur':>8} "
      f"{'v top':>6} {'on rev':>7} | {'kWh':>6} {'kWh/lap':>8} {'avg kW':>7} | {'d pts':>6}")
print(f"{'':>6} {'[km/h]':>7} {'[km/h]':>6} {'[km/h]':>7} {'[s]':>7} {'[s]':>8} | {'[s]':>7} {'[s]':>7} {'[s]':>8} "
      f"{'[km/h]':>6} {'[%lap]':>7} | {'':>6} {'':>8} {'':>7} | {'':>6}")
for c in candidates:
    i = int(np.argmin(np.abs(RATIOS - c)))
    margin = v_rev_list[i] - accel_trap[i]
    flag = "  accel on limiter" if accel_on_rev[i] else ""
    print(f"{RATIOS[i]:>6.2f} {v_rev_list[i]:>7.0f} {accel_trap[i]:>6.0f} {margin:>7.0f} {acc[i]:>7.3f} "
          f"{acc[i] - acc[i_now]:>+8.3f} | {lap_time[i]:>7.2f} {end[i]:>7.0f} {end[i] - end[i_now]:>+8.1f} "
          f"{v_top[i]:>6.0f} {on_rev[i]:>7.1f} | {en_kwh[i]:>6.2f} {energy_lap[i]:>8.3f} {avg_power[i]:>7.1f} | "
          f"{points[i] - points[i_now]:>+6.1f}{flag}")
print()
print("margin = rev-limit speed minus accel trap speed, keep it positive with some room")
print("on rev = share of the endurance lap sitting on the rev limiter")
print("d      = difference vs the current ratio (negative time = faster)")

# ── robust choice: smallest worst-case loss over uncertain inputs ──
robust = None
if RUN_SENSITIVITY:
    print()
    print("=" * 100)
    print("SENSITIVITY, best ratio (lap time only) when one uncertain input changes")
    print("=" * 100)
    cases = [
        ("baseline",              None,               None),
        ("rpm_cap 20000",         "rpm_cap",          20_000),
        ("efficiency_scale 1.00", "efficiency_scale", 1.00),
        ("mass +10 kg",           "mass_no_driver",   car.mass_no_driver + 10),
        ("J_motor x2",            "J_motor",          car.J_motor * 2),
        ("power_cap 70 kW",       "power_cap",        70_000),
        # grip on the tire, +/- 10 %: lateral moves the corner speeds, longitudinal moves launch and braking
        (f"muyScale {tire.muyScale * 0.9:.3f} (-10 %)", "muyScale", tire.muyScale * 0.9),
        (f"muyScale {tire.muyScale * 1.1:.3f} (+10 %)", "muyScale", tire.muyScale * 1.1),
        (f"muxScale {tire.muxScale * 0.9:.3f} (-10 %)", "muxScale", tire.muxScale * 0.9),
        (f"muxScale {tire.muxScale * 1.1:.3f} (+10 %)", "muxScale", tire.muxScale * 1.1),
    ]

    loss = []   # one row per case: % lap time lost at each ratio vs that case's own best
    for name, attr, value in cases:
        on_tire = attr in ("muyScale", "muxScale")
        target = tire if on_tire else car
        if attr is not None:
            old = getattr(target, attr)
            setattr(target, attr, value)             # setting a tire scale rebuilds its grip table
        v_max_case = corner_speed_ceiling(track, car, tire) if on_tire else v_max   # grip changes the corner limits

        a, e = [], []
        for g in SENS_RATIOS:
            car.gear_ratio = g
            a.append(AccelSolver().simulate(tire, car)["lap_time"])
            e.append(endurance_solve(track, car, tire, v_max=v_max_case)["total_time"])
        a, e = np.asarray(a), np.asarray(e)
        cost = (a / a.min() - 1) * 100 + (e / e.min() - 1) * 100
        loss.append(cost)
        print(f"{name:26s} best {SENS_RATIOS[np.argmin(cost)]:5.2f}   accel {a.min():.3f} s   endurance {e.min():.0f} s")

        if attr is not None:
            setattr(target, attr, old)
    car.gear_ratio = baseline

    loss = np.asarray(loss)
    worst = loss.max(axis=0)                  # worst case at each ratio
    i_rob = int(np.argmin(worst))
    robust = SENS_RATIOS[i_rob]
    print()
    print(f"{'ratio':>6} {'worst-case loss':>16}")
    for g, w in zip(SENS_RATIOS, worst):
        mark = "  <- most robust" if g == robust else ""
        print(f"{g:>6.2f} {w:>15.2f} %{mark}")
    print()
    print(f"most robust ratio: {robust:.2f}  (never more than {worst[i_rob]:.2f} % from the best, in any case)")

# ── summary ───────────────────────────────────────────────────
print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
print(f"best lap time:        {RATIOS[i_lap]:.1f}")
print(f"best balanced:        {RATIOS[int(np.argmin(costs['balanced']))]:.1f}")
print(f"best points:          {RATIOS[int(np.argmax(points))]:.1f}")
print(f"safe window:          {w_low:.1f} to {w_high:.1f}, center {w_center:.2f}")
if robust is not None:
    print(f"most robust:          {robust:.2f}")
print(f"current:              {baseline:.2f}")
print("pick the buildable ratio (tooth counts) closest to the robust value or the window center")

# ── plots ─────────────────────────────────────────────────────
fig, axs = plt.subplots(2, 3, figsize=(16, 8))

results = [
    (axs[0, 0], accel_time, "Acceleration Time", "Acceleration (s)"),
    (axs[0, 1], endurance_time, f"Endurance Time, {TRACK}", "Endurance Time (s)"),
    (axs[0, 2], energy, f"Endurance Energy, {TRACK}", "Energy (kWh)"),
    (axs[1, 0], efficiency_score, "Efficiency Score", "Efficiency Score (-)"),
    (axs[1, 1], points, "Accel + Endurance + Efficiency", "Points (-)"),
]

for a, y, title, ylabel in results:
    a.plot(RATIOS, y, linewidth=2)
    a.ticklabel_format(axis="y", useOffset=False)   # show real values, not "1e-11 + ..."
    a.axvline(baseline, color="k", linestyle="--", alpha=0.5, label=f"current {baseline:.2f}")
    a.set_title(title, fontweight="bold")
    a.set_xlabel("Gear Ratio (-)")
    a.set_ylabel(ylabel)
    a.grid(True)
    a.legend()

# mark the fastest ratio on the time plots, and the most points on the score plots
for a, y in [(axs[0, 0], accel_time), (axs[0, 1], endurance_time)]:
    i = np.argmin(y)
    a.plot(RATIOS[i], y[i], "ro")
for a, y in [(axs[1, 0], efficiency_score), (axs[1, 1], points)]:
    i = np.argmax(y)
    a.plot(RATIOS[i], y[i], "ro")

fig.delaxes(axs[1, 2])   # only 5 plots
plt.tight_layout()
plt.show()