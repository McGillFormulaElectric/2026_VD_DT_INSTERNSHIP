import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Path to the Lap Sim folder, so we can import MotecData.py
LAPSIM_ROOT = Path(__file__).resolve().parents[1]   # folder that holds MotecData.py
sys.path.insert(0, str(LAPSIM_ROOT))

from MotecData import MotecData
from TrackMap import loadTrack
from CarProperties import MFE26
from Tire import Tire
from SolverFunctions import corner_speed_ceiling, lap_profile, energy_and_time   # same steps endurance uses for lap 2

APEX_WINDOW = 5.0   # m, search window around each apex for the minimum speed

TIRE_DIR = LAPSIM_ROOT / "1. Data" / "Tire"

car = MFE26()
tire = Tire(TIRE_DIR / "MF61_Parameters.mat", TIRE_DIR / "FZ_Reference.mat")   # correlated defaults in Tire.py

# ── measured lap ──────────────────────────────────────────────
data = MotecData(LAPSIM_ROOT / "1. Data" / "Motec" / "autocross.mat")
time = data.getTime("velX")
velX = data.getValue("velX")        # m/s, GPS north/east, use the norm
velY = data.getValue("velY")
power = data.getValue("power")      # kW, pack
torque_meas = (data.getValue("TorqueActualFL") + data.getValue("TorqueActualFR")
               + data.getValue("TorqueActualRL") + data.getValue("TorqueActualRR"))   # Nm, 4 motors
dt = np.mean(np.diff(time))

V_meas = savgol_filter(np.sqrt(velX**2 + velY**2), 31, 2)
ax_meas = np.gradient(V_meas, dt) / car.g       # g, + accelerating
dist_meas = np.cumsum(V_meas) * dt
dist_meas = dist_meas - dist_meas[0]
t_meas = time - time[0]

# ── track built from this same file ───────────────────────────
track = loadTrack(LAPSIM_ROOT / "1. Data" / "Track", "autocrossID16")
print(f"track: {track.lap_length:.1f} m, {len(track.apex)} apexes, tightest R = {1 / np.max(np.abs(track.k)):.1f} m")
print("       (expect about 336 m, 25 apexes, R about 4.7 m)")
print()

torque_profile = np.interp(track.s, dist_meas, torque_meas)   # measured torque on the track grid
v_max = corner_speed_ceiling(track, car, tire)                # grip limit, computed once for both runs

def flying_lap(torque_profile=None):
    # endurance lap 2: closed loop, start speed = end speed
    v, v_fwd, v_bwd = lap_profile(track, car, tire, v_max, standing_start=False, torque_profile=torque_profile)
    t, P_pack, energy_J = energy_and_time(track, car, tire, v, v_max)
    return {"s": track.s, "v": v, "v_max": v_max, "t": t, "P_pack": P_pack,
            "flying_lap_time": t[-1] + track.ds / max(v[-1], 0.1),
            "energy_per_lap_kWh": energy_J / 3.6e6}

sim_free = flying_lap()
sim = flying_lap(torque_profile)
lap_free = sim_free["flying_lap_time"]
lap_sim = sim["flying_lap_time"]

print(f"flying lap, full torque:      {lap_free:.2f} s")
print(f"flying lap, measured torque:  {lap_sim:.2f} s")
print(f"cost of throttle + wheelspin: {lap_sim - lap_free:.2f} s")
print()

# everything below compares the measured-torque sim to the data
s_sim = np.asarray(sim["s"])
v_sim = np.asarray(sim["v"])
t_sim = np.asarray(sim["t"])
v_limit = np.asarray(sim["v_max"])          # corner speed limit from grip
P_sim = np.asarray(sim["P_pack"]) / 1000    # kW, solver returns W
ax_sim = v_sim * np.gradient(v_sim, s_sim) / car.g   # g, a = v dv/ds
v_free = np.asarray(sim_free["v"])

# put the sim on the measured distance so both can be compared point by point
v_sim_m = np.interp(dist_meas, s_sim, v_sim)
t_sim_m = np.interp(dist_meas, s_sim, t_sim - t_sim[0])
delta = t_sim_m - t_meas                    # s, negative = sim ahead

print(f"lap time:      sim {lap_sim:.2f} s   measured {t_meas[-1]:.2f} s   "
      f"gap {lap_sim - t_meas[-1]:+.2f} s ({(lap_sim / t_meas[-1] - 1) * 100:+.1f} %)")
print(f"avg speed:     sim {np.mean(v_sim_m) * 3.6:.1f} km/h   measured {np.mean(V_meas) * 3.6:.1f} km/h")
print()
print(f"pack power:    sim max {np.max(P_sim):.1f} kW   measured max {np.max(power):.1f} kW")
print(f"energy:        sim {sim['energy_per_lap_kWh'] * 1000:.0f} Wh   measured {np.sum(power) * dt / 3.6:.0f} Wh")
print(f"accel:         sim 99th pct {np.percentile(ax_sim, 99):.2f} g   measured {np.percentile(ax_meas, 99):.2f} g")
print(f"braking:       sim 99th pct {-np.percentile(ax_sim, 1):.2f} g   measured {-np.percentile(ax_meas, 1):.2f} g")
print()

# minimum speed around each apex, same window for both
# the slowest point is not always exactly at the curvature peak
print(f"{'apex':>5} {'s [m]':>6} {'R [m]':>6} {'test [km/h]':>12} {'sim [km/h]':>11} {'error':>7}")
for n, i in enumerate(track.apex):
    s_apex = track.s[i]
    v_test = np.min(V_meas[np.abs(dist_meas - s_apex) < APEX_WINDOW]) * 3.6
    v_apex = np.min(v_sim[np.abs(s_sim - s_apex) < APEX_WINDOW]) * 3.6
    R = 1 / abs(track.k[i])
    print(f"{n + 1:>5} {s_apex:>6.0f} {R:>6.1f} {v_test:>12.1f} {v_apex:>11.1f} {(v_apex / v_test - 1) * 100:>+6.1f}%")

# ── plots vs distance ─────────────────────────────────────────
fig, axs = plt.subplots(4, 1, sharex=True, figsize=(12, 11))

axs[0].plot(dist_meas, V_meas * 3.6, label="test")
axs[0].plot(s_sim, v_sim * 3.6, label="sim, measured torque")
axs[0].plot(s_sim, v_free * 3.6, "--", alpha=0.6, label="sim, full torque")
axs[0].plot(s_sim, v_limit * 3.6, ":", color="gray", label="sim grip limit")
for i in track.apex:
    axs[0].axvline(track.s[i], color="k", alpha=0.15)
axs[0].set_ylim(0, 1.2 * max(np.max(V_meas), np.max(v_free)) * 3.6)
axs[0].set_ylabel("V [km/h]")
axs[0].legend()

axs[1].plot(dist_meas, delta)
axs[1].axhline(0, color="k", linewidth=0.8)
axs[1].set_ylabel("sim - test time [s]\n(down = sim ahead)")

axs[2].plot(dist_meas, ax_meas, label="test")
axs[2].plot(s_sim, ax_sim, label="sim, measured torque")
axs[2].axhline(0, color="k", linewidth=0.8)
axs[2].set_ylabel("ax [g]")
axs[2].legend()

axs[3].plot(dist_meas, power, label="test")
axs[3].plot(s_sim, P_sim, label="sim, measured torque")
axs[3].axhline(80, color="r", linestyle="--", alpha=0.5)
axs[3].set_ylabel("pack power [kW]")
axs[3].set_xlabel("distance [m]")
axs[3].legend()

for a in axs:
    a.grid(True)
fig.suptitle("Autocross, sim vs test")
plt.tight_layout()
plt.show()