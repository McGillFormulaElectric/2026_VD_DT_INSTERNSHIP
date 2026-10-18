import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Path to the Lap Sim folder
LAPSIM_ROOT = Path(__file__).resolve().parents[1]   # folder that holds MotecData.py
sys.path.insert(0, str(LAPSIM_ROOT))

from MotecData import MotecData
from TrackMap import loadTrack
from CarProperties import MFE26
from Tire import Tire
from SolverFunctions import corner_speed_ceiling, lap_profile, energy_and_time
from Scoring import Scoring

LOG = "autocross.mat"   # test lap
TRACK = "autocrossID16"      # track map built from the same file
STANDING_START = False             # True = start from 0 m/s, False = flying lap
APEX_WINDOW = 5.0                  # m, search window around each apex for the minimum speed
APEX_FLAG = 5.0                    # %, apex error beyond this gets a dot: red = sim too fast, green = sim too slow
R_X = -0.255                       # m, IMU is 255 mm behind the CG (x forward)
WHEELS = ["FL", "FR", "RL", "RR"]

TIRE_DIR = LAPSIM_ROOT / "1. Data" / "Tire"
car = MFE26()
tire = Tire(TIRE_DIR / "MF61_Parameters.mat", TIRE_DIR / "FZ_Reference.mat")   # correlated defaults in Tire.py
G = car.g

# ── test lap ──────────────────────────────────────────────────
data = MotecData(LAPSIM_ROOT / "1. Data" / "Motec" / LOG)
time = data.getTime("velX")                                           # s
velX = data.getValue("velX")                                          # m/s
velY = data.getValue("velY")                                          # m/s
accX = data.getValue("accX")                                          # m/s^2, IMU, mounted backwards
accY = data.getValue("accY")                                          # m/s^2, IMU, mounted backwards
pitch = np.radians(data.getValue("Pitch"))                            # rad, IMU frame, includes the 4.2 deg tilt
roll = np.radians(data.getValue("Roll"))                              # rad, IMU frame
yaw_rate = data.getValue("gyrZ_HR")                                   # rad/s, + left
power = data.getValue("power")                                        # kW, pack
torque = sum(data.getValue(f"TorqueActual{w}") for w in WHEELS)       # Nm, sum of 4 motors
dt = np.mean(np.diff(time))                                           # s

V_test = savgol_filter(np.sqrt(velX**2 + velY**2), 31, 2)   # m/s, GPS
s_test = np.cumsum(V_test) * dt                              # m, distance travelled
s_test = s_test - s_test[0]
t_test = time - time[0]                                      # s
r = savgol_filter(yaw_rate, 31, 2)                           # rad/s
r_dot = np.gradient(r, dt)                                   # rad/s^2

# IMU: sign flipped (mounted backwards), gravity from tilt and roll removed, moved to the CG
ax_test = savgol_filter(-(accX + G * np.sin(pitch)) + R_X * r**2, 31, 2) / G   # g, + accelerating
ay_test = savgol_filter(-(accY - G * np.sin(roll)) - R_X * r_dot, 31, 2) / G   # g, + left

# ── sim on the same track ─────────────────────────────────────
track = loadTrack(LAPSIM_ROOT / "1. Data" / "Track", TRACK)
v_max = np.asarray(corner_speed_ceiling(track, car, tire))   # m/s, grip limit at each point
torque_profile = np.interp(track.s, s_test, torque)          # Nm, measured torque on the track grid
s_sim = np.asarray(track.s)                                  # m
k_sim = np.asarray(track.k)                                  # 1/m

def run_lap(torque_profile=None):
    # returns speed [m/s], time [s], pack power [kW], lap time [s], energy [Wh], forward pass speed [m/s]
    v, v_fwd, _ = lap_profile(track, car, tire, v_max, standing_start=STANDING_START, torque_profile=torque_profile)
    t, P_pack, energy_J = energy_and_time(track, car, tire, v, v_max)
    lap_time = t[-1] + track.ds / max(v[-1], 0.1)
    return np.asarray(v), np.asarray(t), np.asarray(P_pack) / 1000, lap_time, energy_J / 3600, np.asarray(v_fwd)

v_free, t_free, P_free, lap_free, E_free, _ = run_lap()                   # full torque, ideal driver
v_sim, t_sim, P_sim, lap_sim, E_sim, v_fwd = run_lap(torque_profile)     # driver's measured torque, compared to the test
ax_sim = v_sim * np.gradient(v_sim, s_sim) / G                            # g, a = v dv/ds
ay_sim = v_sim**2 * np.abs(k_sim) / G                                     # g
on_fwd = np.abs(v_sim - v_fwd) < 0.01                                     # sim driving or coasting, not braking
v_rev = np.max(v_free)                                                    # m/s, about the rev limit top speed

# ── 1. lap ────────────────────────────────────────────────────
E_test = np.sum(power) * dt / 3.6                                         # Wh, kW x s / 3.6
dist_ratio = s_test[-1] / track.lap_length
print("── lap ──")
print(f"distance:  test {s_test[-1]:.1f} m   map {track.lap_length:.1f} m   ratio {dist_ratio:.3f}   (1.00 = same lap)")
print(f"lap time:  test {t_test[-1]:.2f} s   sim {lap_sim:.2f} s   error {(lap_sim / t_test[-1] - 1) * 100:+.1f} %"
      f"   (full torque {lap_free:.2f} s, {Scoring.getAutocrossScore(lap_free):.1f} pts)")
print(f"energy:    test {E_test:.0f} Wh   sim {E_sim:.0f} Wh   error {(E_sim / E_test - 1) * 100:+.1f} %")
print(f"lateral:   test {np.percentile(np.abs(ay_test), 99):.2f} g   sim {np.percentile(ay_sim, 99):.2f} g   (99th pct)")
print(f"braking:   test {-np.percentile(ax_test, 1):.2f} g   sim {-np.percentile(ax_sim, 1):.2f} g   (99th pct)")
print(f"power:     test max {power.max():.1f} kW   sim max {P_sim.max():.1f} kW")

# ── 2. apex speeds (lateral grip) ─────────────────────────────
print("\n── apex speeds ──")
print(f"{'apex':>4} {'s [m]':>6} {'R [m]':>6} {'test':>7} {'sim':>7} {'error':>7}")
errs = []
apex_s, apex_v, apex_err = [], [], []                                     # for the dots on the graph
for n_a, i in enumerate(track.apex):
    s_apex = s_sim[i]                                                     # m
    near_t = np.abs(s_test - s_apex) < APEX_WINDOW
    near_s = np.abs(s_sim - s_apex) < APEX_WINDOW
    if not near_t.any():
        continue                                                          # apex past the end of the test lap
    v_t = np.min(V_test[near_t]) * 3.6                                    # km/h
    v_s = np.min(v_sim[near_s]) * 3.6                                     # km/h
    err = (v_s / v_t - 1) * 100                                           # %, + = sim faster
    errs.append(err)
    apex_s.append(s_apex); apex_v.append(v_s); apex_err.append(err)
    print(f"{n_a + 1:>4} {s_apex:>6.0f} {1 / abs(k_sim[i]):>6.1f} {v_t:>7.1f} {v_s:>7.1f} {err:>+6.1f}%")
print(f"mean apex error {np.mean(errs):+.1f} %   (+ = sim corners faster than the car, lower muyScale)")

# ── 3. accel on straights by speed (powertrain) ───────────────
print("\n── accel on straights, driving only ──")
print(f"{'V [m/s]':>8} {'test [g]':>9} {'sim [g]':>8} {'error':>7}")
st_t = (torque > 20) & (np.abs(ay_test) < 0.3)
st_s = (torque_profile > 20) & on_fwd & (ay_sim < 0.3) & (v_sim < 0.98 * v_rev)
for lo, hi in [(10, 14), (14, 18), (18, 22), (22, 26), (26, 29.5)]:
    mt = st_t & (V_test >= lo) & (V_test < hi)
    ms = st_s & (v_sim >= lo) & (v_sim < hi)
    if mt.sum() < 5 or ms.sum() < 5:
        continue
    a_t, a_s = np.mean(ax_test[mt]), np.mean(ax_sim[ms])
    print(f"{lo:>3}-{hi:<4} {a_t:>9.3f} {a_s:>8.3f} {(a_s / a_t - 1) * 100:>+6.1f}%")

# ── plots vs distance ─────────────────────────────────────────
delta = np.interp(s_test, s_sim, t_sim - t_sim[0]) - t_test   # s, negative = sim ahead

apex_s, apex_v, apex_err = np.array(apex_s), np.array(apex_v), np.array(apex_err)
fast = apex_err > APEX_FLAG                                    # sim corners too fast
slow = apex_err < -APEX_FLAG                                   # sim corners too slow

fig, axs = plt.subplots(4, 1, sharex=True, figsize=(12, 11))

axs[0].plot(s_test, V_test * 3.6, label="test")
axs[0].plot(s_sim, v_sim * 3.6, label="sim, measured torque")
axs[0].plot(s_sim, v_free * 3.6, "--", label="sim, full torque")
axs[0].plot(apex_s[fast], apex_v[fast], "o", color="red", ms=4, label=f"sim too fast (> +{APEX_FLAG:.0f} %)")
axs[0].plot(apex_s[slow], apex_v[slow], "o", color="green", ms=4, label=f"sim too slow (< -{APEX_FLAG:.0f} %)")
axs[0].set_ylabel("speed [km/h]")
axs[0].legend(fontsize=8)

axs[1].plot(s_test, delta)
axs[1].axhline(0, color="k", linewidth=0.8)
axs[1].set_ylabel("sim - test [s]\n(down = sim ahead)")

axs[2].plot(s_test, ax_test, label="test (IMU)")
axs[2].plot(s_sim, ax_sim, label="sim, measured torque")
axs[2].set_ylabel("ax [g]")
axs[2].legend()

axs[3].plot(s_test, power, label="test")
axs[3].plot(s_sim, P_sim, label="sim, measured torque")
axs[3].axhline(car.power_cap / 1000, color="r", linestyle="--", linewidth=0.8, label="power cap")
axs[3].set_ylabel("pack power [kW]")
axs[3].set_xlabel("distance [m]")
axs[3].legend()

for ax in axs:
    ax.grid(True)
    for i in track.apex:
        ax.axvline(s_sim[i], color="k", alpha=0.1)   # apex markers
plt.tight_layout()
plt.show()