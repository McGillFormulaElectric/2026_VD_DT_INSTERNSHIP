import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Path to the Lap Sim folder
LAPSIM_ROOT = Path(__file__).resolve().parents[1]   # folder that holds CarProperties.py
sys.path.insert(0, str(LAPSIM_ROOT))

from TrackMap import loadTrack
from CarProperties import MFE26
from Tire import Tire
from SolverFunctions import corner_speed_ceiling, braking_decel
from Autocross import solve as autocross_solve

# setup
TIRE_DIR = LAPSIM_ROOT / "1. Data" / "Tire"
TRACK_DIR = LAPSIM_ROOT / "1. Data" / "Track"

BIASES = np.arange(0.40, 0.801, 0.01)   # brake_bias_front, fraction of braking force on the front axle
SPEEDS = [10, 15, 20, 25, 29]            # m/s

car = MFE26()
tire = Tire(TIRE_DIR / "MF61_Parameters.mat", TIRE_DIR / "FZ_Reference.mat")
track = loadTrack(TRACK_DIR, "Michigan2026_Autocross")
baseline = car.brake_bias_front

def max_decel(v):
    # straight line max braking [g], load transfer converged by repeating
    dec = 0.0                                                  # m/s^2
    for _ in range(10):
        dec = braking_decel(car, tire, v, 1e3, k=0.0, ax_prev=-dec)   # v_max far away = no cornering
    return dec / car.g

# sweep
decel = {v: [] for v in SPEEDS}   # g
lap = []                          # s
v_max = corner_speed_ceiling(track, car, tire)   # bias does not change corner speed, compute once
for b in BIASES:
    car.brake_bias_front = b
    for v in SPEEDS:
        decel[v].append(max_decel(v))
    lap.append(autocross_solve(tire, car, track, v_max=v_max)["lap_time"])
car.brake_bias_front = baseline
lap = np.asarray(lap)

# print best bias per speed
print(f"current brake_bias_front {baseline:.2f}   max_decel cap {car.max_decel / car.g:.2f} g")
for v in SPEEDS:
    d = np.asarray(decel[v])
    i = np.argmax(d)
    print(f"{v:>3} m/s: best bias {BIASES[i]:.2f} -> {d[i]:.3f} g   (current {np.interp(baseline, BIASES, d):.3f} g)")
i = np.argmin(lap)
print(f"autocross: best bias {BIASES[i]:.2f} -> {lap[i]:.3f} s   (current {np.interp(baseline, BIASES, lap):.3f} s)")

# plot
fig, (ax_d, ax_l) = plt.subplots(1, 2, figsize=(14, 5.5))
for v in SPEEDS:
    d = np.asarray(decel[v])
    i = np.argmax(d)
    ax_d.plot(BIASES, d, label=f"{v} m/s ({v * 3.6:.0f} km/h)")
    ax_d.plot(BIASES[i], d[i], "ko", markersize=4)
ax_d.axvline(baseline, color="k", linestyle="--", label=f"current {baseline:.2f}")
ax_d.set_xlabel("brake bias front (-)")
ax_d.set_ylabel("max deceleration (g)")
ax_d.set_title("straight line braking", fontweight="bold")
ax_d.grid(True)
ax_d.legend()

i = np.argmin(lap)
ax_l.plot(BIASES, lap, linewidth=2)
ax_l.plot(BIASES[i], lap[i], "ro", label=f"best {BIASES[i]:.2f}")
ax_l.axvline(baseline, color="k", linestyle="--", label=f"current {baseline:.2f}")
ax_l.ticklabel_format(axis="y", useOffset=False)
ax_l.set_xlabel("brake bias front (-)")
ax_l.set_ylabel("autocross lap time (s)")
ax_l.set_title("Michigan 2026 autocross", fontweight="bold")
ax_l.grid(True)
ax_l.legend()
plt.tight_layout()
plt.show()