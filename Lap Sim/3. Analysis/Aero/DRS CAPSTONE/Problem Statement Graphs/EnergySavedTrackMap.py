"""
DRS_TrackMapEnergy.py  -  track map coloured by the energy DRS saves, point by point.

"""
import sys
from pathlib import Path
import csv
import numpy as np
import matplotlib
try:
    import tkinter
    matplotlib.use("TkAgg")
except ImportError:
    pass
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm

# ── find the Lap Sim root and the DRS capstone folder, however deep this file sits
HERE = Path(__file__).resolve().parent
LAPSIM_ROOT = next(p for p in [HERE, *HERE.parents] if (p / "SolverFunctions.py").exists())
DRS_DIR = LAPSIM_ROOT / "3. Analysis" / "Aero" / "DRS CAPSTONE"
sys.path.insert(0, str(LAPSIM_ROOT))
sys.path.insert(0, str(DRS_DIR))

from TrackMap import loadTrack
from carProperties import MFE26
from Tire import Tire
from Endurance import N_LAPS
import DRS_Physics as DRS

# ── settings ──────────────────────────────────────────────
TRACK = "Michigan2026_Endurance"
CONCEPT = "C1"                 # "C1" dual rear flap, "C2" all rear flaps
MODE = "controller"            # "controller" (what the car will do) or "ideal" (upper bound)
T_ACT = 0.0                    # s, actuator latency
LAT_MARGIN = 0.95

CLOSED = dict(CLA=4.43, CDA=1.92, bal=0.41)                        # C0 baseline, CFD tab
OPEN = {"C1": dict(CLA=3.23, CDA=1.14, bal=0.65),
        "C2": dict(CLA=2.960, CDA=1.072, bal=0.66)}[CONCEPT]
DRS_MASS = {"C1": 0.15, "C2": 0.2}[CONCEPT]                        # kg

OUT_CSV = HERE / f"DRS_{CONCEPT}_trackmap_energy.csv"
OUT_PNG = HERE / f"DRS_{CONCEPT}_trackmap_energy.png"

TIRE_DIR = LAPSIM_ROOT / "1. Data" / "Tire"
TRACK_DIR = LAPSIM_ROOT / "1. Data" / "Track"

# ── run ───────────────────────────────────────────────────
tire = Tire(TIRE_DIR / "MF61_Parameters.mat", TIRE_DIR / "FZ_Reference.mat")
track = loadTrack(TRACK_DIR, TRACK)
car_c, car_o = DRS.make_states(MFE26(), CLOSED["CLA"], CLOSED["CDA"], CLOSED["bal"],
                               OPEN["CLA"], OPEN["CDA"], OPEN["bal"], DRS_MASS)

base = DRS.solve_track_event(tire, car_c, car_o, track, False, mode="closed")
drs  = DRS.solve_track_event(tire, car_c, car_o, track, False, mode=MODE,
                             t_act=T_ACT, lat_margin=LAT_MARGIN)

# pack energy per metre at each point: P [W] * dt [s] / ds [m] = P / v  [J/m]
e_base = base["P_pack"] / np.maximum(base["v"], 0.1)        # J/m
e_drs  = drs["P_pack"]  / np.maximum(drs["v"], 0.1)         # J/m
saved  = (e_base - e_drs) / 3600.0                          # Wh/m, positive = DRS saves
cum_saved = np.concatenate(([0.0], np.cumsum(saved[:-1] * track.ds)))   # Wh along the lap

E_base = base["energy_J"] / 3.6e6
E_drs  = drs["energy_J"] / 3.6e6
dE = E_base - E_drs
dt_lap = base["lap_time"] - drs["lap_time"]

print(f"{CONCEPT} {MODE}, {TRACK}")
print(f"  closed  : {base['lap_time']:.3f} s   {E_base:.4f} kWh/lap")
print(f"  DRS     : {drs['lap_time']:.3f} s   {E_drs:.4f} kWh/lap   open {100*drs['open_fraction']:.1f} % of lap")
print(f"  saving  : {dt_lap:.3f} s/lap   {dE*1000:.1f} Wh/lap   {dE*N_LAPS:.3f} kWh over {N_LAPS} laps "
      f"({100*dE/E_base:.1f} %)")

# ── csv ───────────────────────────────────────────────────
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["s m", "x m", "y m", "curvature 1/m",
                "v closed m/s", "v DRS m/s", "DRS open",
                "P pack closed kW", "P pack DRS kW",
                "energy closed Wh/m", "energy DRS Wh/m",
                "saved Wh/m", "cumulative saved Wh"])
    for i in range(len(track.s)):
        w.writerow([f"{track.s[i]:.2f}", f"{track.x[i]:.2f}", f"{track.y[i]:.2f}", f"{track.k[i]:.5f}",
                    f"{base['v'][i]:.3f}", f"{drs['v'][i]:.3f}", int(drs["state"][i]),
                    f"{base['P_pack'][i]/1000:.3f}", f"{drs['P_pack'][i]/1000:.3f}",
                    f"{e_base[i]/3600:.5f}", f"{e_drs[i]/3600:.5f}",
                    f"{saved[i]:.5f}", f"{cum_saved[i]:.3f}"])
    w.writerow([])
    w.writerow(["lap time closed s", f"{base['lap_time']:.3f}"])
    w.writerow(["lap time DRS s", f"{drs['lap_time']:.3f}"])
    w.writerow(["energy closed kWh/lap", f"{E_base:.4f}"])
    w.writerow(["energy DRS kWh/lap", f"{E_drs:.4f}"])
    w.writerow([f"saved kWh over {N_LAPS} laps", f"{dE*N_LAPS:.4f}"])
    w.writerow(["DRS open fraction", f"{drs['open_fraction']:.4f}"])
print(f"csv  -> {OUT_CSV}")

# ── plot ──────────────────────────────────────────────────
def coloured_track(ax, values, cmap, vmin, vmax, size=60, **kw):
    """Track drawn as dense points coloured by `values` (smooth, no beading on bends)."""
    sc = ax.scatter(track.x, track.y, c=values, cmap=cmap, vmin=vmin, vmax=vmax,
                    s=size, edgecolors="none", **kw)
    ax.set_xlim(track.x.min() - 12, track.x.max() + 12)
    ax.set_ylim(track.y.min() - 12, track.y.max() + 12)
    ax.set_aspect("equal")
    return sc

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
ax_map = fig.add_subplot(gs[0, :])
ax_open = fig.add_subplot(gs[1, 0])
ax_cum = fig.add_subplot(gs[1, 1])

# map 1: local energy saved, sequential (DRS almost never costs energy, so no diverging scale)
vmax = np.percentile(saved, 99.5)
ax_map.plot(track.x, track.y, color="#d0d0d0", lw=14, solid_capstyle="round", zorder=0)   # grey underlay = track
sc = coloured_track(ax_map, np.clip(saved, 0, None), "turbo", 0.0, vmax, size=90, zorder=2)
cb = fig.colorbar(sc, ax=ax_map, fraction=0.025, pad=0.01)
cb.set_label("energy saved by DRS [Wh / m]")
# start/finish and direction
ax_map.plot(track.x[0], track.y[0], "k*", ms=18, zorder=5, label="start / finish")
ax_map.annotate("", xy=(track.x[40], track.y[40]), xytext=(track.x[0], track.y[0]),
                arrowprops=dict(arrowstyle="-|>", lw=2.5, color="k"), zorder=5)
ax_map.set_title(f"Where DRS saves energy  -  {TRACK}\n"
                 f"{dE*1000:.0f} Wh/lap  ({100*dE/E_base:.1f} %)   {dE*N_LAPS:.2f} kWh over {N_LAPS} laps   "
                 f"{dt_lap:.2f} s/lap faster", fontsize=12)
ax_map.set_xlabel("x [m]")
ax_map.set_ylabel("y [m]")
ax_map.legend(loc="lower right")
ax_map.grid(True, alpha=0.3)

# map 2: speed, with the wing-open sections drawn thick and the closed sections thin
v_kmh = drs["v"] * 3.6
ax_open.plot(track.x, track.y, color="#d0d0d0", lw=10, solid_capstyle="round", zorder=0)
sc2 = coloured_track(ax_open, v_kmh, "plasma", v_kmh.min(), v_kmh.max(), size=12, zorder=1)
openc = np.where(drs["state"], v_kmh, np.nan)
coloured_track(ax_open, openc, "plasma", v_kmh.min(), v_kmh.max(), size=70, zorder=2)
cb2 = fig.colorbar(sc2, ax=ax_open, fraction=0.04, pad=0.01)
cb2.set_label("speed [km/h]")
ax_open.set_title(f"wing open (thick) vs closed (thin)   -   open {100*drs['open_fraction']:.0f} % of the lap", fontsize=11)
ax_open.set_xticks([]); ax_open.set_yticks([])

# cumulative saving around the lap, with open zones shaded
ax_cum.plot(track.s, cum_saved, color="#1b7f3b", lw=2.5)
ax_cum.fill_between(track.s, 0, cum_saved, color="#1b7f3b", alpha=0.15)
ax_cum.fill_between(track.s, ax_cum.get_ylim()[0], cum_saved.max() * 1.05,
                    where=drs["state"], color="#2a78d6", alpha=0.15, label="DRS open")
ax_cum.set_xlim(0, track.s[-1])
ax_cum.set_xlabel("distance along lap [m]")
ax_cum.set_ylabel("cumulative energy saved [Wh]")
ax_cum.set_title("energy saved accumulating around the lap", fontsize=10)
ax_cum.grid(True, alpha=0.3)
ax_cum.legend(loc="upper left")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f"plot -> {OUT_PNG}")
plt.show()