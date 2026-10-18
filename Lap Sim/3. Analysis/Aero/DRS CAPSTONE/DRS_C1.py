"""DRS capstone: C0 baseline vs C1 dual rear flap DRS, all events.

Aero from DRS_CAPSTONE_NOTEBOOK.xlsx (CFD tab), C1 row:
  closed  CdA 1.92  ClA 4.43  CoP 41 % front
  open    CdA 1.14  ClA 3.23  CoP 65 % front   +0.15 kg
Controller: open when not grip limited (see DRS.py). "ideal" = per-point best state,
the upper bound. Sweeps: actuator latency, added mass.

Outputs (same folder): DRS_C1_results.csv, DRS_C1_*.png
"""
import sys, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAPSIM_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(LAPSIM_ROOT))
OUT = Path(__file__).resolve().parent

from TrackMap import loadTrack
from CarProperties import MFE26
from Tire import Tire
from SkidPad import SkidPadSolver
from Scoring import Scoring
import DRS_Physics as DRS

TIRE_DIR = LAPSIM_ROOT / "1. Data" / "Tire"
TRACK_DIR = LAPSIM_ROOT / "1. Data" / "Track"
N_LAPS = 22

# ── inputs ───────────────────────────────────────────────────────────────
CLOSED = dict(CLA=4.43, CDA=1.92, bal=0.41)
OPEN   = dict(CLA=3.23, CDA=1.14, bal=0.65)
DRS_MASS = 0.15        # kg
LAT_MARGIN = 0.95      # open only if v < 0.95 x open-wing corner ceiling
LATENCIES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]   # s
MASSES = [0.0, 0.15, 0.5, 1.0, 2.0, 4.0]           # kg, breakeven sweep

C = {"c0": "#898781", "c1": "#2a78d6", "ideal": "#eb6834", "open": "#1baf7a"}

tire = Tire(TIRE_DIR / "MF61_Parameters.mat", TIRE_DIR / "FZ_Reference.mat")
base = MFE26()
track_ax = loadTrack(TRACK_DIR, "Michigan2026_Autocross")
track_en = loadTrack(TRACK_DIR, "Michigan2026_Endurance")


def states(mass, open_=OPEN):
    return DRS.make_states(base, CLOSED["CLA"], CLOSED["CDA"], CLOSED["bal"],
                           open_["CLA"], open_["CDA"], open_["bal"], mass)


def run_all(mode, mass=DRS_MASS, t_act=0.0, lat_margin=LAT_MARGIN):
    cc, co = states(mass)
    acc = DRS.solve_accel(tire, cc, co, mode=mode, t_act=t_act)
    skid = SkidPadSolver().simulate(tire, cc)                 # always closed
    ax = DRS.solve_track_event(tire, cc, co, track_ax, True, mode, t_act, lat_margin)
    en = DRS.solve_track_event(tire, cc, co, track_en, False, mode, t_act, lat_margin)
    t_en = en["lap_time"] * N_LAPS
    E_en = en["energy_J"] * N_LAPS / 3.6e6
    r = {"mode": mode, "mass": mass, "t_act": t_act,
         "t_acc": acc["lap_time"], "t_skid": skid["lap_time"],
         "t_ax": ax["lap_time"], "t_en_lap": en["lap_time"], "t_en": t_en, "E_en_kWh": E_en,
         "open_ax": ax["open_fraction"], "open_en": en["open_fraction"], "open_acc": acc["open_fraction"],
         "pts_acc": Scoring.getAccelScore(acc["lap_time"]),
         "pts_skid": Scoring.getSkidpadScore(skid["lap_time"]),
         "pts_ax": Scoring.getAutocrossScore(ax["lap_time"]),
         "pts_en": Scoring.getEnduranceScore(t_en),
         "pts_eff": Scoring.getEfficiencyScore(t_en, E_en, track_en),
         "_acc": acc, "_ax": ax, "_en": en}
    r["pts_total"] = r["pts_acc"] + r["pts_skid"] + r["pts_ax"] + r["pts_en"] + r["pts_eff"]
    # uncapped points: slope of the scoring formulas without the 125 / 250 / 100 ceilings
    r["pts_ax_unc"] = 118.5 * ((Scoring.T_MIN_AUTOCROSS * 1.45 / ax["lap_time"]) - 1) / 0.45 + 6.5
    r["pts_en_unc"] = 250 * ((Scoring.T_MIN_ENDURANCE * 1.45 / t_en) - 1) / 0.45 + 25
    r["pts_acc_unc"] = 95.5 * ((Scoring.T_MIN_ACCEL * 1.5 / acc["lap_time"]) - 1) / 0.5 + 4.5
    r["pts_total_unc"] = r["pts_acc_unc"] + r["pts_skid"] + r["pts_ax_unc"] + r["pts_en_unc"] + r["pts_eff"]
    return r


# ── main comparison ──────────────────────────────────────────────────────
print("running C0 / C1 controller / C1 ideal ...")
c0 = run_all("closed", mass=0.0)
c1 = run_all("controller")
ideal = run_all("ideal")

def fmt(r):
    return (f"{r['mode']:<11} acc {r['t_acc']:.3f}s  skid {r['t_skid']:.3f}s  ax {r['t_ax']:.3f}s  "
            f"en lap {r['t_en_lap']:.3f}s  E {r['E_en_kWh']:.3f} kWh  open ax/en {r['open_ax']*100:.0f}/{r['open_en']*100:.0f}%  "
            f"pts {r['pts_total']:.1f} (uncapped {r['pts_total_unc']:.1f})")
for r in (c0, c1, ideal):
    print(fmt(r))

# ── latency sweep ────────────────────────────────────────────────────────
print("latency sweep ...")
lat = [run_all("controller", t_act=ta) for ta in LATENCIES]

# ── mass sweep ───────────────────────────────────────────────────────────
print("mass sweep ...")
ms = [run_all("controller", mass=m) for m in MASSES]

# ── csv ──────────────────────────────────────────────────────────────────
keys = ["mode", "mass", "t_act", "t_acc", "t_skid", "t_ax", "t_en_lap", "t_en", "E_en_kWh",
        "open_acc", "open_ax", "open_en", "pts_acc", "pts_skid", "pts_ax", "pts_en", "pts_eff",
        "pts_total", "pts_acc_unc", "pts_ax_unc", "pts_en_unc", "pts_total_unc"]
with open(OUT / "DRS_C1_results.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["case"] + keys)
    for name, r in [("C0 baseline", c0), ("C1 controller", c1), ("C1 ideal", ideal)] + \
                   [(f"C1 latency {r['t_act']}s", r) for r in lat] + \
                   [(f"C1 mass {r['mass']}kg", r) for r in ms]:
        w.writerow([name] + [r[k] for k in keys])

# ── plots ────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "xtick.color": "#898781",
                     "ytick.color": "#898781", "grid.color": "#e1e0d9", "axes.grid": True, "axes.axisbelow": True,
                     "figure.facecolor": "white", "axes.facecolor": "white"})

# 1. event breakdown: time deltas and uncapped point deltas
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
ev = ["Accel", "Skidpad", "Autocross", "Endurance\n(22 laps)"]
tk = ["t_acc", "t_skid", "t_ax", "t_en"]
x = np.arange(len(ev)); w = 0.36
d_c1 = [c1[k] - c0[k] for k in tk]; d_id = [ideal[k] - c0[k] for k in tk]
a1.bar(x - w/2, d_c1, w, color=C["c1"], label="C1 controller")
a1.bar(x + w/2, d_id, w, color=C["ideal"], label="C1 ideal switch")
a1.axhline(0, color="#c3c2b7", lw=1)
a1.set_xticks(x); a1.set_xticklabels(ev); a1.set_ylabel("time vs baseline [s]  (negative = faster)")
for i, (v1, v2) in enumerate(zip(d_c1, d_id)):
    a1.text(i - w/2, v1, f"{v1:+.2f}", ha="center", va="top", fontsize=8, color="#52514e")
    a1.text(i + w/2, v2, f"{v2:+.2f}", ha="center", va="top", fontsize=8, color="#52514e")
a1.legend(frameon=False); a1.set_title("Event time delta", loc="left", fontweight="bold")
pk = ["pts_acc_unc", "pts_skid", "pts_ax_unc", "pts_en_unc", "pts_eff"]
ev2 = ["Accel", "Skidpad", "Autocross", "Endurance", "Efficiency"]
x2 = np.arange(len(ev2))
p_c1 = [c1[k] - c0[k] for k in pk]; p_id = [ideal[k] - c0[k] for k in pk]
a2.bar(x2 - w/2, p_c1, w, color=C["c1"], label="C1 controller")
a2.bar(x2 + w/2, p_id, w, color=C["ideal"], label="C1 ideal switch")
a2.axhline(0, color="#c3c2b7", lw=1)
a2.set_xticks(x2); a2.set_xticklabels(ev2); a2.set_ylabel("points vs baseline (uncapped)")
for i, (v1, v2) in enumerate(zip(p_c1, p_id)):
    a2.text(i - w/2, v1, f"{v1:+.1f}", ha="center", va="bottom", fontsize=8, color="#52514e")
    a2.text(i + w/2, v2, f"{v2:+.1f}", ha="center", va="bottom", fontsize=8, color="#52514e")
a2.set_title(f"Points delta  (total {c1['pts_total_unc']-c0['pts_total_unc']:+.1f} controller, "
             f"{ideal['pts_total_unc']-c0['pts_total_unc']:+.1f} ideal)", loc="left", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "DRS_C1_event_breakdown.png", dpi=160); plt.close(fig)

# 2. endurance lap: speed, wing state, cumulative time gain, grip utilisation vs distance
def lap_plot(r0, r1, track, name, fname):
    s = r1["s"]; v0 = r0["v"]; v1 = r1["v"]; st = r1["state"]
    dt0 = np.cumsum(track.ds / np.maximum(v0, 0.1)); dt1 = np.cumsum(track.ds / np.maximum(v1, 0.1))
    util = np.minimum((v1 / np.maximum(r1["v_max_open"], 1e-9))**2, 1.0)   # vs open-wing GRIP ceiling, the controller check
    fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True,
                           gridspec_kw={"height_ratios": [3, 0.7, 1.5, 1.5]})
    ax[0].plot(s, v0 * 3.6, color=C["c0"], lw=1.5, label="C0 baseline")
    ax[0].plot(s, v1 * 3.6, color=C["c1"], lw=1.5, label="C1 controller")
    ax[0].fill_between(s, 0, 1, where=st, transform=ax[0].get_xaxis_transform(), color=C["open"], alpha=0.12, lw=0)
    ax[0].set_ylabel("speed [km/h]"); ax[0].legend(frameon=False, loc="lower right")
    ax[0].set_title(f"{name}: C1 dual rear flap DRS, green = wing open ({r1['open_fraction']*100:.0f} % of lap)",
                    loc="left", fontweight="bold")
    ax[1].fill_between(s, 0, st.astype(float), color=C["open"], step="mid", lw=0)
    ax[1].set_yticks([0, 1]); ax[1].set_yticklabels(["closed", "open"]); ax[1].set_ylabel("wing")
    ax[2].plot(s, dt1 - dt0, color=C["c1"], lw=1.5)
    ax[2].axhline(0, color="#c3c2b7", lw=1); ax[2].set_ylabel("cum. time\nvs C0 [s]")
    ax[3].plot(s, util, color="#52514e", lw=1.2, label="lateral grip use with wing open, (v/v_grip,open)²")
    ax[3].axhline(LAT_MARGIN**2, color=C["ideal"], lw=1, ls="--", label=f"open threshold ({LAT_MARGIN}²)")
    ax[3].set_ylim(0, 1.05); ax[3].set_ylabel("lateral\ngrip use"); ax[3].set_xlabel("distance [m]"); ax[3].legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=160); plt.close(fig)

lap_plot(c0["_en"], c1["_en"], track_en, "Endurance flying lap", "DRS_C1_endurance_lap.png")
lap_plot(c0["_ax"], c1["_ax"], track_ax, "Autocross", "DRS_C1_autocross_lap.png")

# 3. wing state on the track map + where the time is gained
def map_plot(r0, r1, track, name, fname):
    st = r1["state"]; x, y = track.x[:len(st)], track.y[:len(st)]
    dt_gain = (track.ds / np.maximum(r0["v"], 0.1) - track.ds / np.maximum(r1["v"], 0.1))
    fig, (m1, m2) = plt.subplots(2, 1, figsize=(12, 9))
    m1.plot(x, y, color="#e1e0d9", lw=5, solid_capstyle="round")
    m1.scatter(x[st], y[st], s=10, color=C["open"], lw=0, zorder=3, label="wing open")
    m1.scatter(x[~st], y[~st], s=4, color=C["c0"], lw=0, zorder=3, label="closed")
    m1.set_aspect("equal"); m1.grid(False); m1.axis("off"); m1.legend(frameon=False, loc="upper left", markerscale=3)
    m1.set_title(f"{name}: wing state", loc="left", fontweight="bold")
    sc = m2.scatter(x, y, c=dt_gain * 1000, s=12, cmap="Blues", lw=0, vmin=0, zorder=3)
    m2.set_aspect("equal"); m2.grid(False); m2.axis("off")
    cb = fig.colorbar(sc, ax=m2, shrink=0.7); cb.set_label("time gained per 0.25 m step [ms]")
    m2.set_title("where the lap time comes from", loc="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / fname, dpi=160); plt.close(fig)

map_plot(c0["_en"], c1["_en"], track_en, "Endurance", "DRS_C1_endurance_map.png")
map_plot(c0["_ax"], c1["_ax"], track_ax, "Autocross", "DRS_C1_autocross_map.png")

# 4. sensitivities: gain vs latency, gain vs mass
fig, (g1, g2) = plt.subplots(1, 2, figsize=(11, 4))
ta = [r["t_act"] for r in lat]
g1.plot(ta, [r["pts_total_unc"] - c0["pts_total_unc"] for r in lat], "o-", color=C["c1"], lw=2, ms=6, label="total (uncapped)")
g1.plot(ta, [r["pts_en_unc"] + r["pts_eff"] - c0["pts_en_unc"] - c0["pts_eff"] for r in lat], "o-", color=C["open"], lw=2, ms=6, label="endurance + efficiency")
g1.plot(ta, [r["pts_ax_unc"] - c0["pts_ax_unc"] for r in lat], "o-", color=C["ideal"], lw=2, ms=6, label="autocross")
g1.plot(ta, [r["pts_acc_unc"] - c0["pts_acc_unc"] for r in lat], "o-", color="#6250d6", lw=2, ms=6, label="accel")
g1.axhline(0, color="#c3c2b7", lw=1); g1.set_xlabel("actuator latency, open and close [s]"); g1.set_ylabel("points vs baseline")
g1.legend(frameon=False); g1.set_title("Gain vs actuator latency", loc="left", fontweight="bold")
mm = [r["mass"] for r in ms]
g2.plot(mm, [r["pts_total_unc"] - c0["pts_total_unc"] for r in ms], "o-", color=C["c1"], lw=2, ms=6)
g2.axhline(0, color="#c3c2b7", lw=1); g2.axvline(DRS_MASS, color=C["open"], lw=1, ls="--")
g2.text(DRS_MASS, g2.get_ylim()[1] * 0.95, "  C1 estimate 0.15 kg", color=C["open"], va="top", fontsize=8)
g2.set_xlabel("DRS system added mass [kg]"); g2.set_ylabel("points vs baseline (uncapped)")
g2.set_title("Gain vs added mass (breakeven)", loc="left", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "DRS_C1_sensitivity.png", dpi=160); plt.close(fig)

# 5. accel run
fig, a = plt.subplots(figsize=(8, 3.5))
a.plot(c0["_acc"]["s"], c0["_acc"]["v"] * 3.6, color=C["c0"], lw=1.5, label="C0")
a.plot(c1["_acc"]["s"], c1["_acc"]["v"] * 3.6, color=C["c1"], lw=1.5, label="C1 controller")
a.fill_between(c1["_acc"]["s"], 0, 1, where=c1["_acc"]["state"], transform=a.get_xaxis_transform(), color=C["open"], alpha=0.12, lw=0)
s_open = c1["_acc"]["s"][c1["_acc"]["state"]]
a.set_xlabel("distance [m]"); a.set_ylabel("speed [km/h]"); a.legend(frameon=False)
a.set_title(f"Accel: wing opens at {s_open[0]:.1f} m ({c1['_acc']['v'][c1['_acc']['state']][0]*3.6:.0f} km/h), "
            f"{c0['t_acc']:.3f} s -> {c1['t_acc']:.3f} s", loc="left", fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "DRS_C1_accel.png", dpi=160); plt.close(fig)
print("done")

print("LAPSIM_ROOT =", LAPSIM_ROOT)
print([p.name for p in LAPSIM_ROOT.glob("*.py")])