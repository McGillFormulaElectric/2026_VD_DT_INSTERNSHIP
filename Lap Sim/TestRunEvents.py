# Author: Cris
# Summary: Runs all four dynamic events and prints a results table.
#          Includes a synthetic autocross/endurance track built to the FSAE
#          course rules (straight, turn, hairpin, and slalom limits), so the
#          lap solvers can be exercised before real Motec data is available.
#          To use real data: replace FakeMotec with the team's Motec reader.

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from carProperties import MFE27
from Tire import Tire
from TrackMap import TrackMap
import Acceleration, SkidPad, Autocross, Endurance

HERE = Path(__file__).parent


# ── Synthetic rules-compliant track ──────────────────────────────────────────
# FSAE course guidance (endurance): straights <= 77 m, constant turns ~30-54 m
# diameter, hairpins >= 9 m outside diameter, slalom cones 9-15 m spacing.
# The layout below stays inside those limits and closes into a ~500 m loop.

def _line(p0, p1, npts):
    t = np.linspace(0, 1, npts)[:, None]
    return np.array(p0) * (1 - t) + np.array(p1) * t

def _arc(cx, cy, R, a0, a1, npts):
    a = np.linspace(a0, a1, npts)
    return np.stack([cx + R * np.cos(a), cy + R * np.sin(a)], axis=1)

def build_rules_track():
    """Closed loop: 70 m straight, 20 m-radius sweeper, slalom (12 m cone
    spacing), 5 m-radius hairpin (10 m OD), medium corners home."""
    # slalom heading -x: weave of +/-1.6 m, wavelength 24 m (cones every 12 m)
    u = np.linspace(0, 60, 60)
    w = 1.6 * np.sin(2 * np.pi * u / 24)
    slalom = np.stack([70 - u, 40 + w], axis=1)

    return np.vstack([
        _line((0, 0), (70, 0), 40),                        # 70 m straight (<= 77)
        _arc(70, 20, 20, -np.pi / 2, np.pi / 2, 45),       # 40 m dia sweeper
        slalom,                                            # slalom section
        _arc(10, 45, 5, -np.pi / 2, -3 * np.pi / 2, 30),   # hairpin, 10 m OD (>= 9)
        _line((10, 50), (55, 50), 25),                     # 45 m straight
        _arc(55, 64, 14, -np.pi / 2, np.pi / 2, 35),       # 28 m dia right
        _line((55, 78), (-15, 78), 40),                    # 70 m back straight
        _arc(-15, 51.5, 26.5, np.pi / 2, np.pi, 30),       # 53 m dia sweeper
        _line((-41.5, 51.5), (-41.5, 20), 20),
        _arc(-21.5, 20, 20, np.pi, 3 * np.pi / 2, 35),     # 40 m dia back to start
    ])


class FakeMotec:
    """Wraps synthetic x,y (meters) as the GPS channels TrackMap expects.
    Swap this for the real Motec reader to run actual logged data."""
    def __init__(self, pts, lat0=45.6):
        Re = 6371000.0
        self.d = {
            "GPS_Latitude":  lat0 + np.degrees(pts[:, 0] / Re),
            "GPS_Longitude": np.degrees(pts[:, 1] / (Re * np.cos(np.radians(lat0)))),
        }
    def getValue(self, key):
        return self.d[key]


# ── Run all events ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    car = MFE27()
    tire = Tire(HERE / "MF61_Coefficients.csv")

    motec = FakeMotec(build_rules_track())
    # NOTE: createTrack mutates and returns the same TrackMap object, so each
    # event needs its own TrackMap instance (or tracks silently alias).
    t_ax = TrackMap(motec).createTrack(event="autocross", ds=0.5)
    t_en = TrackMap(motec).createTrack(event="endurance", ds=0.5)

    print(f"track: {t_ax.lap_length:.0f} m lap, "
          f"R_min = {1 / np.abs(t_ax.k).max():.1f} m\n")

    # accel and skidpad: class solvers, argument order (tire, car)
    r_ac = Acceleration.AccelSolver().simulate(tire, car)
    r_sp = SkidPad.SkidPadSolver().simulate(tire, car)

    # autocross and endurance: function solvers, (track, car, tire)
    r_ax = Autocross.solve(t_ax, car, tire)
    r_en = Endurance.solve(t_en, car, tire)

    # ── Results table ────────────────────────────────────────────────────────
    print(f"{'EVENT':<12} {'TIME':>10}   NOTES")
    print(f"{'Accel':<12} {r_ac['time_total']:>8.3f} s   "
          f"exit {r_ac['v_final'] * 3.6:.0f} km/h, "
          f"{r_ac['EnergyPack'] / 3.6e3:.0f} Wh")
    print(f"{'Skidpad':<12} {r_sp['time']:>8.3f} s   "
          f"{r_sp['AyG']:.2f} g, {r_sp['limit']}-limited")
    print(f"{'Autocross':<12} {r_ax['lap_time']:>8.2f} s   "
          f"{r_ax['energy_kWh'] * 1000:.0f} Wh, avg {r_ax['avg_power_kW']:.0f} kW")
    print(f"{'Endurance':<12} {r_en['total_time']:>8.1f} s   "
          f"{r_en['n_laps']} laps ({r_en['first_lap_time']:.2f} + "
          f"{r_en['n_laps'] - 1}x{r_en['flying_lap_time']:.2f}), "
          f"{r_en['total_energy_kWh']:.2f} kWh flat-out")

    # ── Figure: track map + speed profiles ───────────────────────────────────
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5),
                                 gridspec_kw={"width_ratios": [1, 1.4]})
    n = min(len(t_ax.x), len(r_ax["v"]))
    sc = a1.scatter(t_ax.y[:n], t_ax.x[:n], c=r_ax["v"][:n] * 3.6,
                    cmap="turbo", s=5)
    fig.colorbar(sc, ax=a1, label="Speed [km/h]")
    a1.set_aspect("equal")
    a1.set_title("Rules-compliant synthetic track")

    a2.plot(r_ax["s"], r_ax["v_max"] * 3.6, color="#999", lw=1, label="v_max")
    a2.plot(r_ax["s"], r_ax["v"] * 3.6, "k", lw=1.6,
            label="autocross (standing start)")
    a2.plot(r_en["s"], r_en["v"] * 3.6, color="#2b8cbe", lw=1.2, alpha=0.8,
            label="endurance flying lap")
    a2.set_xlabel("s [m]")
    a2.set_ylabel("Speed [km/h]")
    a2.legend(fontsize=9)
    a2.grid(alpha=0.3)
    a2.set_title("Speed profiles")

    plt.tight_layout()
    plt.savefig(HERE / "event_results.png", dpi=130)
    plt.show()