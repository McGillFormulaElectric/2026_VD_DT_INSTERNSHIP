# Author : Lap Sim team
# Summary: Convert the MATLAB lap sim's track files ("0 - Tracks/*.mat", a struct
#          `track` with x, y in metres) into this sim's .npz format.
#          The x, y points are re-fit through TrackMap.buildFromGPS so curvature,
#          spacing and closure are computed the same way as for Motec-built tracks.
#
# Usage:   python ImportMatlabTracks.py "path/to/Pseudo-TwoTrack-LapSim/0 - Tracks"
#          -> writes Data/Track/<name>.npz for every file in TRACKS below

import sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from TrackMap import TrackMap, saveTrack

HERE = Path(__file__).resolve().parent
OUT = HERE / "Data" / "Track"

# matlab file -> (npz name, event used for n_laps, smooth_m, thin_m)
#   smooth_m : allowed spline deviation [m]. Michigan files were drawn in a track
#              builder with sharp polyline corners -> 1.0 rounds them to real
#              hairpin radii (>= ~4.5 m). GPS-logged tracks are fine at 0.5.
#   thin_m   : resample the raw points to this spacing before fitting (None = keep).
#              The Lincoln files are 0.25 m dense and ring with heavy smoothing.
TRACKS = {
    "Autocross_Michigan_2022.mat": ("Autocross_Michigan_2022", "autocross", 1.0, None),
    "Endurance_Michigan_2022.mat": ("Endurance_Michigan_2022", "endurance", 1.0, None),
    "Endurance_Michigan_2024.mat": ("Endurance_Michigan_2024", "endurance", 1.0, None),
    "Autocross_Lincoln_2016.mat":  ("Autocross_Lincoln_2016",  "autocross", 0.5, 2.0),
    "Endurance_Lincoln_2016.mat":  ("Endurance_Lincoln_2016",  "endurance", 0.5, 2.0),
    "TorontoAutoX.mat":            ("Toronto_Autocross",       "autocross", 0.5, None),
    "TorontoEndurance.mat":        ("Toronto_Endurance",       "endurance", 0.5, None),
}


class XYSource:
    """Stand-in for MotecData: TrackMap.__init__ reads lat/lon, so hand it x, y
    disguised as degrees around an arbitrary origin (lat0 = 45 deg)."""
    def __init__(self, x, y, lat0=45.0):
        R = 6371000.0
        self.d = {"lat": lat0 + np.degrees(x / R),
                  "lon": np.degrees(y / (R * np.cos(np.radians(lat0))))}
    def getValue(self, key):
        return self.d[key]


def import_track(mat_path, name, event, smooth_m=0.5, thin_m=None, ds=0.25):
    m = loadmat(mat_path, squeeze_me=True, struct_as_record=False)["track"]
    x, y = np.asarray(m.x, float).ravel(), np.asarray(m.y, float).ravel()

    # some files repeat the closing point or contain a lap-change jump; drop
    # any step > 20x the median so the spline does not try to fit it
    step = np.hypot(np.diff(x), np.diff(y))
    keep = np.append(True, step < 20 * np.median(step))
    x, y = x[keep], y[keep]

    if thin_m:                          # resample to a coarser, even spacing
        d = np.append(0, np.cumsum(np.hypot(np.diff(x), np.diff(y))))
        grid = np.arange(0, d[-1], thin_m)
        x, y = np.interp(grid, d, x), np.interp(grid, d, y)

    t = TrackMap(XYSource(x, y))
    # createTrack has no smooth_m argument, so call buildFromGPS directly
    t.x, t.y, t.s, t.k = t.buildFromGPS(ds, smooth_m=smooth_m)
    t.ds = ds
    t.lap_length = t.s[-1]
    t.n_laps = round(22000.0 / t.lap_length) if event == "endurance" else 1
    t.apex = t.findApexes(15)
    t.event = name                      # saveTrack names the file after .event
    path = saveTrack(t, OUT)
    print(f"{name:28s} {t.lap_length:7.1f} m  n_laps={t.n_laps:3d}  "
          f"R_min={1/np.abs(t.k).max():5.1f} m  ({len(x)} pts in) -> {Path(path).name}")
    return t


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("0 - Tracks")
    OUT.mkdir(parents=True, exist_ok=True)
    for f, (name, event, smooth_m, thin_m) in TRACKS.items():
        p = src / f
        if not p.exists():
            print(f"skip {f}: not found"); continue
        try:
            import_track(p, name, event, smooth_m, thin_m)
        except Exception as e:
            print(f"FAILED {f}: {e}")
