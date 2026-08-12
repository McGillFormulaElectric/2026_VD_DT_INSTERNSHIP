# Author: Anne-Sophie
# Summary: Load/save track objects for the lap sim. Two sources:
#   - load_mat_track : a MATLAB-exported track struct (x,y,s,ds,k,event,numberLaps)
#   - save_track / load_track : cache a track as .npz under Data/Track
# The Track container carries BOTH n_laps and number_laps so the event solvers
# (n_laps) and Scoring (number_laps) work off the same object.

import numpy as np
from pathlib import Path
from scipy.io import loadmat


class Track:
    """Plain container the solvers consume: s, k, ds, n_laps (+ number_laps alias)."""
    def __init__(self, s, k, ds, n_laps, event="endurance", x=None, y=None):
        self.s = np.asarray(s, float)
        self.k = np.asarray(k, float)
        self.ds = float(ds)
        self.n_laps = int(n_laps)
        self.number_laps = int(n_laps)      # alias: ends the n_laps/number_laps split
        self.event = str(event)
        self.x = None if x is None else np.asarray(x, float)
        self.y = None if y is None else np.asarray(y, float)
        self.lap_length = float(self.s[-1])

    def __repr__(self):
        return (f"Track(event={self.event!r}, lap={self.lap_length:.0f} m, "
                f"ds={self.ds:.3f}, n_laps={self.n_laps}, points={len(self.s)})")


def load_mat_track(path):
    """Load a MATLAB track struct saved as .mat (fields x,y,s,ds,k,event,numberLaps)."""
    m = loadmat(str(path))
    t = m["track"][0, 0]
    def col(name): return np.asarray(t[name]).squeeze()
    return Track(
        s=col("s"), k=col("k"),
        ds=float(np.asarray(t["ds"]).squeeze()),
        n_laps=int(np.asarray(t["numberLaps"]).squeeze()),
        event=str(np.asarray(t["event"]).squeeze()),
        x=col("x"), y=col("y"))


def save_track(track, path):
    """Cache a Track as .npz. Creates the folder if needed. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path,
             s=track.s, k=track.k, ds=track.ds,
             n_laps=track.n_laps, event=track.event,
             x=(track.x if track.x is not None else np.array([])),
             y=(track.y if track.y is not None else np.array([])))
    return path if str(path).endswith(".npz") else Path(str(path) + ".npz")


def load_track(path):
    """Load a Track cached by save_track (.npz)."""
    d = np.load(str(path), allow_pickle=True)
    x = d["x"] if d["x"].size else None
    y = d["y"] if d["y"].size else None
    return Track(s=d["s"], k=d["k"], ds=float(d["ds"]),
                 n_laps=int(d["n_laps"]), event=str(d["event"]), x=x, y=y)
    
    

# Exemple of track being saved and loaded     
import sys, pathlib
 
CORE = pathlib.Path(__file__).resolve().parent      # SaveTrack.py is in Lap Sim
sys.path.insert(0, str(CORE))

 
DATA   = CORE / "Data"
folder = DATA / "Track"
mat    = folder / "Endurance_Michigan_2024.mat"      # source .mat (put it here)
npz    = folder / "Endurance_Michigan_2024.npz"      # cached output
 
print("CORE =", CORE)                                # sanity: should end in ...\Lap Sim
 
track = load_mat_track(mat)
print("loaded :", track)
 
out = save_track(track, npz)
print("saved  :", out)
 
# verify it reads back
check = load_track(npz)
print("reload :", check, "| laps:", check.n_laps, "| number_laps:", check.number_laps)