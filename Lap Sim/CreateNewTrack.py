import sys
from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Path to the Lap Sim folder
LAPSIM_ROOT = Path(__file__).resolve().parents[0]   # 0 = script is in Lap Sim, 3 = script is in Lap Sim/3. Analysis/x/2027
sys.path.insert(0, str(LAPSIM_ROOT))

from MotecData import MotecData
from TrackMap import TrackMap, saveTrack

NAME = "TorontoShootout2026"   # track name, also the saved file name
N_APEX = 35              # number of apexes to find

# ── build the track from the Motec log and save it ────────────
motec = MotecData(LAPSIM_ROOT / "1. Data" / "Motec" / "TorontoShootout2026.mat")
track = TrackMap(motec, name=NAME).createTrack(event="autocross", n_apex=N_APEX, curvature="car")   # curvature from yaw rate / speed
saveTrack(track, LAPSIM_ROOT / "1. Data" / "Track", name=NAME)

print(f"event:      {track.event}")
print(f"lap length: {track.lap_length:.1f} m")
print(f"laps:       {track.n_laps}")
print(f"apexes:     {len(track.apex)}")

# ── track with apexes marked ──────────────────────────────────
plt.plot(track.x, track.y)                                 # m
plt.plot(track.x[track.apex], track.y[track.apex], "ro")   # apexes
plt.axis("equal")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.grid(True)
plt.show()