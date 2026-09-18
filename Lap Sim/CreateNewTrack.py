import sys, pathlib
import matplotlib.pyplot as plt

CORE = pathlib.Path(__file__).resolve().parents[0]   # 2027 -> PowerTrain -> Analysis -> Lap Sim
sys.path.insert(0, str(CORE))

from MotecData import MotecData
from TrackMap import TrackMap, saveTrack

DATA = CORE / "Data"

"""
# build a track from a Motec log and save it
motec = MotecData(DATA / "Motec"/ "MotecData.mat")
track = TrackMap(motec).createTrack(event="autocross", n_apex=15)
saveTrack(track, DATA / "Track")

print("event      :", track.event)
print("lap length :", round(track.lap_length, 1), "m")
print("n laps     :", track.n_laps)
print("n apex     :", len(track.apex))

# quick look: track with apexes marked
plt.plot(track.x, track.y, "-")
plt.plot(track.x[track.apex], track.y[track.apex], "ro")
plt.axis("equal")
plt.show()
"""

track = TrackMap()