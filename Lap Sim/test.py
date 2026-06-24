from TrackMap import TrackMap
import numpy as np

tm = TrackMap('C:\\Users\\KarineFille1\\Downloads\\test2.mat')

print("== loaded channels ==")
for key in tm.data:
    print(" ", key, tm.data[key].size, "samples")

print("\n== build each event ==")
for ev in ["autocross", "endurance", "accel", "skidpad"]:
    track = tm.createTrack(event=ev, n_apex=18)
    print(f"\n{ev}")
    print("  laps        :", track.n_laps)
    print("  lap length  :", round(track.lap_length, 1), "m")
    print("  points      :", track.x.size)
    print("  ds          :", track.ds, "m")
    print("  apexes      :", len(track.apex))
    print("  min_peak    :", round(track.min_peak, 4), "1/m")