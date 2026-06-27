import matplotlib.pyplot as plt
from MotecData import MotecData
from TrackMap import TrackMap

# load the motec file once, reuse it for every event
motec = MotecData('test1.mat')

# Graph 1: autocross track map
track = TrackMap(motec).createTrack(event="autocross", n_apex=6)
plt.figure(figsize=(6,6))
plt.plot(track.x, track.y, 'k-', lw=1)
plt.plot(track.x[track.apex], track.y[track.apex], 'r.', ms=8)
plt.axis('equal')
plt.title("autocross")
plt.xlabel("x [m]"); plt.ylabel("y [m]")
plt.grid(alpha=0.3)

# Graph 2: endurance track map
track = TrackMap(motec).createTrack(event="endurance", n_apex=6)
plt.figure(figsize=(6,6))
plt.plot(track.x, track.y, 'k-', lw=1)
plt.plot(track.x[track.apex], track.y[track.apex], 'r.', ms=8)
plt.axis('equal')
plt.title("endurance")
plt.xlabel("x [m]"); plt.ylabel("y [m]")
plt.grid(alpha=0.3)

# Graph 3: accel track map
track = TrackMap(motec).createTrack(event="accel", n_apex=6)
plt.figure(figsize=(6,6))
plt.plot(track.x, track.y, 'k-', lw=1)
plt.axis('equal')
plt.title("accel")
plt.xlabel("x [m]"); plt.ylabel("y [m]")
plt.grid(alpha=0.3)

# Graph 4: skidpad track map
track = TrackMap(motec).createTrack(event="skidpad", n_apex=6)
plt.figure(figsize=(6,6))
plt.plot(track.x, track.y, 'k-', lw=1)
plt.axis('equal')
plt.title("skidpad")
plt.xlabel("x [m]"); plt.ylabel("y [m]")
plt.grid(alpha=0.3)

# Graph 5: curvature vs distance (autocross)
track = TrackMap(motec).createTrack(event="autocross", n_apex=6)
plt.figure(figsize=(8,5))
plt.plot(track.s, track.k, 'k-', lw=1)
plt.plot(track.s[track.apex], track.k[track.apex], 'r.', ms=8)
plt.title("curvature")
plt.xlabel("Distance [m]"); plt.ylabel("Curvature [1/m]")
plt.grid(alpha=0.3)

plt.show()