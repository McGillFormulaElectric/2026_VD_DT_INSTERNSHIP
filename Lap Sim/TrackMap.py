import os
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.signal import find_peaks, savgol_filter

class TrackMap:
    def __init__(self, motec, name):
        self.lat = motec.getValue("lat")          # deg
        self.lon = motec.getValue("lon")          # deg
        self.time = motec.getTime("velX")         # s
        self.velX = motec.getValue("velX")        # m/s, GPS north
        self.velY = motec.getValue("velY")        # m/s, GPS east
        self.name = name

    def createTrack(self, event="autocross", n_apex=18, ds=0.25, endurance_distance=22000.0,
                    min_gap=7.0, smooth_m=0.2, curvature="gps", fix_length=True):
        # smooth_m [m]: GPS spline smoothing, 0.2 = close to raw, 1.0 = smoother
        # curvature: "gps" from the spline, "car" from GPS course rate / speed (true path, no position noise, no sideslip error)
        # fix_length: GPS jitter makes the path too long, scale it to the distance from GPS speed
        x, y, s, k = self.buildFromGPS(ds, smooth_m)
        if curvature == "car":
            k = self.curvatureFromCar(s, k)
        if fix_length:
            scale = self.lengthFromSpeed() / s[-1]
            s = s * scale
            ds = ds * scale
            k = k / scale        # same total turn over the new length, so the lap still closes
        lap_length = s[-1]                                                        # m
        n_laps = round(endurance_distance / lap_length) if event == "endurance" else 1

        self.x, self.y, self.s, self.ds, self.k = x, y, s, ds, k   # m, m, m, m, 1/m
        self.apex = self.findApexes(n_apex, min_gap)
        self.n_laps = n_laps
        self.lap_length = lap_length
        self.event = event
        return self

    def buildFromGPS(self, ds, smooth_m=0.2):
        # GPS to x, y [m], origin at the first point
        R = 6371000.0   # m, earth radius
        x = R * np.deg2rad(self.lat - self.lat[0])
        y = R * np.cos(np.deg2rad(self.lat[0])) * np.deg2rad(self.lon - self.lon[0])

        # close the loop and drop repeated points so distance always increases
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        cumdist = np.append(0, np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)))   # m
        moved = np.append(True, np.diff(cumdist) > 0)
        cumdist, x, y = cumdist[moved], x[moved], y[moved]

        # periodic smoothing spline, closes the loop with no kink
        tck, u = splprep([x, y], u=cumdist, s=len(x) * smooth_m**2, per=1)

        # sample on an even grid of TRUE distance along the smooth curve
        # (the spline parameter is raw GPS distance, which is longer)
        u_fine = np.linspace(0, cumdist[-1], 20 * len(x))
        x_fine, y_fine = splev(u_fine, tck)
        true_dist = np.append(0, np.cumsum(np.sqrt(np.diff(x_fine)**2 + np.diff(y_fine)**2)))   # m
        s = np.arange(0, true_dist[-1], ds)                                                     # m
        u_even = np.interp(s, true_dist, u_fine)
        x, y = splev(u_even, tck)

        # curvature from the spline derivatives [1/m]
        dx, dy = splev(u_even, tck, der=1)
        ddx, ddy = splev(u_even, tck, der=2)
        k = (dx*ddy - dy*ddx) / (dx**2 + dy**2)**1.5

        # close the loop exactly
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        s = np.append(s, s[-1] + ds)
        k = np.append(k, k[0])
        return x, y, s, k

    def lengthFromSpeed(self):
        # lap length [m] from GPS speed x time, speed is much less noisy than position
        dt = np.mean(np.diff(self.time))                 # s
        V = np.sqrt(self.velX**2 + self.velY**2)         # m/s
        return np.sum(V) * dt

    def curvatureFromCar(self, s, k_gps):
        # curvature [1/m] from the GPS course rate / speed
        # course = direction of the velocity vector, so this is the real path the car drove:
        # no position noise like the spline, no sideslip error like yaw rate
        dt = np.mean(np.diff(self.time))                                         # s
        V = savgol_filter(np.sqrt(self.velX**2 + self.velY**2), 31, 2)           # m/s, about 0.3 s window at 100 Hz
        course = np.unwrap(np.arctan2(self.velY, self.velX))                     # rad, direction of travel
        course_rate = savgol_filter(course, 31, 2, deriv=1, delta=dt)            # rad/s
        k_car = course_rate / np.maximum(V, 1.0)                                 # 1/m, 1 m/s floor, no blow up when stopped
        dist = np.cumsum(V) * dt                                                 # m
        dist = dist - dist[0]

        # stretch the car distance onto the track grid
        k = np.interp(s, dist * s[-1] / dist[-1], k_car)

        # match the total turn and sign of the GPS track, so the lap still closes
        k = k * np.sum(k_gps) / np.sum(k)
        return k

    def findApexes(self, n_apex, min_gap=7.0):
        # peaks of |k| at least min_gap [m] apart, so one corner gives one apex
        peaks = find_peaks(np.abs(self.k), distance=max(int(min_gap / self.ds), 1))[0]
        heights = np.abs(self.k)[peaks]   # 1/m

        # fewer peaks than asked, keep them all
        if len(peaks) <= n_apex:
            self.min_peak = heights.min() if len(peaks) else 0.0
            return list(np.sort(peaks))

        # keep the n_apex strongest, min_peak = curvature of the weakest kept
        strongest = np.argsort(heights)[-n_apex:]
        self.min_peak = np.sort(heights)[-n_apex]
        return list(np.sort(peaks[strongest]))


def saveTrack(track, folder, name):
    # save the track to folder/<name>.npz
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name + ".npz")
    np.savez(path,
             x=track.x, y=track.y, s=track.s, ds=track.ds, k=track.k,
             apex=track.apex, n_laps=track.n_laps,
             lap_length=track.lap_length, event=track.event)
    return path


def loadTrack(folder, name):
    # read folder/<name>.npz back into a TrackMap, no Motec file needed
    data = np.load(os.path.join(folder, name + ".npz"))
    track = TrackMap.__new__(TrackMap)   # skip __init__, nothing to read from Motec
    track.x, track.y, track.s, track.k = data["x"], data["y"], data["s"], data["k"]
    track.ds = float(data["ds"])
    track.apex = list(data["apex"])
    track.n_laps = int(data["n_laps"])
    track.lap_length = float(data["lap_length"])
    track.event = str(data["event"])
    return track