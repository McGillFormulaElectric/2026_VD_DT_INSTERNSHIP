# Author : Anne-Sophie Nadeau
# Summary : Load a Matlab file exported from Motec, build a track map, and save it to a folder

import os
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.signal import find_peaks, savgol_filter


class TrackMap:
    def __init__(self, motec, name):
        # Put the name use in motec
        self.lat = motec.getValue("lat")          # GPS latitude  [deg]
        self.lon = motec.getValue("lon")          # GPS longitude [deg]
        self.time = motec.getTime("velX")         # [s]
        self.velX = motec.getValue("velX")        # [m/s] GPS north
        self.velY = motec.getValue("velY")        # [m/s] GPS east
        self.yawRate = motec.getValue("gyrZ_HR")  # [rad/s]
        self.name = name

    def createTrack(self, event="autocross", n_apex=18, ds=0.25, endurance_distance=22000.0,
                    min_gap=7.0, smooth_m=0.2, curvature="gps", fix_length=True):
        # smooth_m: GPS spline smoothing [m]. 0.2 = close to raw GPS, 1.0 = smoother
        # curvature: "gps" = from the GPS spline, "car" = from yaw rate / speed (no GPS kinks in slow corners)
        # fix_length: GPS position jitter makes the path too long, so scale the track to the
        #             distance from GPS speed. Same shape and apexes, right length.
        x, y, s, k = self.buildFromGPS(ds, smooth_m)
        if curvature == "car":
            k = self.curvatureFromCar(s, k)
        if fix_length:
            scale = self.lengthFromSpeed() / s[-1]
            s = s * scale
            ds = ds * scale
            k = k / scale        # same total turn over a shorter track, so the lap still closes
        lap_length = s[-1]
        n_laps = round(endurance_distance / lap_length) if event == "endurance" else 1

        # save the results so we can use track.x, track.y, track.s, track.ds, track.k
        self.x, self.y, self.s, self.ds, self.k = x, y, s, ds, k
        self.apex = self.findApexes(n_apex, min_gap)   # auto-tunes min_peak to hit n_apex
        self.n_laps = n_laps
        self.lap_length = lap_length
        self.event = event
        return self

    def buildFromGPS(self, ds, smooth_m=0.2):
        # Step 1: turn GPS into x, y in meters (origin = first point)
        R = 6371000.0
        x = R * np.deg2rad(self.lat - self.lat[0])
        y = R * np.cos(np.deg2rad(self.lat[0])) * np.deg2rad(self.lon - self.lon[0])

        # Step 2: close the data so the fit makes a loop
        x = np.append(x, x[0])
        y = np.append(y, y[0])

        # Step 3: distance travelled along the track
        step = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        cumdist = np.append(0, np.cumsum(step))

        # Step 4: drop repeated GPS points so distance always increases
        moved = np.append(True, np.diff(cumdist) > 0)
        cumdist, x, y = cumdist[moved], x[moved], y[moved]

        # Step 5: periodic smoothing spline, closes the loop with no kink
        # smooth_m is the allowed deviation in meters, higher = smoother
        tck, u = splprep([x, y], u=cumdist, s=len(x) * smooth_m**2, per=1)

        # Step 6: read the smooth track on an even distance grid
        # the spline parameter is the RAW GPS distance, longer than the smooth path,
        # so measure the true length along the smooth curve first, then sample on that
        u_fine = np.linspace(0, cumdist[-1], 20 * len(x))
        x_fine, y_fine = splev(u_fine, tck)
        true_dist = np.append(0, np.cumsum(np.sqrt(np.diff(x_fine)**2 + np.diff(y_fine)**2)))
        s = np.arange(0, true_dist[-1], ds)
        u_even = np.interp(s, true_dist, u_fine)     # spline parameter at each true distance
        x, y = splev(u_even, tck)

        # Step 7: curvature from the spline derivatives (true path geometry)
        dx, dy = splev(u_even, tck, der=1)
        ddx, ddy = splev(u_even, tck, der=2)
        k = (dx*ddy - dy*ddx) / (dx**2 + dy**2)**1.5

        # Step 8: close the loop exactly, return to the first point
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        s = np.append(s, s[-1] + ds)
        k = np.append(k, k[0])

        return x, y, s, k

    def lengthFromSpeed(self):
        # lap length from GPS speed integrated over time [m]
        # speed is much less noisy than position, so this is the real distance driven
        dt = np.mean(np.diff(self.time))
        V = np.sqrt(self.velX**2 + self.velY**2)
        return np.sum(V) * dt

    def curvatureFromCar(self, s, k_gps):
        # Curvature from what the car actually drove: yaw rate / speed.
        # GPS position is noisy in slow corners (points only cm apart), so its curvature
        # spikes there. The car's own curvature has the right radius everywhere.
        dt = np.mean(np.diff(self.time))
        V = savgol_filter(np.sqrt(self.velX**2 + self.velY**2), 31, 2)
        dist = np.cumsum(V) * dt
        dist = dist - dist[0]
        k_car = savgol_filter(self.yawRate, 31, 2) / V

        # put it on the track distance grid (stretch the car distance to the track length)
        k = np.interp(s, dist * s[-1] / dist[-1], k_car)

        # scale so the lap turns exactly as much as the GPS track, so the track still closes
        # this also matches the sign convention of k_gps
        k = k * np.sum(k_gps) / np.sum(k)
        return k

    def findApexes(self, n_apex, min_gap=7.0):
        # local maxima of |curvature|, at least min_gap meters apart
        # so one corner gives one apex, not several small bumps on the same corner
        peaks = find_peaks(np.abs(self.k), distance=max(int(min_gap / self.ds), 1))[0]
        heights = np.abs(self.k)[peaks]

        # if there are not even that many, keep them all
        if len(peaks) <= n_apex:
            self.min_peak = heights.min() if len(peaks) else 0.0
            return list(np.sort(peaks))

        # keep the n_apex strongest peaks
        strongest = np.argsort(heights)[-n_apex:]
        apex = np.sort(peaks[strongest])

        # the tuned min_peak is the curvature of the weakest one we kept
        self.min_peak = np.sort(heights)[-n_apex]
        return list(apex)


# save every track field into folder/<event>.npz
def saveTrack(track, folder, name):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name + ".npz")
    np.savez(path,
             x=track.x, y=track.y, s=track.s, ds=track.ds, k=track.k,
             apex=track.apex, n_laps=track.n_laps,
             lap_length=track.lap_length, event=track.event)
    return path


# read folder/<event>.npz back into a TrackMap (no Motec file needed)
def loadTrack(folder, event):
    data = np.load(os.path.join(folder, event + ".npz"))
    track = TrackMap.__new__(TrackMap)   # skip __init__, no Motec file to read
    track.x, track.y, track.s, track.k = data["x"], data["y"], data["s"], data["k"]
    track.ds = float(data["ds"])
    track.apex = list(data["apex"])
    track.n_laps = int(data["n_laps"])
    track.lap_length = float(data["lap_length"])
    track.event = str(data["event"])
    return track