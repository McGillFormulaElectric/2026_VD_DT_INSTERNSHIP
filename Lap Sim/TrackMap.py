# Author : Anne-Sophie Nadeau
# Summary : Convert a Matlab file exported from Motec to a track map

import scipy.io as mat
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.signal import find_peaks

class TrackMap:
    # Summary : load the matlab file and keep the 4 track channels
    # Input   : filepath, path to the .mat file (str)
    # Output  : none, fills self.data with lon, lat, a_lat, speed
    def __init__(self, filepath):
        # load the matlab file
        raw = mat.loadmat(filepath, squeeze_me=True, struct_as_record=False)

        # pick out the 4 channels we need, in any order
        self.data = {}
        for name, obj in raw.items():
            if name.startswith('__'):
                continue
            value = np.ravel(obj.Value)
            name = name.lower()
            if 'lon' in name:
                self.data['lon'] = value        # GPS longitude [deg]
            elif 'lat' in name:
                self.data['lat'] = value        # GPS latitude  [deg]

    # Summary : build the track for the chosen event
    # Input   : event (autocross, endurance, accel, skidpad), n_apex, ds step [m], smooth value, endurance distance [m], accel length [m], skidpad radius [m]
    # Output  : self, with x, y, s, ds, k, n_laps, lap_length, event
    def createTrack(self, event="autocross",n_apex=10, ds=0.25, endurance_distance=22000.0, accel_length=75.0, skidpad_radius=9.125):
        event = event.lower()

        # autocross and endurance both use the real GPS track
        if event == "autocross" or event == "endurance":
            x, y, s, k = self.build_from_gps(ds)
            lap_length = s[-1]
            if event == "endurance":
                n_laps = round(endurance_distance / lap_length)
            else:
                n_laps = 1

        # accel is just a straight line, no GPS
        elif event == "accel":
            s = np.arange(0, accel_length + ds, ds)   # 0 to 75 m
            x = s.copy()
            y = np.zeros(len(s))
            k = np.zeros(len(s))                      # straight, so no curvature
            lap_length = accel_length
            n_laps = 1

        # skidpad is two circles (figure 8)
        elif event == "skidpad":
            x, y, s, k = self.figure_eight(skidpad_radius, ds)
            lap_length = s[-1]
            n_laps = 1

        # save the results so we can use track.x, track.y, track.s, track.ds, track.k
        self.x = x
        self.y = y
        self.s = s
        self.ds = ds
        self.k = k
        self.apex = self.find_apexes(n_apex)   # auto-tunes min_peak to hit n_apex
        self.n_laps = n_laps
        self.lap_length = lap_length
        self.event = event
        return self

    # Summary : turn GPS into a smooth track and find its curvature
    # Input   : ds step [m], smooth value (0 heavy, 1 light)
    # Output  : x [m], y [m], s distance [m], k curvature [1/m]
    def build_from_gps(self, ds, smooth_m=0.5):
        lat = self.data['lat']
        lon = self.data['lon']

        # Step 1: turn GPS into x, y in meters (origin = first point)
        R = 6371000.0
        x = R * np.deg2rad(lat - lat[0])
        y = R * np.cos(np.deg2rad(lat[0])) * np.deg2rad(lon - lon[0])

        # Step 2: close the data so the fit makes a loop
        x = np.append(x, x[0])
        y = np.append(y, y[0])

        # Step 3: distance travelled along the track
        step = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        cumdist = np.append(0, np.cumsum(step))

        # Step 4: drop repeated GPS points so distance always increases
        moved = np.append(True, np.diff(cumdist) > 0)
        cumdist = cumdist[moved]
        x = x[moved]
        y = y[moved]

        # Step 5: periodic smoothing spline, closes the loop with no kink
        # smooth_m is the allowed deviation in meters, higher = smoother
        s_factor = len(x) * smooth_m**2
        tck, u = splprep([x, y], u=cumdist, s=s_factor, per=1)

        # Step 6: read the smooth track on an even distance grid
        s = np.arange(0, cumdist[-1], ds)
        x, y = splev(s, tck)

        # Step 7: curvature from the spline derivatives (true path geometry)
        dx, dy = splev(s, tck, der=1)
        ddx, ddy = splev(s, tck, der=2)
        k = (dx*ddy - dy*ddx) / (dx**2 + dy**2)**1.5

        # Step 8: close the loop exactly, return to the first point
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        s = np.append(s, s[-1] + ds)
        k = np.append(k, k[0])

        return x, y, s, k
    
    # Summary : build the skidpad figure 8 from two circles
    # Input   : R circle radius [m], ds step [m]
    # Output  : x [m], y [m], s distance [m], k curvature [1/m]
    def figure_eight(self, R, ds):
        # angle steps so the spacing along the circle is ds
        angle = np.arange(0, 2*np.pi, ds / R)

        # right circle (clockwise), curvature is negative
        x1 = R + R * np.cos(np.pi - angle)
        y1 = R * np.sin(np.pi - angle)

        # left circle (anticlockwise), curvature is positive
        x2 = -R + R * np.cos(angle)
        y2 = R * np.sin(angle)

        x = np.append(x1, x2)
        y = np.append(y1, y2)
        s = np.arange(len(x)) * ds

        # constant curvature 1/R, negative on first circle, positive on second
        k = np.append(-np.ones(len(x1)), np.ones(len(x2))) / R

        return x, y, s, k
    
    # Summary : find exactly n_apex apexes by auto-tuning min_peak
    # Input   : n_apex, how many apexes the real track has
    # Output  : list of apex indices into x, y, s, k
    def find_apexes(self, n_apex):

        # all local maxima of |curvature| (every possible apex)
        peaks = find_peaks(np.abs(self.k))[0]
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