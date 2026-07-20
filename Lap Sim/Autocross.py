# Author: Ludih
# Summary: Autocross solver — one lap from a standing start.
#          Usage:
#              track = TrackMap(motec).createTrack(event="autocross", ds=0.5)
#              result = solveAutocross.solve(track, car, tire)

import numpy as np
from Solver_functions import corner_speed_ceiling, lap_profile, energy_and_time


def solve(track, car, tire, v_max=None):
    """Autocross: single lap from a standing start."""
    if v_max is None:
        v_max = corner_speed_ceiling(track, car, tire)
    v, v_fwd, v_bwd = lap_profile(track, car, tire, v_max, standing_start=True)
    t, P_pack, energy_J = energy_and_time(track, car, tire, v)
    lap_time = t[-1] + track.ds / max(v[-1], 0.1)
    return {"event": "autocross", "s": track.s, "v": v, "v_max": v_max,
            "v_fwd": v_fwd, "v_bwd": v_bwd, "t": t, "P_pack": P_pack,
            "lap_time": lap_time, "energy_kWh": energy_J / 3.6e6,
            "avg_power_kW": energy_J / lap_time / 1000}