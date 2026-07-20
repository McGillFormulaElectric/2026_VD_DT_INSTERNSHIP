# Author: Ludih
# Summary: Endurance solver: standing-start first lap plus flying laps to the
#          22 km target distance. 
#          Usage:
#              track = TrackMap(motec).createTrack(event="endurance", ds=0.5)
#              result = solveEndurance.solve(track, car, tire)

import numpy as np
from Solver_functions import corner_speed_ceiling, lap_profile, energy_and_time


def solve(track, car, tire, v_max=None):
    """Endurance: standing-start first lap plus flying laps to the target
    distance. Totals assume every lap is driven flat out (no energy management),
    so total_energy_kWh is an upper bound on consumption."""
    if v_max is None:
        v_max = corner_speed_ceiling(track, car, tire)

    v1, _, _ = lap_profile(track, car, tire, v_max, standing_start=True)
    t1, P1, E1 = energy_and_time(track, car, tire, v1)
    first_lap = t1[-1] + track.ds / max(v1[-1], 0.1)

    v, v_fwd, v_bwd = lap_profile(track, car, tire, v_max, standing_start=False)
    t, P_pack, E = energy_and_time(track, car, tire, v)
    flying_lap = t[-1] + track.ds / max(v[-1], 0.1)

    n_laps = track.n_laps
    total_time = first_lap + flying_lap * (n_laps - 1)
    total_E = E1 + E * (n_laps - 1)
    return {"event": "endurance", "s": track.s, "v": v, "v_max": v_max,
            "v_fwd": v_fwd, "v_bwd": v_bwd, "t": t, "P_pack": P_pack,
            "first_lap_time": first_lap, "flying_lap_time": flying_lap,
            "n_laps": n_laps, "total_time": total_time,
            "energy_per_lap_kWh": E / 3.6e6,
            "total_energy_kWh": total_E / 3.6e6,
            "avg_power_kW": total_E / total_time / 1000}