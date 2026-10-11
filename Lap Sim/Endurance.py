import numpy as np
from SolverFunctions import corner_speed_ceiling, lap_profile, energy_and_time
from Scoring import Scoring

N_LAPS = 22   # laps, Michigan endurance

def solve(tire, car, track, v_max=None):
    """Endurance: 22 flying laps, same definition as the scoring benchmark (fastest lap x 22).
    Every lap flat out (no energy management), so total_energy_kWh is an upper bound on consumption."""
    if v_max is None:
        v_max = corner_speed_ceiling(track, car, tire)   # m/s, grip limit at each point

    # flying lap
    v, v_fwd, v_bwd = lap_profile(track, car, tire, v_max, standing_start=False)
    t, P_pack, E = energy_and_time(track, car, tire, v, v_max)
    flying_lap = t[-1] + track.ds / max(v[-1], 0.1)     # s

    total_time = flying_lap * N_LAPS                     # s
    total_E = E * N_LAPS                                 # J

    endurance_score  = Scoring.getEnduranceScore(total_time)
    efficiency_score = Scoring.getEfficiencyScore(total_time, total_E / 3.6e6, track)
    return {"event": "endurance", "s": track.s, "v": v, "v_max": v_max,
            "v_fwd": v_fwd, "v_bwd": v_bwd, "t": t, "P_pack": P_pack,
            "lap_time": flying_lap,                       # s, same key as accel and autocross
            "flying_lap_time": flying_lap,
            "n_laps": N_LAPS, "total_time": total_time,
            "energy_per_lap_kWh": E / 3.6e6,
            "total_energy_kWh": total_E / 3.6e6,
            "avg_power_kW": total_E / total_time / 1000,
            "endurance_score": endurance_score,
            "efficiency_score": efficiency_score}