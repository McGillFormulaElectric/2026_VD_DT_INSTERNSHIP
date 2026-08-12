# Author: Anne-So
# Summary : Solves the single fastest steady-state speed the car can hold at the 9.125 m radius, then lap time is circumference over speed.

from dataclasses import dataclass
import numpy as np
import Limits as lim
from Scoring import Scoring

@dataclass
class SkidPadSolver:
    path_radius: float = 9.125     # m, midway between the 15.25 m and 21.25 m cone circles

    def simulate(self, tire, car):
        """Returns the FSAE skidpad result: lap time, corner speed, and the four corner loads.

        tire : Tire object
        car  : CarProperties
        """
        r = lim.corner_speed(tire, car, self.path_radius)
        v = r["v"]
        lap_time = 2 * np.pi * self.path_radius / v
        energy_J = r["P_pack"] * lap_time

        return {"event": "skidpad",
                "distance": 2 * np.pi * self.path_radius,   # one lap circumference [m]
                "v": v, "v_max": v,                          # at the limit, v == v_max
                "AyG": r["AyG"],
                "P_pack": r["P_pack"],
                "time": lap_time, "lap_time": lap_time, "total_time": lap_time,
                "energy_kWh": energy_J / 3.6e6,
                "FzFL": r["FzFL"], "FzFR": r["FzFR"],
                "FzRL": r["FzRL"], "FzRR": r["FzRR"],
                "comp_point": Scoring.getSkidpadScore(lap_time)}