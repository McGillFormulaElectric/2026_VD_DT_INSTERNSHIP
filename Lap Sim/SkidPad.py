# Author: Anne-So
# Summary : Solves the single fastest steady-state speed the car can hold at the 9.125 m radius, then lap time is circumference over speed.

from dataclasses import dataclass
import numpy as np
import Limits as lim

@dataclass
class SkidPadSolver:
    path_radius: float = 9.125     # m, midway between the 15.25 m and 21.25 m cone circles

    def simulate(self, tire, car):
        """Returns the FSAE skidpad result: lap time, corner speed, and the four corner loads.

        tire : Tire object
        car  : CarProperties
        """
        out = lim.corner_speed(tire, car, self.path_radius)
        out["time"] = 2 * np.pi * self.path_radius / out["v"]
        return out