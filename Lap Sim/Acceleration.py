# Author: Ludih
# Summary: Solver for the acceleration event. Simulates the car accelerating
#          from standstill over 75m, stepping through time to compute speed,
#          distance, and elapsed time at each step.


from dataclasses import dataclass
import numpy as np
import Physics as ph
import Limits as lim


@dataclass
class AccelSolver:
    track_length: float = 75.0     # m  length of acceleration event
    run_up:       float = 0.3      # m  distance before start line where car starts moving
    v0:           float = 0.0      # m/s  initial speed

    def simulate(self, tire, car):
        
        def step_size(k):
            if k < 1000: return 1e-4
            if k < 2000: return 1e-3
            return 1e-2

        # Initialize variables
        v = self.v0
        s = -self.run_up # Start before the start line
        Ax_prev = 0.0
        k = 0

        # Lists to store results for plotting or analysis
        log = {key: [] for key in (
            "dist", "ds", "dt", "v", "Ax",
            "FzF", "FzR", "Downforce", "Drag", "Frolling", "Fx_penalties",
            "Fx_tractive", "Fx_net",
            "P_F", "P_R", "EfficiencyF", "EfficiencyR", "PackPower",
            "rpm", "TmotorF", "TmotorR",
        )}

        # Simulation loop
        while s < self.track_length:
            if k > 200_000:
                raise RuntimeError(f"Stalled at {s:.2f} m.")
            ds = step_size(k)

            # Aero
            Downforce = ph.downforce(car, v)
            Drag = ph.drag(car, v)

            # Calculate normal force (N)
            FzF = max(ph.FL_Fz(car, Downforce, 0.0, Ax_prev), 0.0)
            FzR = max(ph.RL_Fz(car, Downforce, 0.0, Ax_prev), 0.0)

            # Calculate penalties (N)
            Frolling = ph.rolling_resistance(car, v, 2.0*FzF + 2.0*FzR)
            Fx_penalties = Drag + Frolling

            # Calculate propulsive force (N)
            t = lim.tractive_force_4wd(tire, car, FzF, FzR, v)
            v = t["v"]
            Fx_net = t["Fx_total"] - Fx_penalties
            
            # Calculate acceleration (m/s²)
            Ax = Fx_net / car.mass_total

            # Update speed and position
            v_next = np.sqrt(max(v**2 + 2.0*Ax*ds, 0.0))
            v_bar = 0.5*(v + v_next)
            if v_bar > 1e-6:
                dt = ds / v_bar
            elif Ax > 0:
                dt = np.sqrt(2.0*ds/Ax)
            else:
                raise RuntimeError("Car is stationary and Ax <= 0. "
                                   "Check gear_ratio, tire_radius, torque_front_bias, torque curve.")

            # Store results
            for key, val in (
                ("dist", s), ("ds", ds), ("dt", dt), ("v", v), ("Ax", Ax),
                ("FzF", FzF), ("FzR", FzR),
                ("Downforce", Downforce), ("Drag", Drag), ("Frolling", Frolling),
                ("Fx_penalties", Fx_penalties),
                ("Fx_tractive", t["Fx_total"]), ("Fx_net", Fx_net),
                ("P_F", t["P_F"]), ("P_R", t["P_R"]),
                ("EfficiencyF", t["EfficiencyF"]), ("EfficiencyR", t["EfficiencyR"]),
                ("PackPower", t["PackPower"]), ("rpm", t["rpm"]),
                ("TmotorF", t["TmotorF"]), ("TmotorR", t["TmotorR"]),
            ):
                log[key].append(val)

            Ax_prev = Ax
            v = v_next
            s += ds
            k += 1

        out = {key: np.asarray(v, dtype=float) for key, v in log.items()}
        out["time"] = np.concatenate(([0.0], np.cumsum(out["dt"])[:-1]))
        out["EnergyPack"] = float(np.sum(out["PackPower"] * out["dt"]))
        out["time_total"] = float(np.sum(out["dt"]))
        out["v_final"] = float(v)
        out["n_steps"] = k
        return out