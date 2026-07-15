import numpy as np
import Physics as ph

class Acceleration:

    def acceleration(self, tire, car, distance=75.0):
        """Straight line acceleration event, flat out from standstill.
        Returns a dict of equal length arrays, plus scalar summary values.
        """
        
        def step_size(k):
            if k < 1000:
                return 1e-4
            if k < 2000:
                return 1e-3
            return 1e-2
    
        Vx = 0.0
        Ax_prev = 0.0
        dist = 0.0
        k = 0
    
        log = {key: [] for key in (
            "dist", "ds", "dt", "Vx", "Ax",
            "FzF", "FzR", "Downforce", "Drag", "Frolling", "Fx_penalties",
            "Fx_tractive", "Fx_net",
            "P_F", "P_R", "EfficiencyF", "EfficiencyR", "PackPower",
            "rpm", "TmotorF", "TmotorR",
        )}
    
        while dist < distance:
            if k > 200_000:
                raise RuntimeError(f"Stalled at {dist:.2f} m. Ax went to zero, or ds is too small.")
            ds = step_size(k)
            
            # Aero
            Downforce = ph.downforce(car, Vx)
            Drag = ph.drag(car, Vx)

            #Vertical loads
            FzF = ph.FL_Fz(car, Downforce, 0.0, Ax_prev)          # one front wheel [N]
            FzR = ph.RL_Fz(car, Downforce, 0.0, Ax_prev)          # one rear wheel  [N]
            FzF = max(FzF, 0.0)                                   # a lifted wheel has no grip
            FzR = max(FzR, 0.0)
    
            # Rolling resistance and drag penalties
            Frolling = ph.rolling_resistance(car, Vx, 2.0 * FzF + 2.0 * FzR)
            Fx_penalties = Drag + Frolling
    
            # Tractive force and resulting acceleration
            t = self.tractive_force_4wd(tire, car, FzF, FzR, Vx)
            Vx = t["Vx"]
            Fx_net = t["Fx_total"] - Fx_penalties
            Ax = Fx_net / car.mass_total
            
            # Next speed and time step
            Vx_next = np.sqrt(max(Vx**2 + 2.0 * Ax * ds, 0.0))
            v_bar = 0.5 * (Vx + Vx_next)
            if v_bar > 1e-6:
                dt = ds / v_bar
            elif Ax > 0:
                dt = np.sqrt(2.0 * ds / Ax)
            else:
                raise RuntimeError("Car is stationary and Ax <= 0. It will never move. "
                                "Check gear_ratio, tire_radius, torque_front_bias, and the torque curve.")
    
            for key, val in (
                ("dist", dist), ("ds", ds), ("dt", dt), ("Vx", Vx), ("Ax", Ax),
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
            Vx = Vx_next
            dist += ds
            k += 1
    
        out = {key: np.asarray(v, dtype=float) for key, v in log.items()}
        out["time"] = np.concatenate(([0.0], np.cumsum(out["dt"])[:-1]))
        out["EnergyPack"] = float(np.sum(out["PackPower"] * out["dt"]))   # [J]
        out["time_total"] = float(np.sum(out["dt"]))    # [s]  75 m time
        out["Vx_final"] = float(Vx)                     # [m/s] trap speed
        out["n_steps"] = k
    
        return out