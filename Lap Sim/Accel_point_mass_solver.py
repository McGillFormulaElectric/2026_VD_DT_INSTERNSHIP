# Author: Ludih
# Summary: Solver for the acceleration event. Simulates the car accelerating
#          from standstill over 75m, stepping through time to compute speed,
#          distance, and elapsed time at each step.

from dataclasses import dataclass
import carProperties
import physics as ph

@dataclass
class AccelSolver():
    track_length:  float = 75.0  # m  length of acceleration event
    run_up:        float = 0.3  # m  distance before start line where car starts moving
    v0:            float = 0.0  # m/s  initial speed
    dt:            float = 0.01  # s  time step for simulation
    
    def simulate(self, car):
        v_max = ph.rpm_to_speed(car,car.motor_rpm_max) 
        # Initialize variables
        v = self.v0
        s = -self.run_up  # Start before the start line
        t = 0.0

        # Lists to store results for plotting or analysis
        speeds_list = []
        distance_list = []
        time_list = []
        Ax_list = []
        Fx_list = []
        propulsive_force_list = []
        drag_list = []
        downforce_list = []
        rolling_resistance_list = []
        propulsion_power_list = []
        motor_power_list = []
        rpm_list = []
        traction_limit_long_list = []
        torque_limited_list = []
        power_limited_list = []



        # Simulation loop
        while s < self.track_length:
            # Calculate normal force (N)
            N = car.mass_total * car.g + ph.downforce(car, v)

            # Calculate propulsive force (N)
            Fx = ph.propulsive_force(car, v, N) - ph.drag(car, v) - ph.rolling_resistance(car, N)

            # Calculate acceleration (m/s²)
            Ax = Fx / car.mass_total

            # Update speed and position using simple Euler integration
            v = min(v + Ax * self.dt, v_max)
            s += v * self.dt

            if s >= 0:  # Only count time after crossing the start line
                t += self.dt

            # Store results
            speeds_list.append(v)
            distance_list.append(s)
            time_list.append(t)
            Ax_list.append(Ax)
            Fx_list.append(Fx)
            drag_list.append(ph.drag(car, v))
            downforce_list.append(ph.downforce(car, v))
            rolling_resistance_list.append(ph.rolling_resistance(car, N))
            propulsion_power_list.append(Fx * v / 1000)  # kW
            motor_power_list.append(car.peak_motor_torque * ph.speed_to_rpm(car, v) * 2 * 3.14159 / 60 / 1000)  
            rpm_list.append(ph.speed_to_rpm(car, v))
            traction_limit_long_list.append(ph.traction_limit_long(car, N))
            torque_limited_list.append(ph.F_torque_limited(car))
            power_limited_list.append(ph.F_power_limited(car, v))

        return {
            'speed':          speeds_list,
            'distance':       distance_list,
            'time':           time_list,
            'Ax':             Ax_list,
            'Fx':             Fx_list,
            'drag':           drag_list,
            'downforce':      downforce_list,
            'rr':             rolling_resistance_list,
            'propulsion_power': propulsion_power_list,
            'motor_power':    motor_power_list,
            'rpm':            rpm_list,
            'traction_limit': traction_limit_long_list,
            'torque_limit':   torque_limited_list,
            'power_limit':    power_limited_list,
        }


