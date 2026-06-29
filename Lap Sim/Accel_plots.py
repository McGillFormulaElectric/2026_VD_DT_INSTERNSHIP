# Author: Ludih
# Summary: Plotting script for the acceleration event solver.
#          Import results from AccelSolver and plot any channel against any other.

import matplotlib.pyplot as plt
import carProperties
from Accel_point_mass_solver import AccelSolver

# ── Run Simulation ─────────────────────────────────────────────────────────────
car = carProperties.MFE27()
results = AccelSolver().simulate(car)

# ── Helper ─────────────────────────────────────────────────────────────────────
def plot(x_key, y_key, xlabel=None, ylabel=None, title=None):
    """Plot any two channels from results dict against each other."""
    plt.figure()
    plt.plot(results[x_key], results[y_key])
    plt.xlabel(xlabel or x_key)
    plt.ylabel(ylabel or y_key)
    plt.title(title or f"{y_key} vs {x_key}")
    plt.grid(True)
    plt.tight_layout()

def plot_multi(x_key, y_keys, labels=None, xlabel=None, ylabel=None, title=None):
    """Plot multiple channels on the same axes."""
    plt.figure()
    for i, y_key in enumerate(y_keys):
        label = labels[i] if labels else y_key
        plt.plot(results[x_key], results[y_key], label=label)
    plt.xlabel(xlabel or x_key)
    plt.ylabel(ylabel or y_key)
    plt.title(title or "")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

# ── Plots ──────────────────────────────────────────────────────────────────────
print(results['time'][-1])

plot("time", "distance", "Time (s)", "Distance (m)", "Distance vs Time")

# Speed vs Distance
plot('distance', 'speed', 'Distance (m)', 'Speed (m/s)', 'Speed vs Distance')

# Speed vs Time
plot('time', 'speed', 'Time (s)', 'Speed (m/s)', 'Speed vs Time')

# Acceleration vs Distance
plot('distance', 'Ax', 'Distance (m)', 'Acceleration (m/s²)', 'Acceleration vs Distance')

# Downforce vs Distance
plot('distance', 'downforce', 'Distance (m)', 'Downforce (N)', 'Downforce vs Distance')

# Propulsion Power vs Distance
plot('distance', 'propulsion_power', 'Distance (m)', 'Propulsion Power (kW)', 'Propulsion Power vs Distance')

# RPM vs Distance
plot('distance', 'rpm', 'Distance (m)', 'Motor RPM', 'RPM vs Distance')

# Motor Power vs Distance
plot('distance', 'motor_power', 'Distance (m)', 'Motor Power (kW)', 'Motor Power vs Distance')

plt.show()