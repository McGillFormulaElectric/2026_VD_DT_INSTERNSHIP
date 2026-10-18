from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# DR Q1: effective rolling radius Re.
# v = omega * Re  ->  plot GPS speed against wheel speed, the slope of the line through zero is Re.

# Path to the Lap Sim folder
LAPSIM_ROOT = Path(__file__).resolve().parents[1]   # this file is in "2. Correlation"
DATA_DIR = LAPSIM_ROOT / "1. Data" / "Motec"

# ── logs and filters, change these ────────────────────────────
LOGS = ["accel.mat", "autocross.mat", "TorontoShootout2026.mat"]   # brake.mat left out: GPS glitch
GEAR_RATIO = 13.39   # -
V_MIN = 6.0          # m/s, GPS speed is noisy below this
AY_MAX = 2.5         # m/s^2, straights only
CUTOFF = 2.0         # Hz, low pass on GPS speed and accY
R_SIM = 0.2032        # m, tire_radius in CarProperties
WHEELS = ["FL", "FR", "RL", "RR"]


def lowpass(x, fs=100.0):
    b, a = butter(2, CUTOFF / (fs / 2))
    return filtfilt(b, a, x)


omega, v = [], []
for log in LOGS:
    mat = sio.loadmat(DATA_DIR / log, squeeze_me=True, struct_as_record=False)
    speed = lowpass(np.hypot(mat["velX"].Value, mat["velY"].Value))     # m/s, GPS speed (velX, velY are East, North)
    rpm = np.array([mat["SpeedActual" + c].Value for c in WHEELS])    # rpm, motor, rows FL FR RL RR
    w = np.median(rpm, axis=0) / GEAR_RATIO * 2 * np.pi / 60          # rad/s, wheel, median of the 4 ignores one spinning wheel
    ay = lowpass(mat["accY"].Value)                                   # m/s^2, only |ay| is used so the IMU sign doesn't matter
    keep = (speed > V_MIN) & (np.abs(ay) < AY_MAX)
    omega.append(w[keep])
    v.append(speed[keep])
omega, v = np.concatenate(omega), np.concatenate(v)

Re = np.sum(v * omega) / np.sum(omega ** 2)                           # m, least squares slope through zero
r2 = 1 - np.sum((v - Re * omega) ** 2) / np.sum((v - v.mean()) ** 2)
print(f"Re = {Re:.4f} m  (R^2 {r2:.4f}, {len(v)} samples)   sim uses {R_SIM} m ({(R_SIM / Re - 1) * 100:+.1f} %)")

# plot
x = np.array([0, omega.max() * 1.05])
plt.figure(figsize=(8, 6))
plt.plot(omega, v, ".", ms=3, alpha=0.4, label="data, straights")
plt.plot(x, Re * x, "r", lw=2, label=f"fit  Re = {Re * 1000:.1f} mm")
plt.plot(x, R_SIM * x, "k--", label=f"sim  R = {R_SIM * 1000:.0f} mm")
plt.xlabel("Wheel speed (rad/s)")
plt.ylabel("GPS speed (m/s)")
plt.title("Effective rolling radius", fontweight="bold")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()