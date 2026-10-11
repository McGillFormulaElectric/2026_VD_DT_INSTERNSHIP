# Tire model for the lap sim.
# Loads the MF6.1 fit made in MATLAB from TTC data. The sim only needs peak force
# vs vertical load, so that is computed once into a table and looked up while running.
# Units: Fz in N, slip angle and slip ratio in rad, camber in deg.
# Equations: TNO MF-Tyre 6.2 manual, 1.2 (Fx) and 1.3 (Fy), steady state, pure slip.

import numpy as np
import scipy.io


class Tire:
    def __init__(self, param_path, fzref_path, muxScale=0.55, muyScale=0.6162,
                 Fz_max=3500.0, n_grid=80, build_camber=0.0):
        # param_path: MF61_Parameters.mat, Fz0, coefficients (p) and scaling factors (lamda)
        # fzref_path: FZ_Reference.mat, loads the TTC fit was done at
        # muxScale, muyScale: grip correction from correlation, the raw TTC fit is too grippy (sandpaper belt)
        # Fz_max [N], n_grid: the grip table covers 0 to Fz_max in n_grid points
        # build_camber [deg]: static camber, the table is built at this value
        mat = scipy.io.loadmat(param_path, struct_as_record=False, squeeze_me=True)
        p = mat["p"]
        lam = mat["lamda"]
        self.FZref = np.asarray(scipy.io.loadmat(fzref_path, squeeze_me=True)["FZref"], float)   # N
        self.Fz0 = float(mat["Fz0"])   # N, reference load of the fit

        # MATLAB names to manual names (p.Cx1 -> PCX1), missing lambdas = 1.0 = no scaling
        self.coefficient = {
            # longitudinal
            "PCX1": p.Cx1, "PDX1": p.Dx1, "PDX2": p.Dx2, "PDX3": p.Dx3,
            "PEX1": p.Ex1, "PEX2": p.Ex2, "PEX3": p.Ex3, "PEX4": p.Ex4,
            "PKX1": p.Kx1, "PKX2": p.Kx2, "PKX3": p.Kx3,
            "PVX1": p.Vx1, "PVX2": p.Vx2,
            "LMUX": lam.mu_x, "LKX": lam.Kx_kappa, "LVX": 1.0,
            # lateral
            "PCY1": p.Cy1, "PDY1": p.Dy1, "PDY2": p.Dy2, "PDY3": p.Dy3,
            "PEY1": p.Ey1, "PEY2": p.Ey2, "PEY3": p.Ey3, "PEY4": p.Ey4, "PEY5": p.Ey5,
            "PKY1": p.Ky1, "PKY2": p.Ky2, "PKY3": p.Ky3, "PKY4": p.Ky4,
            "PKY5": p.Ky5, "PKY6": p.Ky6, "PKY7": p.Ky7,
            "PHY1": p.Hy1, "PHY2": p.Hy2,
            "PVY1": p.Vy1, "PVY2": p.Vy2, "PVY3": p.Vy3, "PVY4": p.Vy4,
            "LMUY": lam.mu_y, "LKY": 1.0, "LKYC": 1.0, "LHY": 1.0, "LVY": 1.0,
        }

        self._muxScale = muxScale
        self._muyScale = muyScale
        self.Fz_max = Fz_max
        self.n_grid = n_grid
        self.build_camber = build_camber
        self.build_table()

    # changing muxScale or muyScale rebuilds the table, so peak() is up to date right after
    @property
    def muxScale(self):
        return self._muxScale

    @muxScale.setter
    def muxScale(self, value):
        self._muxScale = value
        self.build_table()

    @property
    def muyScale(self):
        return self._muyScale

    @muyScale.setter
    def muyScale(self, value):
        self._muyScale = value
        self.build_table()

    # Magic Formula: F = D sin(C atan(B x - E (B x - atan(B x)))) + SV
    #   x = slip, D = peak, C = shape, B = stiffness, E = curvature, SV/SH = offsets
    #   dfz = (Fz - Fz0) / Fz0, grip per N drops as load goes up

    def Fx(self, Fz, kappa, camber=0):
        # longitudinal force [N], kappa + drive, - brake
        p = self.coefficient
        gamma = np.radians(camber)
        dfz = (Fz - self.Fz0) / self.Fz0

        Cx = p["PCX1"]
        mux = (p["PDX1"] + p["PDX2"]*dfz) * (1 - p["PDX3"]*gamma**2) * p["LMUX"] * self.muxScale   # friction coefficient
        Dx = mux * Fz
        Ex = np.minimum((p["PEX1"] + p["PEX2"]*dfz + p["PEX3"]*dfz**2) * (1 - p["PEX4"]*np.sign(kappa)), 1.0)
        Kx = Fz * (p["PKX1"] + p["PKX2"]*dfz) * np.exp(p["PKX3"]*dfz) * p["LKX"]   # slip stiffness
        Bx = Kx / (Cx*Dx + 1e-6)   # 1e-6 avoids divide by zero at Fz = 0
        SVx = Fz * (p["PVX1"] + p["PVX2"]*dfz) * p["LVX"] * p["LMUX"] * self.muxScale

        return Dx * np.sin(Cx*np.arctan(Bx*kappa - Ex*(Bx*kappa - np.arctan(Bx*kappa)))) + SVx

    def Fy(self, Fz, alpha, camber=0):
        # lateral force [N], alpha in rad
        p = self.coefficient
        gamma = np.radians(camber)
        dfz = (Fz - self.Fz0) / self.Fz0

        Cy = p["PCY1"]
        muy = (p["PDY1"] + p["PDY2"]*dfz) * (1 - p["PDY3"]*gamma**2) * p["LMUY"] * self.muyScale   # friction coefficient
        Dy = muy * Fz
        Ky = p["PKY1"]*self.Fz0 * (1 - p["PKY3"]*abs(gamma)) * np.sin(p["PKY4"]*np.arctan((Fz/self.Fz0) / (p["PKY2"] + p["PKY5"]*gamma**2))) * p["LKY"]   # cornering stiffness
        Kyg = Fz * (p["PKY6"] + p["PKY7"]*dfz) * p["LKYC"]   # camber stiffness

        SVyg = Fz * (p["PVY3"] + p["PVY4"]*dfz) * gamma * p["LKYC"] * p["LMUY"] * self.muyScale   # camber thrust
        SVy = Fz * (p["PVY1"] + p["PVY2"]*dfz) * p["LVY"] * p["LMUY"] * self.muyScale + SVyg
        SHy = (p["PHY1"] + p["PHY2"]*dfz) * p["LHY"] + (Kyg*gamma - SVyg) / (Ky + 1e-6)
        alpha_y = alpha + SHy

        Ey = np.minimum((p["PEY1"] + p["PEY2"]*dfz) * (1 + p["PEY5"]*gamma**2 - (p["PEY3"] + p["PEY4"]*gamma)*np.sign(alpha_y)), 1.0)
        By = Ky / (Cy*Dy + 1e-6)

        return Dy * np.sin(Cy*np.arctan(By*alpha_y - Ey*(By*alpha_y - np.arctan(By*alpha_y)))) + SVy

    def grip(self, Fz, camber=0):
        # peak (drive, brake, lateral) force [N], max over slip 0 to 0.3, slow, only used to build the table
        kappa = np.linspace(0, 0.3, 200)
        alpha = np.linspace(0, 0.3, 200)
        drive = np.max(np.abs(self.Fx(Fz, kappa, camber)))
        brake = np.max(np.abs(self.Fx(Fz, -kappa, camber)))
        lateral = np.max(np.abs(self.Fy(Fz, alpha, camber)))
        return drive, brake, lateral

    def build_table(self):
        # peak grip at n_grid loads from 0 to Fz_max, computed once
        self.Fz_grid = np.linspace(0.0, self.Fz_max, self.n_grid)   # N
        self.drive_table = np.zeros(self.n_grid)                     # N
        self.brake_table = np.zeros(self.n_grid)                     # N
        self.lat_table = np.zeros(self.n_grid)                       # N
        for i, Fz in enumerate(self.Fz_grid):
            self.drive_table[i], self.brake_table[i], self.lat_table[i] = self.grip(Fz, self.build_camber)
        self.slip_peak = self.peak_slip_angle()   # deg

    def peak(self, Fz):
        # peak (drive, brake, lateral) force [N] from the table, this is what the sim calls
        # above Fz_max the table stays flat, keep wheel loads under Fz_max
        return (float(np.interp(Fz, self.Fz_grid, self.drive_table)),
                float(np.interp(Fz, self.Fz_grid, self.brake_table)),
                float(np.interp(Fz, self.Fz_grid, self.lat_table)))

    def peak_drive(self, Fz):
        # peak drive force [N] only, for the traction solver
        return float(np.interp(Fz, self.Fz_grid, self.drive_table))

    def peak_slip_angle(self, Fz=None, a_max_deg=12.0, n=600):
        # slip angle [deg] at max lateral force, stored as tire.slip_peak for cornering_drag
        # default load 1.5 x Fz0, a loaded outside wheel
        if Fz is None:
            Fz = 1.5 * self.Fz0
        alpha = np.radians(np.linspace(0.0, a_max_deg, n))
        Fy = np.abs(self.Fy(Fz, alpha, self.build_camber))
        return float(np.degrees(alpha[np.argmax(Fy)]))