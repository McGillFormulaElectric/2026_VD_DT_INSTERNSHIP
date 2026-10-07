# Author: Anne-Sophie
# Tire model for the lap sim.
#
# What this does:
#   Loads the MF6.1 tyre fit we made in MATLAB (from TTC data) and tells the sim
#   how much force each tyre can make at a given vertical load Fz.
#   The sim only needs the PEAK force (the most grip the tyre has), so we compute
#   that once into a table at startup, then just look it up while the sim runs.
#   That lookup is what keeps the sim fast.
#
# Units: Fz in N, slip angle and slip ratio in radians, camber in degrees.
# Equations: TNO MF-Tyre 6.2 Equation Manual, section 1.2 (Fx) and 1.3 (Fy),
#            steady state, pure slip, nominal pressure. Names follow the manual.

import numpy as np
import scipy.io


class Tire:
    def __init__(self, param_path, fzref_path, muxScale=0.474, muyScale=0.65,
                 Fz_max=3500.0, n_grid=80, build_camber=0.0):
        # param_path: MF61_Parameters.mat, holds Fz0, the coefficients (p) and scaling factors (lamda)
        # fzref_path: FZ_Reference.mat, the loads the TTC fit was done at
        # muxScale, muyScale: grip correction from correlation. The raw TTC fit is
        #                     too grippy (sandpaper belt), so we scale it down to match the car.
        # Fz_max, n_grid: the grip table covers 0 to Fz_max N in n_grid points
        # build_camber: static camber of the car [deg], the table is built at this camber
        mat = scipy.io.loadmat(param_path, struct_as_record=False, squeeze_me=True)
        p = mat["p"]
        lam = mat["lamda"]
        self.FZref = np.asarray(scipy.io.loadmat(fzref_path, squeeze_me=True)["FZref"], float)
        self.Fz0 = float(mat["Fz0"])   # reference load of the fit [N]

        # rename the MATLAB fields to the manual's names (p.Cx1 -> PCX1)
        # lambdas not in the MATLAB file are set to 1.0, which means "no scaling"
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

    # muxScale and muyScale rebuild the table when you change them,
    # so "tire.muyScale = 0.85" is all you need, peak() is up to date right after
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

    # Magic Formula, the shape of a tyre force curve:
    #   F = D * sin(C * atan(B*x - E*(B*x - atan(B*x)))) + SV
    #   x  = slip (slip ratio kappa for Fx, slip angle alpha for Fy)
    #   D  = peak force, the top of the curve
    #   C  = shape, how the curve looks after the peak
    #   B  = stiffness, how steep the curve starts
    #   E  = curvature, how round the peak is
    #   SV, SH = small shifts so the curve doesn't have to pass through zero
    #   dfz = how far Fz is from Fz0, in fraction. Grip per N drops as load goes up.

    def Fx(self, Fz, kappa, camber=0):
        # longitudinal force [N] for a slip ratio kappa (+ drive, - brake)
        p = self.coefficient
        gamma = np.radians(camber)
        dfz = (Fz - self.Fz0) / self.Fz0

        Cx = p["PCX1"]
        mux = (p["PDX1"] + p["PDX2"]*dfz) * (1 - p["PDX3"]*gamma**2) * p["LMUX"] * self.muxScale   # friction coefficient
        Dx = mux * Fz
        Ex = np.minimum((p["PEX1"] + p["PEX2"]*dfz + p["PEX3"]*dfz**2) * (1 - p["PEX4"]*np.sign(kappa)), 1.0)   # drive and brake can differ
        Kx = Fz * (p["PKX1"] + p["PKX2"]*dfz) * np.exp(p["PKX3"]*dfz) * p["LKX"]   # slip stiffness, slope at zero slip
        Bx = Kx / (Cx*Dx + 1e-6)   # 1e-6 avoids divide by zero at Fz = 0
        SVx = Fz * (p["PVX1"] + p["PVX2"]*dfz) * p["LVX"] * p["LMUX"] * self.muxScale

        return Dx * np.sin(Cx*np.arctan(Bx*kappa - Ex*(Bx*kappa - np.arctan(Bx*kappa)))) + SVx

    def Fy(self, Fz, alpha, camber=0):
        # lateral force [N] for a slip angle alpha
        p = self.coefficient
        gamma = np.radians(camber)
        dfz = (Fz - self.Fz0) / self.Fz0

        Cy = p["PCY1"]
        muy = (p["PDY1"] + p["PDY2"]*dfz) * (1 - p["PDY3"]*gamma**2) * p["LMUY"] * self.muyScale   # friction coefficient
        Dy = muy * Fz
        Ky = p["PKY1"]*self.Fz0 * (1 - p["PKY3"]*abs(gamma)) * np.sin(p["PKY4"]*np.arctan((Fz/self.Fz0) / (p["PKY2"] + p["PKY5"]*gamma**2))) * p["LKY"]   # cornering stiffness
        Kyg = Fz * (p["PKY6"] + p["PKY7"]*dfz) * p["LKYC"]   # camber stiffness

        # camber thrust: camber adds a bit of side force on its own
        SVyg = Fz * (p["PVY3"] + p["PVY4"]*dfz) * gamma * p["LKYC"] * p["LMUY"] * self.muyScale
        SVy = Fz * (p["PVY1"] + p["PVY2"]*dfz) * p["LVY"] * p["LMUY"] * self.muyScale + SVyg
        SHy = (p["PHY1"] + p["PHY2"]*dfz) * p["LHY"] + (Kyg*gamma - SVyg) / (Ky + 1e-6)
        alpha_y = alpha + SHy

        Ey = np.minimum((p["PEY1"] + p["PEY2"]*dfz) * (1 + p["PEY5"]*gamma**2 - (p["PEY3"] + p["PEY4"]*gamma)*np.sign(alpha_y)), 1.0)
        By = Ky / (Cy*Dy + 1e-6)

        return Dy * np.sin(Cy*np.arctan(By*alpha_y - Ey*(By*alpha_y - np.arctan(By*alpha_y)))) + SVy

    def grip(self, Fz, camber=0):
        # peak (drive, brake, lateral) force [N]
        # sweeps slip from 0 to 0.3 and keeps the max. Slow, so only used to build the table.
        kappa = np.linspace(0, 0.3, 200)
        alpha = np.linspace(0, 0.3, 200)
        drive = np.max(np.abs(self.Fx(Fz, kappa, camber)))
        brake = np.max(np.abs(self.Fx(Fz, -kappa, camber)))
        lateral = np.max(np.abs(self.Fy(Fz, alpha, camber)))
        return drive, brake, lateral

    def build_table(self):
        # compute peak grip at n_grid loads from 0 to Fz_max, once
        # the sim then interpolates in this table instead of calling grip() every step
        self.Fz_grid = np.linspace(0.0, self.Fz_max, self.n_grid)
        self.drive_table = np.zeros(self.n_grid)
        self.brake_table = np.zeros(self.n_grid)
        self.lat_table = np.zeros(self.n_grid)
        for i, Fz in enumerate(self.Fz_grid):
            self.drive_table[i], self.brake_table[i], self.lat_table[i] = self.grip(Fz, self.build_camber)
        self.slip_peak = self.peak_slip_angle()
        
    def peak(self, Fz):
        # peak (drive, brake, lateral) force [N] at load Fz, from the table. This is what the sim calls.
        # careful: above Fz_max the table holds the last value flat, keep wheel loads under Fz_max
        return (float(np.interp(Fz, self.Fz_grid, self.drive_table)),
                float(np.interp(Fz, self.Fz_grid, self.brake_table)),
                float(np.interp(Fz, self.Fz_grid, self.lat_table)))

    def peak_drive(self, Fz):
        # peak drive force only [N], used by the traction solver which doesn't need brake or lateral
        return float(np.interp(Fz, self.Fz_grid, self.drive_table))

    def peak_slip_angle(self, Fz=None, a_max_deg=12.0, n=600):
        # slip angle [deg] where lateral force is max, used as car.slip_peak in cornering_drag
        # default load is 1.5 * Fz0, a loaded outside wheel in a fast corner
        if Fz is None:
            Fz = 1.5 * self.Fz0
        alpha = np.radians(np.linspace(0.0, a_max_deg, n))
        Fy = np.abs(self.Fy(Fz, alpha, self.build_camber))
        return float(np.degrees(alpha[np.argmax(Fy)]))