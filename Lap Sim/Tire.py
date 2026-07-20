# Author: Anne-Sophie
# Summary: Reads MF6.1 fitted coefficients from the exported CSV and returns
#          peak tyre grip vs vertical load and camber, to build GG_Fz.
#          Slip in radians, camber in degrees.
#
# Equations: TNO MF-Tyre/MF-Swift 6.2 Equation Manual (rev 20130706),
#            section 1.2 (Fx) and 1.3 (Fy), steady-state pure slip.
#            Variable names follow the manual's symbols; comments give meaning.
#            Moments, pressure, and MF-Swift transients omitted (nominal pressure).

import numpy as np

class Tire:
    # muxScale/muyScale are the correlated grip levels from the Main.m g-g match
    # (raw fit is too grippy). Change them if correlation says otherwise.
    def __init__(self, file_path, muxScale=0.6, muyScale=0.9):
        name, value = np.loadtxt(file_path, delimiter=",", dtype=str, unpack=True)
        self.coefficient = {n: float(v) for n, v in zip(name, value)}
        self.Fz0 = self.coefficient["Fz0"]         # Fz0, nominal load the fit is referenced to [N]
        self.muxScale = muxScale
        self.muyScale = muyScale
        self._build_grip_table()

    # Magic Formula (manual sec 1.2, 1.3):
    #   F = D*sin(C*atan(B*x - E*(B*x - atan(B*x)))) + SV
    #   D = peak force, C = shape, B = stiffness, E = curvature, S = shifts.
    #   dfz = normalised load change, reshapes the curve with load.

    def Fx(self, Fz, kappa, camber=0):            # longitudinal force, manual 1.2 [N]
        p = self.coefficient
        gamma = np.radians(camber)                  # gamma, inclination angle [rad]
        dfz = (Fz - self.Fz0) / self.Fz0            # dfz, normalised vertical load change
        Cx  = p["PCX1"]                                                                                     # shape factor
        mux = (p["PDX1"] + p["PDX2"]*dfz) * (1 - p["PDX3"]*gamma**2) * p["LMUX"] * self.muxScale            # peak friction coefficient
        Dx  = mux * Fz                                                                                      # peak force
        Ex  = np.minimum((p["PEX1"] + p["PEX2"]*dfz + p["PEX3"]*dfz**2) * (1 - p["PEX4"]*np.sign(kappa)), 1.0)   # curvature (<=1); sign(kappa) = drive/brake asymmetry
        Kxk = Fz * (p["PKX1"] + p["PKX2"]*dfz) * np.exp(p["PKX3"]*dfz) * p["LKX"]                            # slip stiffness Kx_kappa
        Bx  = Kxk / (Cx*Dx + 1e-6)                                                                          # stiffness factor (1e-6 guards Fz=0)
        SVx = Fz * (p["PVX1"] + p["PVX2"]*dfz) * p["LVX"] * p["LMUX"] * self.muxScale                        # vertical shift
        return Dx * np.sin(Cx*np.arctan(Bx*kappa - Ex*(Bx*kappa - np.arctan(Bx*kappa)))) + SVx

    def Fy(self, Fz, alpha, camber=0):            # lateral force, manual 1.3 [N]
        p = self.coefficient
        gamma = np.radians(camber)                  # gamma, inclination angle [rad]
        dfz = (Fz - self.Fz0) / self.Fz0            # dfz, normalised vertical load change
        Cy    = p["PCY1"]                                                                                   # shape factor
        muy   = (p["PDY1"] + p["PDY2"]*dfz) * (1 - p["PDY3"]*gamma**2) * p["LMUY"] * self.muyScale          # peak friction coeff; PDY2<0 -> grip falls with load
        Dy    = muy * Fz                                                                                    # peak force
        Kya   = p["PKY1"]*self.Fz0 * (1 - p["PKY3"]*abs(gamma)) * np.sin(p["PKY4"]*np.arctan((Fz/self.Fz0) / (p["PKY2"] + p["PKY5"]*gamma**2))) * p["LKY"]   # cornering stiffness Ky_alpha
        Kyg0  = Fz * (p["PKY6"] + p["PKY7"]*dfz) * p["LKYC"]                                                # camber stiffness Ky_gamma0
        SVyg  = Fz * (p["PVY3"] + p["PVY4"]*dfz) * gamma * p["LKYC"] * p["LMUY"] * self.muyScale            # camber-thrust part of vertical shift
        SVy   = Fz * (p["PVY1"] + p["PVY2"]*dfz) * p["LVY"] * p["LMUY"] * self.muyScale + SVyg              # total vertical shift
        SHy   = (p["PHY1"] + p["PHY2"]*dfz) * p["LHY"] + (Kyg0*gamma - SVyg) / (Kya + 1e-6)                 # horizontal shift, carries camber onto slip axis
        alphay = alpha + SHy                                                                                # shifted slip angle alpha_y
        Ey    = np.minimum((p["PEY1"] + p["PEY2"]*dfz) * (1 + p["PEY5"]*gamma**2 - (p["PEY3"] + p["PEY4"]*gamma)*np.sign(alphay)), 1.0)   # curvature (<=1)
        By    = Kya / (Cy*Dy + 1e-6)                                                                        # stiffness factor
        return Dy * np.sin(Cy*np.arctan(By*alphay - Ey*(By*alphay - np.arctan(By*alphay)))) + SVy

    def grip(self, Fz, camber=0):                 # peak (drive, brake, lateral) [N]
        kappa = np.linspace(0, 0.3, 200)            # sweep slip, take the max: peak = the capacity the sim uses
        alpha = np.linspace(0, 0.3, 200)
        return (np.max(np.abs(self.Fx(Fz, kappa, camber))), np.max(np.abs(self.Fx(Fz, -kappa, camber))), np.max(np.abs(self.Fy(Fz, alpha, camber))))   # drive +kappa, brake -kappa
    
    def _build_grip_table(self, Fz_max=3500.0, n=80):
        """Precompute peak drive/brake/lateral vs Fz, once, by caching grip().
        The solver then interpolates instead of sweeping slip on every call."""
        self._Fz_grid = np.linspace(0.0, Fz_max, n)
        self._drive = np.zeros(n)
        self._brake = np.zeros(n)
        self._lat   = np.zeros(n)
        for i, f in enumerate(self._Fz_grid):
            self._drive[i], self._brake[i], self._lat[i] = self.grip(f)
    
    def peak(self, Fz, camber=0.0):
        """"Peak (drive, brake, lateral) [N], interpolated from the precomputed table."""
        return (float(np.interp(Fz, self._Fz_grid, self._drive)),
                float(np.interp(Fz, self._Fz_grid, self._brake)),
                float(np.interp(Fz, self._Fz_grid, self._lat)))