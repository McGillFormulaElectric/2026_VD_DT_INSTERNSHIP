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
    # (raw fit is too grippy). Setting either one after construction (tire.muyScale
    # = 0.85) now rebuilds the grip table automatically, so peak() picks it up.
    # build_camber is the static camber the table is baked at — set it to the car's
    # real static camber so peak() reflects camber thrust instead of a 0 deg tyre.
    def __init__(self, file_path, muxScale=0.6, muyScale=0.9,
                 Fz_max=3500.0, n_grid=80, build_camber=0.0):
        name, value = np.loadtxt(file_path, delimiter=",", dtype=str, unpack=True)
        self.coefficient = {n: float(v) for n, v in zip(name, value)}
        self.Fz0 = self.coefficient["Fz0"]         # Fz0, nominal load the fit is referenced to [N]
        # Set backing fields directly so the property setters don't try to rebuild
        # the table before the grid exists; build once at the end of __init__.
        self._muxScale = muxScale
        self._muyScale = muyScale
        self._Fz_max = Fz_max
        self._n_grid = n_grid
        self._build_camber = build_camber
        self._ready = True
        self._build_grip_table()

    # --- grip-level / build knobs: assigning any of these rebuilds the table ---
    @property
    def muxScale(self):
        return self._muxScale

    @muxScale.setter
    def muxScale(self, val):
        self._muxScale = val
        if getattr(self, "_ready", False):
            self._build_grip_table()

    @property
    def muyScale(self):
        return self._muyScale

    @muyScale.setter
    def muyScale(self, val):
        self._muyScale = val
        if getattr(self, "_ready", False):
            self._build_grip_table()

    @property
    def build_camber(self):
        return self._build_camber

    @build_camber.setter
    def build_camber(self, val):
        self._build_camber = val
        if getattr(self, "_ready", False):
            self._build_grip_table()

    @property
    def Fz_max(self):
        return self._Fz_max

    @Fz_max.setter
    def Fz_max(self, val):
        self._Fz_max = val
        if getattr(self, "_ready", False):
            self._build_grip_table()

    def set_scales(self, muxScale=None, muyScale=None, build_camber=None):
        """Set several knobs and rebuild the grip table ONCE (cheaper than
        assigning them one at a time when each assignment rebuilds)."""
        if muxScale is not None:
            self._muxScale = muxScale
        if muyScale is not None:
            self._muyScale = muyScale
        if build_camber is not None:
            self._build_camber = build_camber
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

    def _build_grip_table(self):
        """Precompute peak drive/brake/lateral vs Fz, once, by caching grip()
        at the build camber. The solver then interpolates instead of sweeping
        slip on every call. Called again automatically when a grip-level or
        build knob changes, so peak() never goes stale."""
        self._Fz_grid = np.linspace(0.0, self._Fz_max, self._n_grid)
        self._drive = np.zeros(self._n_grid)
        self._brake = np.zeros(self._n_grid)
        self._lat   = np.zeros(self._n_grid)
        for i, f in enumerate(self._Fz_grid):
            self._drive[i], self._brake[i], self._lat[i] = self.grip(f, self._build_camber)

    def peak(self, Fz, camber=None):
        """Peak (drive, brake, lateral) [N], interpolated from the precomputed
        table. camber is ignored by default (the table is baked at build_camber);
        pass an explicit camber only for a one-off off-table query, which sweeps
        slip live and is slow — do not call it from the solver hot loop.

        NOTE: np.interp clamps above Fz_max, so a wheel loaded past the top of
        the grid returns the grip AT Fz_max held flat, not the continued falloff.
        Check max wheel load across a run stays under Fz_max (raise it if not)."""
        if camber is not None:
            return self.grip(Fz, camber)
        return (float(np.interp(Fz, self._Fz_grid, self._drive)),
                float(np.interp(Fz, self._Fz_grid, self._brake)),
                float(np.interp(Fz, self._Fz_grid, self._lat)))
        
    def peak_drive(self, Fz):
        """Peak drive force [N] only — the traction solver's hot path, avoids
        interpolating the brake/lateral columns it does not use."""
        return float(np.interp(Fz, self._Fz_grid, self._drive))

    def max_table_load(self):
        """Top of the Fz grid [N] — compare against the largest wheel load you
        actually see, to know whether peak() is clamping."""
        return self._Fz_max

    def peak_slip_angle(self, Fz=None, camber=None, a_max_deg=12.0, n=600):
        """Slip angle [deg] at maximum lateral force — the value cornering_drag
        uses as slip_peak. Swept from the MF Fy curve, so it always matches the
        loaded tyre. Defaults: 1.5*Fz0 (a fast-corner outer-wheel load, where
        induced drag matters) and the table's build camber."""
        Fz = 1.5 * self.Fz0 if Fz is None else Fz
        camber = self._build_camber if camber is None else camber
        alpha = np.radians(np.linspace(0.0, a_max_deg, n))
        Fy = np.abs(self.Fy(Fz, alpha, camber))
        return float(np.degrees(alpha[np.argmax(Fy)]))