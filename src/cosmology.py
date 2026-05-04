from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.special import kn


BASE_DIR = Path(__file__).resolve().parent.parent
FILES_DIR = BASE_DIR / "files"
DEFAULT_G_ENERGY_FILE = FILES_DIR / "gEnergy.dat"
DEFAULT_G_ENTROPY_FILE = FILES_DIR / "gEntropy.dat"


class VariableGCosmology:
    def __init__(
        self,
        g_energy_file=DEFAULT_G_ENERGY_FILE,
        g_entropy_file=DEFAULT_G_ENTROPY_FILE,
        Mpl=2.4e18,
        T_gmax=1.0e4,
        T_smax=1.0e3,
        Trh=150.0,
        ai=1.0,
        Tmin=1e-13,
        npts=50000,
        interpolation="cubic",
    ):
        self.Mpl = Mpl
        self.T_gmax = T_gmax
        self.T_smax = T_smax
        self.Trh = Trh
        self.ai = ai
        self.Tmin = Tmin
        self.npts = npts
        self.interpolation = interpolation

        # Load tables
        self.g_energy_data = np.loadtxt(g_energy_file)
        self.g_entropy_data = np.loadtxt(g_entropy_file)

        # Sort by temperature
        self.g_energy_data = self.g_energy_data[np.argsort(self.g_energy_data[:, 0])]
        self.g_entropy_data = self.g_entropy_data[np.argsort(self.g_entropy_data[:, 0])]

        # Interpolate in log10(T)
        xg = np.log10(self.g_energy_data[:, 0])
        yg = self.g_energy_data[:, 1]

        xs = np.log10(self.g_entropy_data[:, 0])
        ys = self.g_entropy_data[:, 1]

        if interpolation == "cubic":
            self._g_rho_interp = CubicSpline(xg, yg, extrapolate=True)
            self._g_s_interp = CubicSpline(xs, ys, extrapolate=True)
        elif interpolation == "pchip":
            self._g_rho_interp = PchipInterpolator(xg, yg, extrapolate=True)
            self._g_s_interp = PchipInterpolator(xs, ys, extrapolate=True)
        else:
            raise ValueError("interpolation must be 'cubic' or 'pchip'.")

        # Build T(a) interpolation once
        self._build_T_of_a()

    def g_rho(self, T):
        T = np.asarray(T, dtype=float)
        T_eval = np.minimum(T, self.T_gmax)
        return self._g_rho_interp(np.log10(T_eval))

    def g_s(self, T):
        T = np.asarray(T, dtype=float)
        T_eval = np.minimum(T, self.T_smax)
        return self._g_s_interp(np.log10(T_eval))

    def entropy_density(self, T):
        T = np.asarray(T, dtype=float)
        return (2.0 * np.pi**2 / 45.0) * self.g_s(T) * T**3

    def rho_rad(self, T):
        T = np.asarray(T, dtype=float)
        return (np.pi**2 / 30.0) * self.g_rho(T) * T**4

    def H_of_T(self, T):
        T = np.asarray(T, dtype=float)
        return np.sqrt(self.rho_rad(T) / (3.0 * self.Mpl**2))

    def Hbar_of_T(self, T):
        T = np.asarray(T, dtype=float)
        T_eval = np.minimum(T, self.T_smax)
        dgs_dlog10T = self._g_s_interp.derivative()(np.log10(T_eval))
        dgs_dT = dgs_dlog10T / (T_eval * np.log(10.0))
        correction = 1.0 + T_eval * dgs_dT / (3.0 * self.g_s(T_eval))
        return self.H_of_T(T_eval) / correction

    def rho_eq_massive(self, T, m):
        T = np.asarray(T, dtype=float)
        z = m / T
        return (m**3 * T / (2.0 * np.pi**2)) * (kn(1, z) + 3.0 * T / m * kn(2, z))

    def nphieq_x(self, x, m):
        x = np.asarray(x, dtype=float)
        return (m**3 / (2.0 * np.pi**2 * x)) * kn(2, x)

    def _build_T_of_a(self):
        T_grid = np.logspace(np.log10(self.Tmin), np.log10(self.Trh), self.npts)
        s_rh = self.entropy_density(self.Trh)
        s_grid = self.entropy_density(T_grid)

        a_grid = self.ai * (s_rh / s_grid) ** (1.0 / 3.0)

        order = np.argsort(a_grid)
        a_grid = a_grid[order]
        T_grid = T_grid[order]

        if self.interpolation == "cubic":
            self._T_of_a_interp = CubicSpline(a_grid, T_grid, extrapolate=True)
        else:
            self._T_of_a_interp = PchipInterpolator(a_grid, T_grid, extrapolate=True)

        self.domain = (float(a_grid.min()), float(a_grid.max()))

    def T_of_a(self, a):
        return self._T_of_a_interp(a)

    def H_of_a(self, a):
        return self.H_of_T(self.T_of_a(a))
