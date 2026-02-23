# src/cosmology.py
import numpy as np
from scipy.interpolate import interp1d

class StdCosmology:
    """
    Standard radiation-dominated cosmology with constant g*.
    """

    def __init__(self, gstar=106.75, Mpl=2.435e18):
        self.gstar = gstar
        self.Mpl = Mpl

    def rho_rad(self, T):
        return (np.pi**2 / 30.0) * self.gstar * T**4

    def H_of_T(self, T):
        return np.sqrt(self.rho_rad(T) / (3.0 * self.Mpl**2))


def build_T_of_a_table(cosmo, Trh=150.0, ai=1.0,
                       Tmin=1e-4, npts=20000):
    """
    Build interpolation T(a) assuming radiation domination
    and entropy conservation.

    Returns:
        T_of_a (callable)
        H_of_a (callable)
        (amin, amax)
    """

    # Log-spaced temperature grid
    T_grid = np.logspace(np.log10(Tmin), np.log10(Trh), npts)

    # Entropy conservation: a T = const (for constant g*)
    a_grid = ai * Trh / T_grid

    # Build interpolation
    T_of_a = interp1d(a_grid, T_grid,
                      kind="linear",
                      bounds_error=False,
                      fill_value="extrapolate")

    def H_of_a(a):
        T = T_of_a(a)
        return cosmo.H_of_T(T)

    return T_of_a, H_of_a, (a_grid.min(), a_grid.max())
