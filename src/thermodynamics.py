import numpy as np
from scipy.special import kn  # modified Bessel K_n


def n_eq_MB(T, m):
    return (m**2 * T / (2.0 * np.pi**2)) * kn(2, m / T)


def N_eq(a, T, m):
    return a**3 * n_eq_MB(T, m)


def Gamma_htophiphi(T, mphi, mh=125.0, v=246.0):
    T = np.asarray(T, dtype=float)
    x = 1.0 - 4.0 * mphi**2 / mh**2

    if x <= 0.0:
        return np.zeros_like(T, dtype=float)

    pref = (v**2 * mh) / (16.0 * np.pi**3)
    out = pref * np.sqrt(x) * T * kn(1, mh / T)
    return out


def Gamma_htophiphi_energy(T, mphi, mh=125.0, v=246.0):
    T = np.asarray(T, dtype=float)
    x = 1.0 - 4.0 * mphi**2 / mh**2

    if x <= 0.0:
        return np.zeros_like(T, dtype=float)

    pref = (v**2 * mh**2) / (32.0 * np.pi**3)
    out = pref * np.sqrt(x) * T * kn(2, mh / T)
    return out
