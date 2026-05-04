import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kn  # modified Bessel K_n

# ---------- helper to infer T and z from your existing interpolator ----------
def infer_T_z_from_interpolator(y_of_p, pmin, pmax, m, npts=3000):
    p_grid = np.logspace(np.log10(pmin), np.log10(pmax), npts)
    y_grid = y_of_p(p_grid)   # y = p^2 f
    E_grid = np.sqrt(p_grid**2 + m**2)

    # number density
    n = np.trapz(y_grid, p_grid) / (2.0 * np.pi**2)

    # T = <p^2/(3E)>
    T = np.trapz(y_grid * (p_grid**2 / (3.0 * E_grid)), p_grid) / max(np.trapz(y_grid, p_grid), 1e-300)

    # neq(T,m)
    neq = (m**2 * T * kn(2, m / T)) / (2.0 * np.pi**2)

    # fugacity
    z = n / max(neq, 1e-300)

    return T, z, n, neq


def number_density_from_grid(p, f, g=1.0):
    p = np.asarray(p, dtype=float)
    f = np.asarray(f, dtype=float)
    integrand = p**2 * f
    return g * np.trapz(integrand, p) / (2.0 * np.pi**2)

def energy_density_from_grid(p, f, m, g=1.0):
    p = np.asarray(p, dtype=float)
    f = np.asarray(f, dtype=float)
    E = np.sqrt(p**2 + m**2)
    integrand = E * p**2 * f
    return g * np.trapz(integrand, p) / (2.0 * np.pi**2)

def inferred_temperature_from_grid(p, f, m):
    p = np.asarray(p, dtype=float)
    f = np.asarray(f, dtype=float)
    E = np.sqrt(p**2 + m**2)

    num = np.trapz(p**2 * f * (p**2 / (3.0 * E)), p)
    den = np.trapz(p**2 * f, p)

    return num / max(den, 1e-300)

def neq_MB(T, m, g=1.0):
    if T <= 0:
        return 0.0
    return g * (m**2 * T * kn(2, m / T)) / (2.0 * np.pi**2)

def inferred_fugacity_from_grid(p, f, m, g=1.0):
    Tchi = inferred_temperature_from_grid(p, f, m)
    n = number_density_from_grid(p, f, g=g)
    neq = neq_MB(Tchi, m, g=g)
    z = n / max(neq, 1e-300)
    return z, Tchi, n, neq

def f_MB(p, m, T, z=1.0):
    E = np.sqrt(p**2 + m**2)
    return z * np.exp(-E / T)

def compare_to_MB(p, f_num, m, g=1.0, label_num="numerical"):
    z, Tchi, n, neq = inferred_fugacity_from_grid(p, f_num, m, g=g)
    f_fit = f_MB(p, m, Tchi, z=z)

    print(f"Tchi = {Tchi}")
    print(f"n    = {n}")
    print(f"neq  = {neq}")
    print(f"z    = {z}")

    plt.figure(figsize=(7,5))
    plt.loglog(p, p**2 * f_num, label=label_num)
    plt.loglog(p, p**2 * f_fit, "--", label=fr"MB fit: $T_\chi={Tchi:.4g}$, $z={z:.4g}$")
    plt.xlabel("p")
    plt.ylabel(r"$p^2 f(p)$")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7,4))
    ratio = np.where(f_fit > 0, f_num / f_fit, np.nan)
    plt.semilogx(p, ratio)
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("p")
    plt.ylabel(r"$f_{\rm num}/f_{\rm MB}$")
    plt.tight_layout()
    plt.show()

    return {"Tchi": Tchi, "z": z, "n": n, "neq": neq, "f_MB": f_fit}
    
    
    
def n_eq_MB(T, m):
    return (m**2 * T / (2.0 * np.pi**2)) * kn(2, m / T)


def N_eq(a, T, m):
    return a**3 * n_eq_MB(T, m)


def rho_eq(T, m):
    T = np.asarray(T, dtype=float)
    z = m / T
    return (m**3 * T / (2.0 * np.pi**2)) * (kn(1, z) + 3.0 * T / m * kn(2, z))


def drho_eq_dT(T, m):
    T = np.asarray(T, dtype=float)
    z = m / T
    return (
        m
        * (
            (m**3 / T + 12.0 * m * T) * kn(0, z)
            + (5.0 * m**2 + 24.0 * T**2) * kn(1, z)
        )
        / (2.0 * np.pi**2)
    )
    
    

def Gamma_htophiphi(T, mphi, mh=125.0, v=246.0):
    T = np.asarray(T, dtype=float)
    x = 1.0 - 4.0 * mphi**2 / mh**2
    theta = (x > 0.0).astype(float) if np.ndim(T) else float(x > 0.0)

    if x <= 0.0:
        return np.zeros_like(T, dtype=float)

    pref = (v**2 * mh) / (16.0 * np.pi**3)
    out = theta * pref * np.sqrt(x) * T * kn(1, mh / T)
    return out


def Gamma_htophiphi_energy(T, mphi, mh=125.0, v=246.0):
    T = np.asarray(T, dtype=float)
    x = 1.0 - 4.0 * mphi**2 / mh**2
    theta = (x > 0.0).astype(float) if np.ndim(T) else float(x > 0.0)

    if x <= 0.0:
        return np.zeros_like(T, dtype=float)

    pref = (v**2 * mh**2) / (32.0 * np.pi**3)
    out = theta * pref * np.sqrt(x) * T * kn(2, mh / T)
    return out
