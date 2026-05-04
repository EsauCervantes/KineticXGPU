import torch
import numpy as np
import math
from scipy.special import expi  # Ei(x)


@torch.no_grad()
def f_eq_MB(E, T):
    # Maxwell-Boltzmann, no chemical potential
    return torch.exp(-E / T)
    

def f_FI_inverse_decay_MB_analytic_p(
    p, a,
    m, Gamma,
    pref=1.0,
    a_i=1.0,
    a0=1.0,
    T0=150.0,
    H0=3.16152130195488e-14,
):
    """
    Analytic solution for pure FI with kernel:
      C_FI = pref * (m/E) * Gamma * exp(-E/T)
    and RHS:
      df/da = C_FI / (a H E),
    with:
      H(a) = H0 * (a0/a)^2
      T(a) = T0 * (a0/a)
      p = q/a  (comoving q conserved)

    Here we return f as a function of *physical* momentum p at scale factor a
    by using q = p*a.

    Inputs:
      p: float or np.ndarray, physical momentum at epoch a
      a: float, scale factor
      m: float, mass of produced particle (= mediator mass in your setup)
      Gamma: float, decay width (same units as H0)
      pref: overall multiplicity factor
      a_i: initial scale factor (where f is taken to be 0)
      a0, T0, H0: reference values; you specified a0=1, T0=150, H0=H(150 GeV)

    Returns:
      f(p,a): float or np.ndarray
    """
    p = np.asarray(p, dtype=float)
    a = float(a)
    a_i = float(a_i)

    if m <= 0:
        raise ValueError("m must be > 0 for this formula.")
    if a <= 0 or a_i <= 0:
        raise ValueError("a and a_i must be > 0.")

    # comoving momentum corresponding to physical momentum p at epoch a
    q = p * a

    beta = T0 * a0  # = T(a)*a is constant for T ~ 1/a
    K = pref * Gamma * m / (H0 * a0**2)  # with a0=1 => pref*Gamma*m/H0

    # y(a) = sqrt(q^2 + m^2 a^2)
    y  = np.sqrt(q*q + (m*a )**2)
    yi = np.sqrt(q*q + (m*a_i)**2)

    def F(z):
        # F(z) = -beta (z+beta) e^{-z/beta} - q^2 Ei(-z/beta)
        return -beta * (z + beta) * np.exp(-z / beta) - (q*q) * expi(-z / beta)

    f = (K / (m**4)) * (F(y) - F(yi))
    return f

def gamma_ap_to_ff(mA, mf, Q, eps, Nc=1, alpha_em=1/137.035999084):
    """
    Γ(A' -> f fbar) using math (scalar version).

    Parameters
    ----------
    mA : float
        Dark photon mass
    mf : float
        Fermion mass
    Q : float
        Electric charge
    eps : float
        Kinetic mixing ε
    Nc : int
        Color factor
    alpha_em : float
        Fine structure constant

    Returns
    -------
    float
        Partial width
    """
    if mA <= 0.0:
        return 0.0

    if mA <= 2.0 * mf:
        return 0.0

    r = (mf / mA) ** 2
    arg = 1.0 - 4.0 * r

    if arg <= 0.0:
        return 0.0

    beta = math.sqrt(arg)

    pref = Nc * alpha_em * (eps ** 2) * (Q)**2 / 3.0

    return pref * mA * (1.0 + 2.0 * r) * beta


# ---- channel list (masses in GeV) ----
# Charges: e, mu, tau: -1; up-type: +2/3; down-type: -1/3
# Nc: leptons 1, quarks 3
_SM_FERMIONS = [
    # leptons
    ("e",   0.00051099895, -1.0, 1),
    ("mu",  0.1056583745,  -1.0, 1),
    ("tau", 1.77686,       -1.0, 1),

    # quarks (partonic)
    ("u", 0.00216,  +2/3, 3),
    ("d", 0.00467,  -1/3, 3),
    ("s", 0.093,    -1/3, 3),
    ("c", 1.27,     +2/3, 3),
    ("b", 4.18,     +2/3, 3),  # same as +2/3
    ("t", 172.76,   +2/3, 3),
]



def gamma_ap_total_to_sm_fermions(
    mA, eps,
    alpha_em=1/137.035999084,
    include=("e","mu","tau","u","d","s","c","b","t"),
):
    """
    Total Γ(A' -> SM fermions), scalar math version.

    Parameters
    ----------
    mA : float
        Dark photon mass
    eps : float
        Kinetic mixing
    include : tuple
        Channels to include

    Returns
    -------
    float
        Total width
    """
    total = 0.0
    include_set = set(include)

    for name, mf, Q, Nc in _SM_FERMIONS:
        if name in include_set:
            total += gamma_ap_to_ff(
                mA, mf, Q, eps,
                Nc=Nc,
                alpha_em=alpha_em
            )

    return total
