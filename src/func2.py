import torch

@torch.no_grad()
def f_eq_MB(E, T):
    # Maxwell-Boltzmann, no chemical potential
    return torch.exp(-E / T)

import torch


@torch.no_grad()
def f_FI_decay_MB_analytic_p_torch(
    p, a,
    m_chi, m_h, g_trilinear,
    pref=1.0,
    a_i=1.0,
    a0=1.0,
    T0=150.0,
    H0=3.16152130195488e-14,
    eps_q=1e-30,
):
    """
    Stable analytic freeze-in solution for h -> chi chi from a Maxwell-Boltzmann bath.

    Production version:
      - no expensive input validation,
      - numerically stable exponent structure,
      - avoids inf * 0 issues from the previous implementation.

    Assumes:
      - m_chi > 0
      - m_h > 2 m_chi
      - a, a_i, a0, T0, H0 > 0
      - p > 0
    """

    # Fast path: assume p is already a tensor in the right dtype/device
    p = torch.as_tensor(p)
    device, dtype = p.device, p.dtype

    a = torch.as_tensor(a, dtype=dtype, device=device)
    a_i = torch.as_tensor(a_i, dtype=dtype, device=device)
    a0 = torch.as_tensor(a0, dtype=dtype, device=device)
    T0 = torch.as_tensor(T0, dtype=dtype, device=device)
    H0 = torch.as_tensor(H0, dtype=dtype, device=device)
    m_chi = torch.as_tensor(m_chi, dtype=dtype, device=device)
    m_h = torch.as_tensor(m_h, dtype=dtype, device=device)
    g_trilinear = torch.as_tensor(g_trilinear, dtype=dtype, device=device)
    pref = torch.as_tensor(pref, dtype=dtype, device=device)
    eps_q = torch.as_tensor(eps_q, dtype=dtype, device=device)

    pi = torch.pi

    # comoving momentum
    q = p * a
    q_safe = torch.clamp(q, min=eps_q)

    beta = T0 * a0

    # safer than raw sqrt(...) against tiny roundoff near threshold
    beta_chi_sq = 1.0 - 4.0 * m_chi**2 / m_h**2
    beta_chi = torch.sqrt(torch.clamp(beta_chi_sq, min=0.0))

    y = torch.sqrt(q * q + (m_chi * a) ** 2)
    yi = torch.sqrt(q * q + (m_chi * a_i) ** 2)

    A = m_h**2 / (2.0 * beta * m_chi**2)

    # Overall prefactor without dangerous exponentials
    prefactor = (
        pref
        * g_trilinear**2
        * beta_chi
        * beta**2
        / (16.0 * pi * H0 * a0**2 * q_safe * m_h**2)
    )

    # Stable exponent combination:
    # exp(B - A y) = exp[-A (y - beta_chi q)]
    exp_i = torch.exp(-A * (yi - beta_chi * q))
    exp_f = torch.exp(-A * (y  - beta_chi * q))

    f = prefactor * (exp_i - exp_f)
    return f
