# src/collision_log.py
import numpy as np
import torch

from grid_log import (
    p_of_a,
    q_of_a,
    grid_edges_from_centers_log,
    remap_two_bin_conservative_energy_logq,
)

from func import f_eq_MB


def energy(p, m):
    return torch.sqrt(p * p + m * m)


def _leggauss_torch(Ng, device, dtype):
    x, w = np.polynomial.legendre.leggauss(Ng)
    mu = torch.tensor(x, device=device, dtype=dtype)
    wt = torch.tensor(w, device=device, dtype=dtype)
    return mu, wt


@torch.no_grad()
def _F_kernel_batch_constant_M2_1d(pi, Ei, pn, En, pm, Em, m, lam, Ng, eps_disc=0.0):
    """
    Constant-|M|^2 kernel with |M|^2 = lam^2.

    Returns:
      F: (B,N,N)
    """
    device, dtype = pi.device, pi.dtype
    mu2, w2 = _leggauss_torch(Ng, device, dtype)

    lam2 = torch.as_tensor(lam, device=device, dtype=dtype) ** 2
    m2 = torch.as_tensor(m, device=device, dtype=dtype) ** 2

    B = pi.shape[0]
    N = pn.shape[1]
    pnpm = pn * pm
    pref = 1.0 / (4.0 * (2.0 * torch.pi) ** 4)

    A0_base = m2 + En * Em - En * Ei - Em * Ei
    Facc = torch.zeros((B, N, N), device=device, dtype=dtype)

    minus_one = torch.as_tensor(-1.0, device=device, dtype=dtype)
    plus_one = torch.as_tensor(1.0, device=device, dtype=dtype)

    for a2 in range(Ng):
        c2 = mu2[a2]
        w2a = w2[a2]
        s2sq = 1.0 - c2 * c2

        A0 = A0_base + pi * (pn * c2)
        A1 = pi * pm - (pnpm * c2)

        K2 = (pnpm * pnpm) * s2sq

        a = -(K2 + A1 * A1)
        b = -2.0 * A0 * A1
        c = K2 - A0 * A0

        disc = b * b - 4.0 * a * c
        mask_disc = disc > eps_disc

        sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))

        denom = 2.0 * a
        r1 = (-b - sqrt_disc) / denom
        r2 = (-b + sqrt_disc) / denom
        rmin = torch.minimum(r1, r2)
        rmax = torch.maximum(r1, r2)

        c3min = torch.maximum(rmin, minus_one)
        c3max = torch.minimum(rmax, plus_one)
        mask_int = mask_disc & (c3max > c3min)

        sqrt_minus_a = torch.sqrt(torch.clamp(-a, min=0.0))
        inv_sqrt_disc = torch.where(mask_int, 1.0 / sqrt_disc, torch.zeros_like(sqrt_disc))

        def u_of(x):
            u = (2.0 * a * x + b) * inv_sqrt_disc
            return torch.clamp(u, -1.0, 1.0)

        umin = u_of(c3min)
        umax = u_of(c3max)

        I = torch.where(
            mask_int,
            (torch.asin(umin) - torch.asin(umax))
            / torch.where(mask_int, sqrt_minus_a, torch.ones_like(sqrt_minus_a)),
            torch.zeros_like(disc),
        )

        contrib = lam2 * pnpm * I
        Facc = Facc + w2a * contrib

    F = pref * Facc
    F = torch.clamp(F, min=0.0)
    return F


@torch.no_grad()
def C_self_torch_logq(
    f, a,
    q, logq0, dlogq, log_space,
    m, lam,
    Ng=16, batch_size=16,
    apply_conservation_projection=True,
    return_diagnostics=False,
):
    """
    Fast production version of the self-scattering collision term on a log-q grid.

    Assumes inputs are already valid:
      - f.shape == q.shape
      - q strictly increasing and positive
      - a > 0
      - m >= 0

    If return_diagnostics=False, avoids extra bookkeeping.
    """
    f = torch.as_tensor(f)
    device, dtype = f.device, f.dtype

    q = torch.as_tensor(q, device=device, dtype=dtype)
    a = torch.as_tensor(a, device=device, dtype=dtype)
    m = torch.as_tensor(m, device=device, dtype=dtype)

    N = q.numel()

    # physical momentum / energy
    p = q / a
    E = torch.sqrt(p * p + m * m)

    # bin widths in p induced by log-q grid
    q_edges = grid_edges_from_centers_log(q)
    p_edges = q_edges / a
    dp_vec = p_edges[1:] - p_edges[:-1]

    dp_n = dp_vec.view(1, N, 1)
    dp_m = dp_vec.view(1, 1, N)

    p2 = p * p
    p2_n = p2.view(1, N, 1)
    p2_m = p2.view(1, 1, N)

    f_n = f.view(1, N, 1)
    f_m = f.view(1, 1, N)

    pn = p.view(1, N, 1)
    pm = p.view(1, 1, N)
    En = E.view(1, N, 1)
    Em = E.view(1, 1, N)

    C = torch.empty_like(f)

    # since gchi = 1 always
    pref0 = 0.5

    if return_diagnostics:
        total_valid = torch.zeros((), device=device, dtype=dtype)
        total_phys = torch.zeros((), device=device, dtype=dtype)
        total_outside_weight = torch.zeros((), device=device, dtype=dtype)
        total_all_weight = torch.zeros((), device=device, dtype=dtype)

    w_nm = dp_n * dp_m

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ib = torch.arange(start, end, device=device)
        B = ib.numel()

        pi = p[ib].view(B, 1, 1)
        Ei = E[ib].view(B, 1, 1)

        pi2 = p2[ib].view(B, 1, 1)
        ptil2 = p2_n + p2_m - pi2
        phys_ptil = (ptil2 >= 0)

        # avoid indexed assignment; cheaper and simpler
        ptil = torch.sqrt(torch.clamp(ptil2, min=0.0))
        qtil = a * ptil

        if return_diagnostics:
            ftil, _, _, _, _, inside_q = remap_two_bin_conservative_energy_logq(
                q_grid=q,
                f=f,
                q_query=qtil,
                a=a,
                m=m,
                clamp=False,
                return_meta=True,
            )
        else:
            ftil = remap_two_bin_conservative_energy_logq(
                q_grid=q,
                f=f,
                q_query=qtil,
                a=a,
                m=m,
                clamp=False,
                return_meta=False,
            )
            inside_q = (qtil >= q[0]) & (qtil <= q[-1])

        F = _F_kernel_batch_constant_M2_1d(
            pi, Ei, pn, En, pm, Em, m, lam, Ng, eps_disc=0.0
        )

        fi = f[ib].view(B, 1, 1)
        valid = phys_ptil & inside_q

        gain_loss = torch.where(
            valid,
            (f_n * f_m) - fi * ftil,
            torch.zeros_like(ftil),
        )

        C_batch = pref0 * torch.sum(w_nm * F * gain_loss, dim=(1, 2))
        C[start:end] = C_batch

        if return_diagnostics:
            total_valid = total_valid + valid.to(dtype).sum()
            total_phys = total_phys + phys_ptil.to(dtype).sum()

            outside = phys_ptil & (~inside_q)
            W = F.abs() * fi.abs()
            total_outside_weight = total_outside_weight + (W.masked_fill(~outside, 0.0)).sum()
            total_all_weight = total_all_weight + W.sum()

    if return_diagnostics:
        I_energy_raw, I_number_raw, rE_raw, rN_raw = check_conservation_nonuniform(C, p, E, dp_vec)

    if apply_conservation_projection:
        C = enforce_number_energy(C, p, E, dp_vec)

    if return_diagnostics:
        I_energy, I_number, rE, rN = check_conservation_nonuniform(C, p, E, dp_vec)

        frac_valid = total_valid / (total_phys + 1e-300)
        frac_outside_weight = total_outside_weight / (total_all_weight + 1e-300)

        diag = {
            "valid_fraction_given_phys": frac_valid,
            "outside_weight_fraction": frac_outside_weight,
            "I_energy_raw": I_energy_raw,
            "I_number_raw": I_number_raw,
            "rel_energy_raw": rE_raw,
            "rel_number_raw": rN_raw,
            "I_energy": I_energy,
            "I_number": I_number,
            "rel_energy": rE,
            "rel_number": rN,
        }
        return C, E, p, dp_vec, diag

    return C, E, p, dp_vec


@torch.no_grad()
def enforce_number_energy(C, p, E, dp, eps=0.0):
    """
    Project C onto the subspace with vanishing discrete energy and number moments.

    Discrete moments:
      energy ~ sum_i p_i^2 dp_i C_i
      number ~ sum_i p_i^2 dp_i C_i / E_i
    """
    dp_t = torch.as_tensor(dp, device=p.device, dtype=p.dtype)
    w = p * p * dp_t

    S0 = torch.sum(w * C)
    S1 = torch.sum(w * (C / (E + eps)))

    A00 = torch.sum(w)
    A01 = torch.sum(w * E)
    A10 = torch.sum(w / (E + eps))
    A11 = torch.sum(w * (E / (E + eps)))

    det = A00 * A11 - A01 * A10
    alpha = (S0 * A11 - S1 * A01) / det
    beta = (-S0 * A10 + S1 * A00) / det

    return C - alpha - beta * E


@torch.no_grad()
def check_conservation_nonuniform(C, p, E, dp_vec):
    dp_vec = torch.as_tensor(dp_vec, device=p.device, dtype=p.dtype)
    w = p * p * dp_vec

    I_energy = torch.sum(w * C)
    I_number = torch.sum(w * (C / E))

    scale_energy = torch.sum(w * torch.abs(C)) + 1e-300
    scale_number = torch.sum(w * torch.abs(C / E)) + 1e-300

    return I_energy, I_number, I_energy / scale_energy, I_number / scale_number



@torch.no_grad()
def C_FI_decay_MB(
    p, T,
    m_chi, m_h, g_trilinear,
    pref=1.0,
):
    """
    Momentum-differential freeze-in source for h -> chi chi
    from a Maxwell-Boltzmann bath.

    C_FI(p,T) =
        pref * [g^2/(32 pi)] * beta_chi * T/(E_chi p)
        * exp[-E_h_min(p)/T]

    with
        beta_chi = sqrt(1 - 4 m_chi^2 / m_h^2)
        E_h_min  = (m_h^2 E_chi - m_h p sqrt(m_h^2 - 4 m_chi^2)) / (2 m_chi^2)

    Valid for m_chi > 0 and m_h > 2 m_chi.
    """
    p = torch.as_tensor(p)
    device, dtype = p.device, p.dtype

    T = torch.as_tensor(T, device=device, dtype=dtype)
    m_chi = torch.as_tensor(m_chi, device=device, dtype=dtype)
    m_h = torch.as_tensor(m_h, device=device, dtype=dtype)
    g_trilinear = torch.as_tensor(g_trilinear, device=device, dtype=dtype)
    pref = torch.as_tensor(pref, device=device, dtype=dtype)

    #if torch.any(m_chi <= 0):
    #    raise ValueError("C_FI_decay_MB requires m_chi > 0.")
    #if torch.any(m_h <= 2.0 * m_chi):
    #    raise ValueError("Need m_h > 2 m_chi for h -> chi chi.")
    #if torch.any(T <= 0):
    #    raise ValueError("Temperature T must be > 0.")
    #if torch.any(p <= 0):
    #    raise ValueError("This massive-daughter formula should be used for p > 0.")

    E = torch.sqrt(p * p + m_chi * m_chi)
    beta_chi = torch.sqrt(1.0 - 4.0 * m_chi**2 / m_h**2)
    Delta = torch.sqrt(m_h**2 - 4.0 * m_chi**2)

    E_h_min = (m_h**2 * E - m_h * p * Delta) / (2.0 * m_chi**2)

    C = (
        pref
        * (g_trilinear**2 / (32.0 * torch.pi))
        * beta_chi
        * T / (E * p)
        * torch.exp(-E_h_min / T)
    )
    return C

@torch.no_grad()
def rhs_df_da_torch_logq(
    f, a,
    q, logq0, dlogq, log_space,
    m_chi, lam,
    H_of_a,
    T_of_a=None,
    m_h=None,
    g_trilinear=None,
    pref_FI=1.0,
    #gchi=1.0,
    batch_size=16,
    Ng=32,
    apply_conservation_projection=True,
    return_diagnostics=False,   # kept only for interface compatibility
):
    """
    RK4-compatible RHS:
        df/da = (C_self + C_FI) / (a H)

    Returns only df/da.
    """

    C_self, E, p, dp_vec = C_self_torch_logq(
        f, a,
        q, logq0, dlogq, log_space,
        m_chi, lam,
        Ng=Ng,
        batch_size=batch_size,
        apply_conservation_projection=apply_conservation_projection,
        return_diagnostics=False,
    )

    C_tot = C_self
    
    #C_tot = 0 #REMOVE LATER!!!!
    
    p = q/a

    if T_of_a is not None:
        if m_h is None or g_trilinear is None:
            raise ValueError("For FI source you must provide both m_h and g_trilinear.")

        Tt = torch.as_tensor(float(T_of_a(float(a))), device=f.device, dtype=f.dtype)

        C_FI = C_FI_decay_MB(
            p=p,
            T=Tt,
            m_chi=m_chi,
            m_h=m_h,
            g_trilinear=g_trilinear,
            pref=pref_FI,
        )

        C_tot = C_tot + C_FI

    Ht = torch.as_tensor(float(H_of_a(float(a))), device=f.device, dtype=f.dtype)
    a_t = torch.as_tensor(a, device=f.device, dtype=f.dtype)

    return C_tot / (a_t * Ht)
    
    
@torch.no_grad()
def rhs_df_da_torch_logq_FI(
    f, a,
    q, logq0, dlogq, log_space,
    m_chi,
    H_of_a,
    T_of_a=None,
    m_h=None,
    g_trilinear=None,
    pref_FI=1.0,
    #gchi=1.0,
    batch_size=16,
    Ng=32,
    apply_conservation_projection=True,
    return_diagnostics=False,   # kept only for interface compatibility
):
    """
    RK4-compatible RHS:
        df/da = (C_self + C_FI) / (a H)

    Returns only df/da.
    """
    
    C_tot = 0
    
    p = q/a

    if T_of_a is not None:
        if m_h is None or g_trilinear is None:
            raise ValueError("For FI source you must provide both m_h and g_trilinear.")

        Tt = torch.as_tensor(float(T_of_a(float(a))), device=f.device, dtype=f.dtype)

        C_FI = C_FI_decay_MB(
            p=p,
            T=Tt,
            m_chi=m_chi,
            m_h=m_h,
            g_trilinear=g_trilinear,
            pref=pref_FI,
        )

        C_tot = C_tot + C_FI

    Ht = torch.as_tensor(float(H_of_a(float(a))), device=f.device, dtype=f.dtype)
    a_t = torch.as_tensor(a, device=f.device, dtype=f.dtype)

    return C_tot / (a_t * Ht)
