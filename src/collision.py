# src/collision_log_linearized.py

import numpy as np
import torch

from functools import partial

from grid_log import grid_edges_from_centers_log
from thermodynamics import N_eq, Gamma_htophiphi, Gamma_htophiphi_energy
from cBE_solver import solve_free_in_loga, make_a_functions_from_loga_solution


# ============================================================
# Cached cBE target
# ============================================================

_CBE_READY = False
_CBE_NS = None
_CBE_TS = None


def init_mb_target_from_cbe(
    cosmo,
    ms,
    lhs,
    xi_inf,
    ai=1.0,
    af=1.0e3,
    mh=125.0,
    v=246.0,
    nX_of_a=None,
    Gamma_X=None,
    mX=None,
    m_other=0.0,
    multiplicity_X=1.0,
    rtol=1e-8,
    atol=1e-10,
    method="Radau",
):
    global _CBE_READY, _CBE_NS, _CBE_TS

    sol = solve_free_in_loga(
        cosmo=cosmo,
        ms=ms,
        lhs=lhs,
        xi_inf=xi_inf,
        ai=ai,
        af=af,
        mh=mh,
        v=v,
        nX_of_a=nX_of_a,
        Gamma_X=Gamma_X,
        mX=mX,
        m_other=m_other,
        multiplicity_X=multiplicity_X,
        rtol=rtol,
        atol=atol,
        method=method,
    )

    _CBE_NS, _CBE_TS = make_a_functions_from_loga_solution(sol)
    _CBE_READY = True
    return sol


def clear_mb_target_cache():
    global _CBE_READY, _CBE_NS, _CBE_TS
    _CBE_READY = False
    _CBE_NS = None
    _CBE_TS = None


def get_cached_cbe_targets():
    if not _CBE_READY:
        raise RuntimeError("Call init_mb_target_from_cbe(...) first.")
    return _CBE_NS, _CBE_TS


@torch.no_grad()
def f_MB_cbe_q(q, a, m):
    if not _CBE_READY:
        raise RuntimeError("Call init_mb_target_from_cbe(...) first.")

    q = torch.as_tensor(q)
    device, dtype = q.device, q.dtype

    a_t = torch.as_tensor(a, device=device, dtype=dtype)
    m_t = torch.as_tensor(m, device=device, dtype=dtype)

    N_val = float(_CBE_NS(float(a_t)))
    T_val = float(_CBE_TS(float(a_t)))
    Neq_val = float(N_eq(float(a_t), T_val, float(m_t)))
    z_val = N_val / max(Neq_val, 1e-300)

    z_t = torch.as_tensor(z_val, device=device, dtype=dtype)
    T_t = torch.as_tensor(T_val, device=device, dtype=dtype)

    p = q / a_t
    E = torch.sqrt(p * p + m_t * m_t)
    return z_t * torch.exp(-E / T_t)


# ============================================================
# Basic utilities
# ============================================================

def two_body_energy_at_rest(m_parent, m_daughter, m_other=0.0):
    return (m_parent**2 + m_daughter**2 - m_other**2) / (2.0 * m_parent)


def energy(p, m):
    return torch.sqrt(p * p + m * m)


def _leggauss_torch(Ng, device, dtype):
    x, w = np.polynomial.legendre.leggauss(Ng)
    return (
        torch.as_tensor(x, device=device, dtype=dtype),
        torch.as_tensor(w, device=device, dtype=dtype),
    )


@torch.no_grad()
def number_density_from_f_logq(f, q, a, gchi=1.0):
    f = torch.as_tensor(f)
    device, dtype = f.device, f.dtype
    q = torch.as_tensor(q, device=device, dtype=dtype)
    a_t = torch.as_tensor(a, device=device, dtype=dtype)
    gchi_t = torch.as_tensor(gchi, device=device, dtype=dtype)

    q_edges = grid_edges_from_centers_log(q)
    p = q / a_t
    p_edges = q_edges / a_t
    dp = p_edges[1:] - p_edges[:-1]

    return (gchi_t / (2.0 * torch.pi**2)) * torch.sum(p * p * dp * f)


@torch.no_grad()
def z_eff_from_current_n_on_grid(q_grid, a, m, n_current, T_target, gchi=1.0):
    """
    Compute z_eff so that z_eff * exp(-E/T_target), evaluated on the monotonic
    solver grid q_grid, has physical number density n_current.
    """
    q_grid = torch.as_tensor(q_grid)
    device, dtype = q_grid.device, q_grid.dtype

    a_t = torch.as_tensor(a, device=device, dtype=dtype)
    m_t = torch.as_tensor(m, device=device, dtype=dtype)
    n_current = torch.as_tensor(n_current, device=device, dtype=dtype)
    T_t = torch.as_tensor(T_target, device=device, dtype=dtype)
    gchi_t = torch.as_tensor(gchi, device=device, dtype=dtype)

    p = q_grid / a_t
    E = torch.sqrt(p * p + m_t * m_t)
    base = torch.exp(-E / T_t)

    q_edges = grid_edges_from_centers_log(q_grid)
    p_edges = q_edges / a_t
    dp = p_edges[1:] - p_edges[:-1]

    n_base = (gchi_t / (2.0 * torch.pi**2)) * torch.sum(p * p * dp * base)
    z_eff = n_current / torch.clamp(n_base, min=1e-300)
    return z_eff


@torch.no_grad()
def f_MB_from_z_at_q(q_eval, a, m, z_eff, T_target):
    """
    Evaluate MB target at arbitrary q points. q_eval does not need to be sorted.
    """
    q_eval = torch.as_tensor(q_eval)
    device, dtype = q_eval.device, q_eval.dtype

    a_t = torch.as_tensor(a, device=device, dtype=dtype)
    m_t = torch.as_tensor(m, device=device, dtype=dtype)
    z_t = torch.as_tensor(z_eff, device=device, dtype=dtype)
    T_t = torch.as_tensor(T_target, device=device, dtype=dtype)

    p = q_eval / a_t
    E = torch.sqrt(p * p + m_t * m_t)
    return z_t * torch.exp(-E / T_t)


@torch.no_grad()
def f_MB_shape_current_n(q_grid, a, m, n_current, T_target, gchi=1.0):
    """
    Build MB target on the monotonic solver grid q_grid.
    """
    z_eff = z_eff_from_current_n_on_grid(
        q_grid=q_grid,
        a=a,
        m=m,
        n_current=n_current,
        T_target=T_target,
        gchi=gchi,
    )
    return f_MB_from_z_at_q(
        q_eval=q_grid,
        a=a,
        m=m,
        z_eff=z_eff,
        T_target=T_target,
    )


@torch.no_grad()
def check_conservation_nonuniform(C, p, E, dp_vec):
    C = torch.as_tensor(C)
    p = torch.as_tensor(p, device=C.device, dtype=C.dtype)
    E = torch.as_tensor(E, device=C.device, dtype=C.dtype)
    dp_vec = torch.as_tensor(dp_vec, device=C.device, dtype=C.dtype)

    w = p * p * dp_vec
    I_number = torch.sum(w * C)
    I_energy = torch.sum(w * E * C)

    scale_number = torch.sum(w * torch.abs(C)) + 1e-300
    scale_energy = torch.sum(w * E * torch.abs(C)) + 1e-300

    return I_energy, I_number, I_energy / scale_energy, I_number / scale_number


# ============================================================
# Moment targets for full RHS
# ============================================================

def make_dndt_target_of_a(T_of_a, ms, lhs, mh, v,
                          nX_of_a=None, Gamma_X=None, multiplicity_X=1.0):
    def dndt_target_of_a(a):
        T = float(T_of_a(a)) if T_of_a is not None else 0.0
        src1 = (lhs**2) * Gamma_htophiphi(T, ms, mh=mh, v=v) if T_of_a is not None else 0.0

        src2 = 0.0
        if nX_of_a is not None and Gamma_X is not None:
            src2 = multiplicity_X * Gamma_X * float(nX_of_a(a))

        return src1 + src2
    return dndt_target_of_a


def make_drhodt_target_of_a(T_of_a, ms, lhs, mh, v,
                            nX_of_a=None, Gamma_X=None, mX=None,
                            m_other=0.0, multiplicity_X=1.0):
    def drhodt_target_of_a(a):
        T = float(T_of_a(a)) if T_of_a is not None else 0.0
        src1 = (lhs**2) * Gamma_htophiphi_energy(T, ms, mh=mh, v=v) if T_of_a is not None else 0.0

        src2 = 0.0
        if nX_of_a is not None and Gamma_X is not None and mX is not None:
            E0 = two_body_energy_at_rest(mX, ms, m_other=m_other)
            src2 = multiplicity_X * E0 * Gamma_X * float(nX_of_a(a))

        return src1 + src2
    return drhodt_target_of_a


# ============================================================
# Projection to target moments
# ============================================================

@torch.no_grad()
def enforce_to_target_moments_weighted(
    C_self_raw,
    C_src,
    f, p, E, dp,
    dndt_target,
    drhodt_target,
    gchi=1.0,
    support_rel=1e-12,
    tail_power=1.0,
    floor_rel=1e-30,
):
    C = torch.as_tensor(C_self_raw)
    C_src = torch.as_tensor(C_src, device=C.device, dtype=C.dtype)
    f = torch.as_tensor(f, device=C.device, dtype=C.dtype)
    p = torch.as_tensor(p, device=C.device, dtype=C.dtype)
    E = torch.as_tensor(E, device=C.device, dtype=C.dtype)
    dp_t = torch.as_tensor(dp, device=C.device, dtype=C.dtype)
    gchi = torch.as_tensor(gchi, device=C.device, dtype=C.dtype)

    w = p * p * dp_t

    target_N = (2.0 * torch.pi**2 / gchi) * torch.as_tensor(dndt_target, device=C.device, dtype=C.dtype)
    target_rho = (2.0 * torch.pi**2 / gchi) * torch.as_tensor(drhodt_target, device=C.device, dtype=C.dtype)

    C_tot_raw = C_src + C
    R_N = torch.sum(w * C_tot_raw) - target_N
    R_rho = torch.sum(w * E * C_tot_raw) - target_rho

    fmax = torch.max(f).clamp(min=torch.finfo(f.dtype).tiny)
    f_rel = f / fmax
    support = f_rel > support_rel

    if torch.count_nonzero(support) < 2:
        return C

    inv_pen = torch.where(
        support,
        torch.clamp(f_rel, min=floor_rel) ** tail_power,
        torch.zeros_like(f_rel),
    )

    A00 = torch.sum(w * inv_pen)
    A01 = torch.sum(w * E * inv_pen)
    A11 = torch.sum(w * E * E * inv_pen)

    det = A00 * A11 - A01 * A01
    tiny = 100.0 * torch.finfo(C.dtype).eps
    if torch.abs(det) <= tiny * (torch.abs(A00 * A11) + 1.0):
        return C

    alpha = (R_N * A11 - R_rho * A01) / det
    beta = (R_rho * A00 - R_N * A01) / det

    return C - inv_pen * (alpha + beta * E)


@torch.no_grad()
def enforce_self_zero_moments_weighted(
    C_self_raw,
    f,
    p,
    E,
    dp,
    support_rel=1e-12,
    tail_power=1.0,
    floor_rel=1e-30,
):
    C = torch.as_tensor(C_self_raw)
    f = torch.as_tensor(f, device=C.device, dtype=C.dtype)
    p = torch.as_tensor(p, device=C.device, dtype=C.dtype)
    E = torch.as_tensor(E, device=C.device, dtype=C.dtype)
    dp = torch.as_tensor(dp, device=C.device, dtype=C.dtype)

    w = p * p * dp

    R_N = torch.sum(w * C)
    R_rho = torch.sum(w * E * C)

    fmax = torch.max(torch.abs(f)).clamp(min=torch.finfo(C.dtype).tiny)
    f_rel = torch.abs(f) / fmax
    support = f_rel > support_rel

    if torch.count_nonzero(support) < 2:
        return C

    inv_pen = torch.where(
        support,
        torch.clamp(f_rel, min=floor_rel) ** tail_power,
        torch.zeros_like(f_rel),
    )

    A00 = torch.sum(w * inv_pen)
    A01 = torch.sum(w * E * inv_pen)
    A11 = torch.sum(w * E * E * inv_pen)

    det = A00 * A11 - A01 * A01
    tiny = 100.0 * torch.finfo(C.dtype).eps

    if torch.abs(det) <= tiny * (torch.abs(A00 * A11) + 1.0):
        return C

    alpha = (R_N * A11 - R_rho * A01) / det
    beta = (R_rho * A00 - R_N * A01) / det

    return C - inv_pen * (alpha + beta * E)


# ============================================================
# Source terms
# ============================================================

@torch.no_grad()
def C_FI_decay_MB(
    p, T,
    m_chi, m_h, g_trilinear,
    pref=1.0,
):
    p = torch.as_tensor(p)
    device, dtype = p.device, p.dtype

    T = torch.as_tensor(T, device=device, dtype=dtype)
    m_chi = torch.as_tensor(m_chi, device=device, dtype=dtype)
    m_h = torch.as_tensor(m_h, device=device, dtype=dtype)
    g_trilinear = torch.as_tensor(g_trilinear, device=device, dtype=dtype)
    pref = torch.as_tensor(pref, device=device, dtype=dtype)

    E = torch.sqrt(p * p + m_chi * m_chi)
    Delta = torch.sqrt(m_h * m_h - 4.0 * m_chi * m_chi)

    E_h_minus = (m_h * m_h * E - m_h * p * Delta) / (2.0 * m_chi * m_chi)
    E_h_plus = (m_h * m_h * E + m_h * p * Delta) / (2.0 * m_chi * m_chi)

    dE = E_h_plus - E_h_minus
    diff = torch.exp(-E_h_minus / T) * (-torch.expm1(-dE / T))

    C = pref * (g_trilinear**2 / (8.0 * torch.pi)) * T / (E * p) * diff
    return C


@torch.no_grad()
def deposit_mono_source_logq(
    q_grid, a, m_chi,
    p0, E0,
    number_rate_phys,
    gchi=1.0,
):
    q_grid = torch.as_tensor(q_grid)
    device, dtype = q_grid.device, q_grid.dtype

    a = torch.as_tensor(a, device=device, dtype=dtype)
    m_chi = torch.as_tensor(m_chi, device=device, dtype=dtype)
    p0 = torch.as_tensor(p0, device=device, dtype=dtype)
    E0 = torch.as_tensor(E0, device=device, dtype=dtype)
    number_rate_phys = torch.as_tensor(number_rate_phys, device=device, dtype=dtype)
    gchi = torch.as_tensor(gchi, device=device, dtype=dtype)

    q0 = a * p0
    q_edges = grid_edges_from_centers_log(q_grid)
    dq = q_edges[1:] - q_edges[:-1]
    C = torch.zeros_like(q_grid)

    if (q0 < q_grid[0]) or (q0 > q_grid[-1]):
        return C

    jR = torch.searchsorted(q_grid, q0, right=False)
    jR = torch.clamp(jR, 1, q_grid.numel() - 1)
    jL = jR - 1

    qL = q_grid[jL]
    qR = q_grid[jR]

    EL = torch.sqrt((qL / a) ** 2 + m_chi ** 2)
    ER = torch.sqrt((qR / a) ** 2 + m_chi ** 2)

    denom = ER - EL
    tiny = 10.0 * torch.finfo(dtype).eps

    if torch.abs(denom) <= tiny:
        wR = torch.tensor(0.5, device=device, dtype=dtype)
    else:
        wR = (E0 - EL) / denom
        wR = torch.clamp(wR, 0.0, 1.0)

    wL = 1.0 - wR
    target_q_moment = (2.0 * torch.pi**2 / gchi) * number_rate_phys

    C[jL] += target_q_moment * wL / (((qL / a) ** 2) * (dq[jL] / a))
    C[jR] += target_q_moment * wR / (((qR / a) ** 2) * (dq[jR] / a))
    return C


@torch.no_grad()
def C_FI_decay_condensate_logq(
    q, a,
    m_chi,
    m_parent,
    n_parent_of_a,
    Gamma_parent,
    m_other=0.0,
    multiplicity=1.0,
    gchi=1.0,
):
    q = torch.as_tensor(q)
    device, dtype = q.device, q.dtype

    a = torch.as_tensor(a, device=device, dtype=dtype)
    m_chi = torch.as_tensor(m_chi, device=device, dtype=dtype)
    m_parent = torch.as_tensor(m_parent, device=device, dtype=dtype)
    m_other = torch.as_tensor(m_other, device=device, dtype=dtype)
    Gamma_parent = torch.as_tensor(Gamma_parent, device=device, dtype=dtype)
    multiplicity = torch.as_tensor(multiplicity, device=device, dtype=dtype)

    n_parent = torch.as_tensor(float(n_parent_of_a(float(a))), device=device, dtype=dtype)

    E0 = (m_parent**2 + m_chi**2 - m_other**2) / (2.0 * m_parent)
    p0sq = E0**2 - m_chi**2
    if p0sq <= 0:
        return torch.zeros_like(q)

    p0 = torch.sqrt(p0sq)
    number_rate_phys = multiplicity * Gamma_parent * n_parent

    return deposit_mono_source_logq(
        q_grid=q,
        a=a,
        m_chi=m_chi,
        p0=p0,
        E0=E0,
        number_rate_phys=number_rate_phys,
        gchi=gchi,
    )


# ============================================================
# 2->2 kernel
# ============================================================

@torch.no_grad()
def _F_kernel_batch_constant_M2_vec(pi, Ei, pn, En, pm, Em, m, lam, Ng, eps_disc=0.0):
    device, dtype = pi.device, pi.dtype
    mu2, w2 = _leggauss_torch(Ng, device, dtype)

    lam2 = torch.as_tensor(lam, device=device, dtype=dtype) ** 2
    m2 = torch.as_tensor(m, device=device, dtype=dtype) ** 2

    pref = 1.0 / (4.0 * (2.0 * torch.pi) ** 4)

    c2 = mu2.view(Ng, 1, 1, 1)
    w2 = w2.view(Ng, 1, 1, 1)

    pi = pi.unsqueeze(0)
    Ei = Ei.unsqueeze(0)
    pn = pn.unsqueeze(0)
    En = En.unsqueeze(0)
    pm = pm.unsqueeze(0)
    Em = Em.unsqueeze(0)

    pnpm = pn * pm
    s2sq = 1.0 - c2 * c2

    A0_base = m2 + En * Em - En * Ei - Em * Ei
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

    c3min = torch.clamp(rmin, min=-1.0, max=1.0)
    c3max = torch.clamp(rmax, min=-1.0, max=1.0)
    mask_int = mask_disc & (c3max > c3min)

    sqrt_minus_a = torch.sqrt(torch.clamp(-a, min=0.0))
    inv_sqrt_disc = torch.where(mask_int, 1.0 / sqrt_disc, torch.zeros_like(sqrt_disc))

    umin = torch.clamp((2.0 * a * c3min + b) * inv_sqrt_disc, -1.0, 1.0)
    umax = torch.clamp((2.0 * a * c3max + b) * inv_sqrt_disc, -1.0, 1.0)

    I = torch.where(
        mask_int,
        (torch.asin(umin) - torch.asin(umax)) /
        torch.where(mask_int, sqrt_minus_a, torch.ones_like(sqrt_minus_a)),
        torch.zeros_like(disc),
    )

    contrib = lam2 * pnpm * I
    Facc = torch.sum(w2 * contrib, dim=0)
    return torch.clamp(pref * Facc, min=0.0)


# ============================================================
# Self-collision
# ============================================================

@torch.no_grad()
def _C_self_torch_logq_impl(
    f, a,
    q,
    m, lam,
    Ng=16, batch_size=16,
    return_diagnostics=False,
):
    f = torch.as_tensor(f)
    device, dtype = f.device, f.dtype

    q = torch.as_tensor(q, device=device, dtype=dtype)
    a = torch.as_tensor(a, device=device, dtype=dtype)
    m = torch.as_tensor(m, device=device, dtype=dtype)

    N = q.numel()
    p = q / a
    E = torch.sqrt(p * p + m * m)

    q_edges = grid_edges_from_centers_log(q)
    p_edges = q_edges / a
    dp_vec = p_edges[1:] - p_edges[:-1]

    dp_n = dp_vec.view(1, N, 1)
    dp_m = dp_vec.view(1, 1, N)

    f_n = f.view(1, N, 1)
    f_m = f.view(1, 1, N)

    pn = p.view(1, N, 1)
    pm = p.view(1, 1, N)
    En = E.view(1, N, 1)
    Em = E.view(1, 1, N)

    # Symmetry factor for identical particles in the integrated pair.
    pref0 = 0.5
    C = torch.empty_like(f)

    if return_diagnostics:
        total_valid = torch.zeros((), device=device, dtype=dtype)
        total_phys = torch.zeros((), device=device, dtype=dtype)
        total_outside_weight = torch.zeros((), device=device, dtype=dtype)
        total_all_weight = torch.zeros((), device=device, dtype=dtype)

    w_nm = dp_n * dp_m

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ib = torch.arange(start, end, device=device)

        pi = p[ib].view(-1, 1, 1)
        Ei = E[ib].view(-1, 1, 1)

        # ------------------------------------------------------------
        # Relativistic energy-conserving fourth momentum:
        #
        #     E_i + E_tilde = E_n + E_m
        #     E_tilde = E_n + E_m - E_i
        #     p_tilde^2 = E_tilde^2 - m^2
        #
        # This replaces the nonrelativistic approximation
        # p_tilde^2 = p_n^2 + p_m^2 - p_i^2.
        # ------------------------------------------------------------
        Etil = En + Em - Ei
        phys_ptil = Etil >= m

        ptil2 = Etil * Etil - m * m
        ptil = torch.sqrt(torch.clamp(ptil2, min=0.0))
        qtil = a * ptil

        inside_q = (qtil >= q[0]) & (qtil <= q[-1])

        ftil = torch.zeros_like(qtil)
        
        
        qv_safe = torch.clamp(qtil, min=q[0], max=q[-1])#This avoids if evaliation
        jR = torch.searchsorted(q, qv_safe, right=False)
        jR = torch.clamp(jR, 1, q.numel() - 1)
        jL = jR - 1
        qL = q[jL]
        qR = q[jR]
        
        EL = torch.sqrt((qL / a) ** 2 + m ** 2)
        ER = torch.sqrt((qR / a) ** 2 + m ** 2)
        EQ = torch.sqrt((qv_safe / a) ** 2 + m ** 2)
        denom = ER - EL
        tiny = 10.0 * torch.finfo(dtype).eps
        
        wR = torch.where(
            torch.abs(denom) > tiny,
            (EQ - EL) / denom,
            torch.zeros_like(EQ),)
        
        wR = torch.clamp(wR, 0.0, 1.0)
        wL = 1.0 - wR
        
        # Log-linear interpolation in energy preserves detailed balance for
        # a Maxwell-Boltzmann shape, log f = const - E/T, much better than
        # linear interpolation of f itself.
        f_floor = torch.clamp(
            torch.max(torch.abs(f)) * 1e-300,
            min=torch.finfo(dtype).tiny,
        )
        logfL = torch.log(torch.clamp(f[jL], min=f_floor))
        logfR = torch.log(torch.clamp(f[jR], min=f_floor))
        ftil_interp = torch.exp(wL * logfL + wR * logfR)
        
        ftil = torch.where(
            inside_q,
            ftil_interp,
            torch.zeros_like(ftil_interp),
            )

        F = _F_kernel_batch_constant_M2_vec(
            pi, Ei, pn, En, pm, Em, m, lam, Ng, eps_disc=0.0
        )

        fi = f[ib].view(-1, 1, 1)
        valid = phys_ptil & inside_q

        gain_loss = torch.where(
            valid,
            (f_n * f_m) - fi * ftil,
            torch.zeros_like(ftil),
        )

        # ------------------------------------------------------------
        # Convention correction:
        #
        # The self-scattering integral constructed here follows the
        # paper convention:
        #
        #     E_i df_i/dt = C_i
        #
        # but rhs_df_da_torch_logq_generic expects a source in the
        # same convention as freeze-in:
        #
        #     df_i/dt
        #
        # Therefore divide by E_i before returning C.
        # ------------------------------------------------------------
        C_batch_paper = pref0 * torch.sum(w_nm * F * gain_loss, dim=(1, 2))
        C[start:end] = C_batch_paper / E[ib]

        if return_diagnostics:
            total_valid = total_valid + valid.to(dtype).sum()
            total_phys = total_phys + phys_ptil.to(dtype).sum()

            outside = phys_ptil & (~inside_q)
            W = F.abs() * (torch.abs(f_n * f_m) + torch.abs(fi * ftil))
            total_outside_weight = total_outside_weight + (
                W.masked_fill(~outside, 0.0)
            ).sum()
            total_all_weight = total_all_weight + W.sum()

    if return_diagnostics:
        I_energy_raw, I_number_raw, rE_raw, rN_raw = check_conservation_nonuniform(
            C, p, E, dp_vec
        )
    
    C = enforce_self_zero_moments_weighted(
        C_self_raw=C,
        f=f,
        p=p,
        E=E,
        dp=dp_vec,) 

    if return_diagnostics:
        I_energy, I_number, rE, rN = check_conservation_nonuniform(
            C, p, E, dp_vec
        )

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

#C_self_torch_logq = torch.compile(_C_self_torch_logq_impl)

def C_self_torch_logq(*args, **kwargs):
    if kwargs.get("return_diagnostics", False):
        return _C_self_torch_logq_impl(*args, **kwargs)
    return _C_self_torch_logq_compiled(*args, **kwargs)

_C_self_torch_logq_compiled = torch.compile(_C_self_torch_logq_impl)

@torch.no_grad()
def C_self_torch_logq_conservative_impl(
    f, a,
    q,
    m, lam,
    Ng=16, batch_size=16,
    return_diagnostics=False,
    f_floor_rel=1e-300,
):
    """
    Maximally conservative self-scattering operator for the log-q grid.

    It implements each accepted microscopic channel as a conservative
    discrete event

        i + j_tilde <-> n + m,

    where j_tilde is fixed by relativistic energy conservation,

        E_j_tilde = E_n + E_m - E_i.

    The fourth leg j_tilde is deposited between neighboring grid bins jL,jR
    using energy-linear weights, so that both the number moment and the
    energy moment are conserved by construction.

    The returned C is in the df/dt convention expected by
    rhs_df_da_torch_logq_generic, not in the paper convention E df/dt.

    Main difference from the previous conservative_scatter version:
      - accumulate directly in Cw = p^2 dp C, i.e. in the number-moment
        variable, instead of repeatedly dividing by p^2 dp during scatter_add.
      - use an energy-linear f_tilde consistent with the same weights used
        for conservative deposition.
      - the 1/E_i factor is applied when forming the event number amount Q,
        not as a post-processing division of the full operator.
    """

    f = torch.as_tensor(f)
    device, dtype = f.device, f.dtype

    q = torch.as_tensor(q, device=device, dtype=dtype)
    a = torch.as_tensor(a, device=device, dtype=dtype)
    m = torch.as_tensor(m, device=device, dtype=dtype)

    N = q.numel()

    p = q / a
    E = torch.sqrt(p * p + m * m)

    q_edges = grid_edges_from_centers_log(q)
    p_edges = q_edges / a
    dp_vec = p_edges[1:] - p_edges[:-1]

    # Moment weight used by diagnostics:
    #
    #   N-moment:   sum_k p_k^2 dp_k C_k
    #   rho-moment: sum_k p_k^2 dp_k E_k C_k
    #
    w_grid = p * p * dp_vec
    w_safe = torch.clamp(w_grid, min=torch.finfo(dtype).tiny)

    # We accumulate Cw_k = w_k C_k directly.
    # This is the conservative moment variable.
    Cw = torch.zeros_like(f)

    dp_n = dp_vec.view(1, N, 1)
    dp_m = dp_vec.view(1, 1, N)
    w_nm = dp_n * dp_m

    f_n = f.view(1, N, 1)
    f_m = f.view(1, 1, N)

    pn = p.view(1, N, 1)
    pm = p.view(1, 1, N)
    En = E.view(1, N, 1)
    Em = E.view(1, 1, N)

    pref0 = 0.5

    tiny = 10.0 * torch.finfo(dtype).eps
    f_floor = torch.clamp(
        torch.max(torch.abs(f)) * f_floor_rel,
        min=torch.finfo(dtype).tiny,
    )

    # Fixed n,m indices.
    n_idx_2d = torch.arange(N, device=device).view(1, N, 1)
    m_idx_2d = torch.arange(N, device=device).view(1, 1, N)

    if return_diagnostics:
        total_valid = torch.zeros((), device=device, dtype=dtype)
        total_phys = torch.zeros((), device=device, dtype=dtype)
        total_outside_weight = torch.zeros((), device=device, dtype=dtype)
        total_all_weight = torch.zeros((), device=device, dtype=dtype)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start

        ib = torch.arange(start, end, device=device)

        pi = p[ib].view(B, 1, 1)
        Ei = E[ib].view(B, 1, 1)
        fi = f[ib].view(B, 1, 1)

        # ------------------------------------------------------------
        # Fourth leg fixed by exact relativistic energy conservation:
        #
        #   E_jtilde = E_n + E_m - E_i
        #   p_jtilde^2 = E_jtilde^2 - m^2
        # ------------------------------------------------------------
        Etil = En + Em - Ei
        phys_ptil = Etil >= m

        ptil2 = Etil * Etil - m * m
        ptil = torch.sqrt(torch.clamp(ptil2, min=0.0))
        qtil = a * ptil

        inside_q = (qtil >= q[0]) & (qtil <= q[-1])
        valid = phys_ptil & inside_q

        # ------------------------------------------------------------
        # Bracketing bins for j_tilde.
        #
        # We clamp qtil only for indexing. Invalid channels are still
        # removed by valid below.
        # ------------------------------------------------------------
        qtil_flat = torch.clamp(qtil.reshape(-1), min=q[0], max=q[-1])

        jR = torch.searchsorted(q, qtil_flat, right=False)
        jR = torch.clamp(jR, 1, N - 1)
        jL = jR - 1

        qL = q[jL]
        qR = q[jR]

        EL = torch.sqrt((qL / a) ** 2 + m ** 2)
        ER = torch.sqrt((qR / a) ** 2 + m ** 2)

        Etil_flat = Etil.reshape(-1)

        denom = ER - EL
        wR_flat = torch.where(
            torch.abs(denom) > tiny,
            (Etil_flat - EL) / denom,
            torch.zeros_like(Etil_flat),
        )
        wR_flat = torch.clamp(wR_flat, 0.0, 1.0)
        wL_flat = 1.0 - wR_flat

        # Energy-linear interpolation of f_jtilde.
        #
        # This is deliberately not log-linear. The same weights used
        # here are also used for deposition. That makes the event
        # bookkeeping internally consistent.
        fL = torch.clamp(f[jL], min=f_floor)
        fR = torch.clamp(f[jR], min=f_floor)
        ftil_flat = wL_flat * fL + wR_flat * fR
        ftil = ftil_flat.view(B, N, N)

        # ------------------------------------------------------------
        # Dense angular kernel.
        # ------------------------------------------------------------
        F = _F_kernel_batch_constant_M2_vec(
            pi, Ei, pn, En, pm, Em, m, lam, Ng, eps_disc=0.0
        )

        # Positive gain_loss means n+m -> i+jtilde.
        # Negative gain_loss means i+jtilde -> n+m.
        gain_loss = torch.where(
            valid,
            (f_n * f_m) - fi * ftil,
            torch.zeros_like(ftil),
        )

        # ------------------------------------------------------------
        # rate_i is in paper convention:
        #
        #   E_i df_i/dt = rate_i.
        #
        # The actual number-moment event amount is therefore
        #
        #   Q = w_i * rate_i / E_i.
        #
        # We accumulate Q directly into Cw = w C.
        # ------------------------------------------------------------
        rate_i_paper = pref0 * w_nm * F * gain_loss

        valid_flat = valid.reshape(-1)
        if not torch.any(valid_flat):
            if return_diagnostics:
                total_phys = total_phys + phys_ptil.to(dtype).sum()
                Wdiag = F.abs() * (torch.abs(f_n * f_m) + torch.abs(fi * ftil))
                outside = phys_ptil & (~inside_q)
                total_outside_weight = total_outside_weight + (
                    Wdiag.masked_fill(~outside, 0.0)
                ).sum()
                total_all_weight = total_all_weight + Wdiag.sum()
            continue

        rate_flat = rate_i_paper.reshape(-1)[valid_flat]

        i_idx = ib.view(B, 1, 1).expand(B, N, N).reshape(-1)[valid_flat]
        n_idx = n_idx_2d.expand(B, N, N).reshape(-1)[valid_flat]
        m_idx = m_idx_2d.expand(B, N, N).reshape(-1)[valid_flat]

        jL_valid = jL[valid_flat]
        jR_valid = jR[valid_flat]
        wL_valid = wL_flat[valid_flat]
        wR_valid = wR_flat[valid_flat]

        Ei_valid = E[i_idx]
        wi_valid = w_grid[i_idx]

        # Conservative number-moment event amount.
        Q = wi_valid * rate_flat / Ei_valid

        # ------------------------------------------------------------
        # Conservative event deposition in Cw = w C:
        #
        #   +Q into i
        #   +Q into j_tilde, split energy-linearly into jL,jR
        #   -Q into n
        #   -Q into m
        #
        # Number:
        #   +Q + Q - Q - Q = 0.
        #
        # Energy:
        #   Q E_i + Q E_jtilde - Q E_n - Q E_m = 0,
        #
        # because E_jtilde is represented by
        #
        #   wL E_jL + wR E_jR.
        # ------------------------------------------------------------
        Cw.scatter_add_(0, i_idx, Q)
        Cw.scatter_add_(0, n_idx, -Q)
        Cw.scatter_add_(0, m_idx, -Q)
        Cw.scatter_add_(0, jL_valid, wL_valid * Q)
        Cw.scatter_add_(0, jR_valid, wR_valid * Q)

        if return_diagnostics:
            total_valid = total_valid + valid.to(dtype).sum()
            total_phys = total_phys + phys_ptil.to(dtype).sum()

            outside = phys_ptil & (~inside_q)
            Wdiag = F.abs() * (torch.abs(f_n * f_m) + torch.abs(fi * ftil))
            total_outside_weight = total_outside_weight + (
                Wdiag.masked_fill(~outside, 0.0)
            ).sum()
            total_all_weight = total_all_weight + Wdiag.sum()

    # Convert from moment variable Cw = w C to C = df/dt.
    C = Cw / w_safe

    if return_diagnostics:
        I_energy_raw, I_number_raw, rE_raw, rN_raw = check_conservation_nonuniform(
            C, p, E, dp_vec
        )

    # Numerical fallback for any residual from boundary truncation,
    # float32 summation, or invalid channels.
    C = enforce_self_zero_moments_weighted(
        C_self_raw=C,
        f=f,
        p=p,
        E=E,
        dp=dp_vec,
    )

    if return_diagnostics:
        I_energy, I_number, rE, rN = check_conservation_nonuniform(
            C, p, E, dp_vec
        )

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

#C_self_torch_logq_conservative_scatter = torch.compile(C_self_torch_logq_conservative_impl)

_C_self_torch_logq_conservative_scatter_compiled = torch.compile(
    C_self_torch_logq_conservative_impl
)

def C_self_torch_logq_conservative_scatter(*args, **kwargs):
    if kwargs.get("return_diagnostics", False):
        return C_self_torch_logq_conservative_impl(*args, **kwargs)

    return _C_self_torch_logq_conservative_scatter_compiled(*args, **kwargs)

# ============================================================
# Generic RHS with optional moment projection
# ============================================================

@torch.no_grad()
def rhs_df_da_torch_logq_generic(
    f, a,
    q,
    m_chi,
    H_of_a,
    C_self_func=None,
    T_of_a=None,
    m_h=None,
    g_trilinear=None,
    n_parent2_of_a=None,
    Gamma_parent2=None,
    m_h2=None,
    m_other2=0.0,
    multiplicity2=1.0,
    gchi=1.0,
    pref_FI=1.0,
):
    device, dtype = f.device, f.dtype
    p = q / a

    C_src = 0.0

    if T_of_a is not None and m_h is not None and g_trilinear is not None:
        Tt = torch.as_tensor(float(T_of_a(float(a))), device=device, dtype=dtype)
        C_src = C_src + C_FI_decay_MB(
            p=p,
            T=Tt,
            m_chi=m_chi,
            m_h=m_h,
            g_trilinear=g_trilinear,
            pref=pref_FI,
        )

    if n_parent2_of_a is not None and Gamma_parent2 is not None and m_h2 is not None:
        C_src = C_src + C_FI_decay_condensate_logq(
            q=q,
            a=a,
            m_chi=m_chi,
            m_parent=m_h2,
            n_parent_of_a=n_parent2_of_a,
            Gamma_parent=Gamma_parent2,
            m_other=m_other2,
            multiplicity=multiplicity2,
            gchi=gchi,
        )

    C_self = 0.0
    if C_self_func is not None:
        C_self, *_ = C_self_func(
            f=f,
            a=a,
            q=q,
            m=m_chi,
            return_diagnostics=False,
        )

    C_tot = C_src + C_self

    Ht = torch.as_tensor(float(H_of_a(float(a))), device=device, dtype=dtype)
    a_t = torch.as_tensor(a, device=device, dtype=dtype)

    return C_tot / (a_t * Ht)



# ============================================================
# FI-only RHS
# ============================================================

@torch.no_grad()
def rhs_df_da_torch_logq_FI(
    f, a,
    q,
    m_chi,
    H_of_a,
    T_of_a=None,
    m_h=None,
    g_trilinear=None,
    n_parent2_of_a=None,
    Gamma_parent2=None,
    m_h2=None,
    m_other2=0.0,
    multiplicity2=1.0,
    gchi=1.0,
    pref_FI=1.0,
):
    C_tot = 0.0
    p = q / a

    if T_of_a is not None and m_h is not None and g_trilinear is not None:
        Tt = torch.as_tensor(float(T_of_a(float(a))), device=f.device, dtype=f.dtype)
        C_tot = C_tot + C_FI_decay_MB(
            p=p,
            T=Tt,
            m_chi=m_chi,
            m_h=m_h,
            g_trilinear=g_trilinear,
            pref=pref_FI,
        )

    if n_parent2_of_a is not None and Gamma_parent2 is not None and m_h2 is not None:
        C_tot = C_tot + C_FI_decay_condensate_logq(
            q=q,
            a=a,
            m_chi=m_chi,
            m_parent=m_h2,
            n_parent_of_a=n_parent2_of_a,
            Gamma_parent=Gamma_parent2,
            m_other=m_other2,
            multiplicity=multiplicity2,
            gchi=gchi,
        )

    Ht = torch.as_tensor(float(H_of_a(float(a))), device=f.device, dtype=f.dtype)
    a_t = torch.as_tensor(a, device=f.device, dtype=f.dtype)

    return C_tot / (a_t * Ht)


# ============================================================
# Diagnostics for Gamma/H estimate
# ============================================================

def estimate_gamma_eff_from_current_f(
    f_t,
    a_star,
    q,
    m_chi,
    C_self_func,
    H_of_a,
):
    C_self_t, E_t, p_t, dp_t, diag = C_self_func(
        f=f_t,
        a=a_star,
        q=q,
        m=m_chi,
        return_diagnostics=True,
    )

    C_self_np = C_self_t.detach().cpu().numpy()
    p_np = p_t.detach().cpu().numpy()
    dp_np = dp_t.detach().cpu().numpy()
    f_np = f_t.detach().cpu().numpy()

    w_np = p_np**2 * dp_np

    num = np.sum(w_np * np.abs(C_self_np))
    den = np.sum(w_np * np.abs(f_np))

    Gamma_eff_rms = num / max(den, 1e-300)
    H_star = H_of_a(a_star)
    if torch.is_tensor(H_star):
        H_star_float = float(H_star.detach().cpu())
    else:
        H_star_float = float(H_star)

    return {
        "Gamma_eff_rms": Gamma_eff_rms,
        "H": H_star_float,
        "Gamma_over_H": Gamma_eff_rms / H_star_float,
        "diag": diag,
        "C_self": C_self_t,
    }
