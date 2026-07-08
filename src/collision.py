# src/collision.py

import numpy as np
import torch

from grid_log import grid_edges_from_centers_log


# ============================================================
# Basic utilities
# ============================================================

def two_body_energy_at_rest(m_parent, m_daughter, m_other=0.0):
    return (m_parent**2 + m_daughter**2 - m_other**2) / (2.0 * m_parent)


def _background_tensor(callback, a, device, dtype, background_torch=False):
    if background_torch:
        value = callback(a)
    else:
        value = float(callback(float(a)))

    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _leggauss_torch(Ng, device, dtype):
    x, w = np.polynomial.legendre.leggauss(Ng)
    return (
        torch.as_tensor(x, device=device, dtype=dtype),
        torch.as_tensor(w, device=device, dtype=dtype),
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
# Projection to self-collision zero moments
# ============================================================


@torch.no_grad()
def project_self_zero_moments(
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
    scale = torch.abs(A00 * A11)

    if (not torch.isfinite(det)) or scale <= 0.0 or torch.abs(det) <= tiny * scale:
        return C

    alpha = (R_N * A11 - R_rho * A01) / det
    beta = (R_rho * A00 - R_N * A01) / det

    return C - inv_pen * (alpha + beta * E)


# ============================================================
# 2->2 kernel
# ============================================================

@torch.no_grad()
def F_contact(pi, Ei, pn, En, pm, Em, m, lam, mu2, w2, eps_disc=0.0):
    device, dtype = pi.device, pi.dtype
    Ng = mu2.numel()

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


@torch.no_grad()
def F_analytical(pi, Ei, pn, En, pm, Em, m, lam, mu2=None, w2=None, eps_disc=0.0):
    del mu2, w2, eps_disc

    device, dtype = pi.device, pi.dtype

    lam2 = torch.as_tensor(lam, device=device, dtype=dtype) ** 2
    m = torch.as_tensor(m, device=device, dtype=dtype)
    m2 = m * m

    Etil = En + Em - Ei
    phys = Etil >= m

    ptil2 = Etil * Etil - m2
    ptil = torch.sqrt(torch.clamp(ptil2, min=0.0))

    Qmin_kin = torch.abs(pi - pn)
    Qmax_kin = pi + pn

    Qmin_tri = torch.abs(pm - ptil)
    Qmax_tri = pm + ptil

    Qmin = torch.maximum(Qmin_kin, Qmin_tri)
    Qmax = torch.minimum(Qmax_kin, Qmax_tri)
    dQ = torch.clamp(Qmax - Qmin, min=0.0)

    pi_safe = torch.clamp(pi, min=torch.finfo(dtype).tiny)
    pref = lam2 / (64.0 * torch.pi ** 3)
    F = pref * dQ / pi_safe

    return torch.where(phys, F, torch.zeros_like(F))


def resolve_contact_kernel_backend(kernel_backend):
    """Return the contact-kernel implementation for a public backend name.

    Both backends evaluate the same contact-interaction kernel. The analytic
    backend is the production default; the quadrature backend is kept for
    validation and for later generalizations to non-contact matrix elements.
    """
    backend = str(kernel_backend).strip().lower()
    if backend not in ("analytic", "quadrature"):
        raise ValueError(
            "Invalid kernel_backend. Allowed values are exactly "
            "'analytic' or 'quadrature'."
        )

    if backend == "analytic":
        return backend, F_analytical, False
    return backend, F_contact, True


# ============================================================
# Self-collision
# ============================================================

@torch.no_grad()
def _C_MB_impl(
    f, a,
    q,
    m, lam,
    Ng=16, batch_size=16,
    return_diagnostics=False,
    enforce_self_projection=True,
    F_func=F_analytical,
    F_needs_quadrature=False,
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

    # External-particle normalization; together with the division by E_i below this gives 1/(2E_i).
    # Symmetry factor for identical particles in the integrated pair is taken to be 1.
    pref0 = 0.5
    C = torch.empty_like(f)

    if return_diagnostics:
        total_valid = torch.zeros((), device=device, dtype=dtype)
        total_phys = torch.zeros((), device=device, dtype=dtype)
        total_outside_weight = torch.zeros((), device=device, dtype=dtype)
        total_all_weight = torch.zeros((), device=device, dtype=dtype)
    else:
        total_valid = None
        total_phys = None
        total_outside_weight = None
        total_all_weight = None

    w_nm = dp_n * dp_m
    if F_needs_quadrature:
        mu2, w2 = _leggauss_torch(Ng, device, dtype)
    else:
        mu2 = torch.empty(0, device=device, dtype=dtype)
        w2 = torch.empty(0, device=device, dtype=dtype)

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
        
        
        qv_safe = torch.clamp(qtil, min=q[0], max=q[-1])#This avoids if evaluation
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

        F = F_func(
            pi, Ei, pn, En, pm, Em, m, lam, mu2, w2, eps_disc=0.0
        )

        fi = f[ib].view(-1, 1, 1)
        valid = phys_ptil & inside_q

        gain_loss = torch.where(
            valid,
            (f_n * f_m) - fi * ftil,
            torch.zeros_like(ftil),
        )

        C[start:end] = pref0 * torch.sum(w_nm * F * gain_loss, dim=(1, 2)) / E[ib]

        if return_diagnostics:
            assert total_valid is not None
            assert total_phys is not None
            assert total_outside_weight is not None
            assert total_all_weight is not None
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
    
    if enforce_self_projection:
        C = project_self_zero_moments(
            C_self_raw=C,
            f=f,
            p=p,
            E=E,
            dp=dp_vec,
        )

    if return_diagnostics:
        assert total_valid is not None
        assert total_phys is not None
        assert total_outside_weight is not None
        assert total_all_weight is not None
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


def _C_MB_analytic_entry(
    f, a,
    q,
    m, lam,
    Ng=16, batch_size=16,
    return_diagnostics=False,
    enforce_self_projection=True,
):
    return _C_MB_impl(
        f=f,
        a=a,
        q=q,
        m=m,
        lam=lam,
        Ng=Ng,
        batch_size=batch_size,
        return_diagnostics=return_diagnostics,
        enforce_self_projection=enforce_self_projection,
        F_func=F_analytical,
        F_needs_quadrature=False,
    )


def _C_MB_quadrature_entry(
    f, a,
    q,
    m, lam,
    Ng=16, batch_size=16,
    return_diagnostics=False,
    enforce_self_projection=True,
):
    return _C_MB_impl(
        f=f,
        a=a,
        q=q,
        m=m,
        lam=lam,
        Ng=Ng,
        batch_size=batch_size,
        return_diagnostics=return_diagnostics,
        enforce_self_projection=enforce_self_projection,
        F_func=F_contact,
        F_needs_quadrature=True,
    )


def C_MB(
    f, a,
    q,
    m, lam,
    Ng=16, batch_size=16,
    return_diagnostics=False,
    enforce_self_projection=True,
    kernel_backend="analytic",
):
    backend, F_func, F_needs_quadrature = resolve_contact_kernel_backend(
        kernel_backend
    )
    if return_diagnostics:
        return _C_MB_impl(
            f=f,
            a=a,
            q=q,
            m=m,
            lam=lam,
            Ng=Ng,
            batch_size=batch_size,
            return_diagnostics=True,
            enforce_self_projection=enforce_self_projection,
            F_func=F_func,
            F_needs_quadrature=F_needs_quadrature,
        )

    compiled = (
        C_MB_analytic_compiled
        if backend == "analytic"
        else C_MB_quadrature_compiled
    )
    return compiled(
        f=f,
        a=a,
        q=q,
        m=m,
        lam=lam,
        Ng=Ng,
        batch_size=batch_size,
        return_diagnostics=False,
        enforce_self_projection=enforce_self_projection,
    )


C_MB_analytic_compiled = torch.compile(_C_MB_analytic_entry)
C_MB_quadrature_compiled = torch.compile(_C_MB_quadrature_entry)


def _stat_eta_from_name(stat):
    key = str(stat).strip().lower().replace("-", "_")
    if key in ("classical", "mb", "maxwell_boltzmann", "maxwell", "none", "0"):
        return 0.0
    if key in ("boson", "bose", "bose_einstein", "be", "+1", "1"):
        return 1.0
    if key in ("fermion", "fermi", "fermi_dirac", "fd", "-1"):
        return -1.0
    raise ValueError(
        "stat must be one of 'classical', 'boson'/'bose', or 'fermion'/'fermi'."
    )


@torch.no_grad()
def _C_quantum_impl(
    f, a,
    q,
    m, lam,
    Ng=16, batch_size=16,
    return_diagnostics=False,
    enforce_self_projection=True,
    stat_eta=1.0,
    F_func=F_analytical,
    F_needs_quadrature=False,
):
    f = torch.as_tensor(f)
    device, dtype = f.device, f.dtype

    q = torch.as_tensor(q, device=device, dtype=dtype)
    a = torch.as_tensor(a, device=device, dtype=dtype)
    m = torch.as_tensor(m, device=device, dtype=dtype)
    eta = torch.as_tensor(stat_eta, device=device, dtype=dtype)
    f_min = torch.finfo(dtype).tiny
    f_safe = torch.clamp(f, min=f_min)
    if stat_eta < 0.0:
        f_safe = torch.clamp(
            f_safe,
            max=torch.as_tensor(1.0 - 10.0 * torch.finfo(dtype).eps, device=device, dtype=dtype),
        )

    N = q.numel()
    p = q / a
    E = torch.sqrt(p * p + m * m)

    q_edges = grid_edges_from_centers_log(q)
    p_edges = q_edges / a
    dp_vec = p_edges[1:] - p_edges[:-1]

    dp_n = dp_vec.view(1, N, 1)
    dp_m = dp_vec.view(1, 1, N)

    f_n = f_safe.view(1, N, 1)
    f_m = f_safe.view(1, 1, N)

    pn = p.view(1, N, 1)
    pm = p.view(1, 1, N)
    En = E.view(1, N, 1)
    Em = E.view(1, 1, N)

    pref0 = 0.5
    C = torch.empty_like(f)

    if return_diagnostics:
        total_valid = torch.zeros((), device=device, dtype=dtype)
        total_phys = torch.zeros((), device=device, dtype=dtype)
        total_outside_weight = torch.zeros((), device=device, dtype=dtype)
        total_all_weight = torch.zeros((), device=device, dtype=dtype)
    else:
        total_valid = None
        total_phys = None
        total_outside_weight = None
        total_all_weight = None

    w_nm = dp_n * dp_m
    if F_needs_quadrature:
        mu2, w2 = _leggauss_torch(Ng, device, dtype)
    else:
        mu2 = torch.empty(0, device=device, dtype=dtype)
        w2 = torch.empty(0, device=device, dtype=dtype)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ib = torch.arange(start, end, device=device)

        pi = p[ib].view(-1, 1, 1)
        Ei = E[ib].view(-1, 1, 1)

        Etil = En + Em - Ei
        phys_ptil = Etil >= m

        ptil2 = Etil * Etil - m * m
        ptil = torch.sqrt(torch.clamp(ptil2, min=0.0))
        qtil = a * ptil

        inside_q = (qtil >= q[0]) & (qtil <= q[-1])

        qv_safe = torch.clamp(qtil, min=q[0], max=q[-1])
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
            torch.zeros_like(EQ),
        )

        wR = torch.clamp(wR, 0.0, 1.0)
        wL = 1.0 - wR

        # Quantum equilibrium has
        #
        #     f / (1 + eta f) = z exp(-E/T),
        #
        # with eta=+1 for Bose-Einstein and eta=-1 for Fermi-Dirac
        # statistics.  Interpolating this logarithm linearly in energy
        # preserves detailed balance for quantum equilibrium shapes.
        fL = f_safe[jL]
        fR = f_safe[jR]
        yL = torch.log(fL / torch.clamp(1.0 + eta * fL, min=f_min))
        yR = torch.log(fR / torch.clamp(1.0 + eta * fR, min=f_min))
        rtil = torch.exp(wL * yL + wR * yR)
        if stat_eta > 0.0:
            rtil = torch.clamp(
                rtil,
                max=torch.as_tensor(1.0 - 10.0 * torch.finfo(dtype).eps, device=device, dtype=dtype),
            )
        ftil_interp = rtil / torch.clamp(1.0 - eta * rtil, min=f_min)

        ftil = torch.where(
            inside_q,
            ftil_interp,
            torch.zeros_like(ftil_interp),
        )

        F = F_func(
            pi, Ei, pn, En, pm, Em, m, lam, mu2, w2, eps_disc=0.0
        )

        fi = f_safe[ib].view(-1, 1, 1)
        valid = phys_ptil & inside_q

        gain = f_n * f_m * (1.0 + eta * fi) * (1.0 + eta * ftil)
        loss = fi * ftil * (1.0 + eta * f_n) * (1.0 + eta * f_m)

        gain_loss = torch.where(
            valid,
            gain - loss,
            torch.zeros_like(ftil),
        )

        C[start:end] = pref0 * torch.sum(w_nm * F * gain_loss, dim=(1, 2)) / E[ib]

        if return_diagnostics:
            assert total_valid is not None
            assert total_phys is not None
            assert total_outside_weight is not None
            assert total_all_weight is not None
            total_valid = total_valid + valid.to(dtype).sum()
            total_phys = total_phys + phys_ptil.to(dtype).sum()

            outside = phys_ptil & (~inside_q)
            W = F.abs() * (torch.abs(gain) + torch.abs(loss))
            total_outside_weight = total_outside_weight + (
                W.masked_fill(~outside, 0.0)
            ).sum()
            total_all_weight = total_all_weight + W.sum()

    if return_diagnostics:
        I_energy_raw, I_number_raw, rE_raw, rN_raw = check_conservation_nonuniform(
            C, p, E, dp_vec
        )

    if enforce_self_projection:
        C = project_self_zero_moments(
            C_self_raw=C,
            f=f,
            p=p,
            E=E,
            dp=dp_vec,
        )

    if return_diagnostics:
        assert total_valid is not None
        assert total_phys is not None
        assert total_outside_weight is not None
        assert total_all_weight is not None
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
            "stat_eta": eta,
        }

        return C, E, p, dp_vec, diag

    return C, E, p, dp_vec


def _C_quantum_analytic_entry(
    f, a,
    q,
    m, lam,
    Ng=16, batch_size=16,
    return_diagnostics=False,
    enforce_self_projection=True,
    stat_eta=1.0,
):
    return _C_quantum_impl(
        f=f,
        a=a,
        q=q,
        m=m,
        lam=lam,
        Ng=Ng,
        batch_size=batch_size,
        return_diagnostics=return_diagnostics,
        enforce_self_projection=enforce_self_projection,
        stat_eta=stat_eta,
        F_func=F_analytical,
        F_needs_quadrature=False,
    )


def _C_quantum_quadrature_entry(
    f, a,
    q,
    m, lam,
    Ng=16, batch_size=16,
    return_diagnostics=False,
    enforce_self_projection=True,
    stat_eta=1.0,
):
    return _C_quantum_impl(
        f=f,
        a=a,
        q=q,
        m=m,
        lam=lam,
        Ng=Ng,
        batch_size=batch_size,
        return_diagnostics=return_diagnostics,
        enforce_self_projection=enforce_self_projection,
        stat_eta=stat_eta,
        F_func=F_contact,
        F_needs_quadrature=True,
    )


C_quantum_analytic_compiled = torch.compile(_C_quantum_analytic_entry)
C_quantum_quadrature_compiled = torch.compile(_C_quantum_quadrature_entry)


def C_quantum(
    f, a,
    q,
    m, lam,
    Ng=16, batch_size=16,
    return_diagnostics=False,
    enforce_self_projection=True,
    statistics="boson",
    kernel_backend="analytic",
):
    stat_eta = _stat_eta_from_name(statistics)
    backend, F_func, F_needs_quadrature = resolve_contact_kernel_backend(
        kernel_backend
    )
    if return_diagnostics:
        return _C_quantum_impl(
            f=f,
            a=a,
            q=q,
            m=m,
            lam=lam,
            Ng=Ng,
            batch_size=batch_size,
            return_diagnostics=True,
            enforce_self_projection=enforce_self_projection,
            stat_eta=stat_eta,
            F_func=F_func,
            F_needs_quadrature=F_needs_quadrature,
        )

    compiled = (
        C_quantum_analytic_compiled
        if backend == "analytic"
        else C_quantum_quadrature_compiled
    )
    return compiled(
        f=f,
        a=a,
        q=q,
        m=m,
        lam=lam,
        Ng=Ng,
        batch_size=batch_size,
        return_diagnostics=False,
        enforce_self_projection=enforce_self_projection,
        stat_eta=stat_eta,
    )

# ============================================================
# Source terms
# ============================================================

@torch.no_grad()
def C_Higgs_decay(
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
def C_condensate_decay(
    q, a,
    m_chi,
    m_parent,
    n_parent_of_a,
    Gamma_parent,
    m_other=0.0,
    multiplicity=1.0,
    gchi=1.0,
    background_torch=False,
):
    q = torch.as_tensor(q)
    device, dtype = q.device, q.dtype

    a = torch.as_tensor(a, device=device, dtype=dtype)
    m_chi = torch.as_tensor(m_chi, device=device, dtype=dtype)
    m_parent = torch.as_tensor(m_parent, device=device, dtype=dtype)
    m_other = torch.as_tensor(m_other, device=device, dtype=dtype)
    Gamma_parent = torch.as_tensor(Gamma_parent, device=device, dtype=dtype)
    multiplicity = torch.as_tensor(multiplicity, device=device, dtype=dtype)

    n_parent = _background_tensor(
        n_parent_of_a,
        a,
        device=device,
        dtype=dtype,
        background_torch=background_torch,
    )

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
# Generic RHS with optional moment projection
# ============================================================

@torch.no_grad()
def rhs_df_da_generic(
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
    background_torch=False,
):
    device, dtype = f.device, f.dtype
    p = q / a

    C_src = 0.0

    if T_of_a is not None and m_h is not None and g_trilinear is not None:
        Tt = _background_tensor(
            T_of_a,
            a,
            device=device,
            dtype=dtype,
            background_torch=background_torch,
        )
        C_src = C_src + C_Higgs_decay(
            p=p,
            T=Tt,
            m_chi=m_chi,
            m_h=m_h,
            g_trilinear=g_trilinear,
            pref=pref_FI,
        )

    if n_parent2_of_a is not None and Gamma_parent2 is not None and m_h2 is not None:
        C_src = C_src + C_condensate_decay(
            q=q,
            a=a,
            m_chi=m_chi,
            m_parent=m_h2,
            n_parent_of_a=n_parent2_of_a,
            Gamma_parent=Gamma_parent2,
            m_other=m_other2,
            multiplicity=multiplicity2,
            gchi=gchi,
            background_torch=background_torch,
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

    Ht = _background_tensor(
        H_of_a,
        a,
        device=device,
        dtype=dtype,
        background_torch=background_torch,
    )
    a_t = torch.as_tensor(a, device=device, dtype=dtype)

    return C_tot / (a_t * Ht)



# ============================================================
# FI-only RHS
# ============================================================

@torch.no_grad()
def rhs_df_da_FI(
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
    background_torch=False,
):
    C_tot = 0.0
    p = q / a

    if T_of_a is not None and m_h is not None and g_trilinear is not None:
        Tt = _background_tensor(
            T_of_a,
            a,
            device=f.device,
            dtype=f.dtype,
            background_torch=background_torch,
        )
        C_tot = C_tot + C_Higgs_decay(
            p=p,
            T=Tt,
            m_chi=m_chi,
            m_h=m_h,
            g_trilinear=g_trilinear,
            pref=pref_FI,
        )

    if n_parent2_of_a is not None and Gamma_parent2 is not None and m_h2 is not None:
        C_tot = C_tot + C_condensate_decay(
            q=q,
            a=a,
            m_chi=m_chi,
            m_parent=m_h2,
            n_parent_of_a=n_parent2_of_a,
            Gamma_parent=Gamma_parent2,
            m_other=m_other2,
            multiplicity=multiplicity2,
            gchi=gchi,
            background_torch=background_torch,
        )

    Ht = _background_tensor(
        H_of_a,
        a,
        device=f.device,
        dtype=f.dtype,
        background_torch=background_torch,
    )
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
