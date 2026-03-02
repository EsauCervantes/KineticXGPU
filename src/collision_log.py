# src/collision_torch_logq.py
import numpy as np
import torch
from grid_log import (
    p_of_a,
    q_of_a,
    grid_edges_from_centers_log,
    interp_linear_log_uniform_vec,  
    interp_nearest_log_uniform_vec
)
from func import f_eq_MB

def energy(p, m):
    return torch.sqrt(p*p + m*m)

def _leggauss_torch(Ng, device, dtype):
    # numpy nodes/weights -> torch tensors
    x, w = np.polynomial.legendre.leggauss(Ng)  # on [-1,1]
    mu = torch.tensor(x, device=device, dtype=dtype)   # cosθ
    wt = torch.tensor(w, device=device, dtype=dtype)
    return mu, wt



@torch.no_grad()
def _F_kernel_batch_constant_M2_1d(pi, Ei, pn, En, pm, Em, m, lam, Ng,
                                  eps_disc=0.0):
    """
    1D version for the *forced* integrand:  lam^2 * (pn*pm)/sqrt(D),
    integrating c3 analytically and c2 by Gauss-Legendre.

    Shapes:
      pi,Ei: (B,1,1)
      pn,En: (1,N,1)
      pm,Em: (1,1,N)
    Returns:
      F: (B,N,N)
    """
    device, dtype = pi.device, pi.dtype
    mu2, w2 = _leggauss_torch(Ng, device, dtype)

    lam2 = torch.as_tensor(lam, device=device, dtype=dtype)**2
    m2 = torch.as_tensor(m, device=device, dtype=dtype)**2

    B = pi.shape[0]
    N = pn.shape[1]
    pnpm = pn * pm  # (1,N,N) via broadcast
    pref = 1.0 / (4.0 * (2.0 * torch.pi) ** 4)

    # Constant (in c3) piece of "num" from Eq. (A3) structure:
    # num = m^2 + En Em - En Ei - Em Ei + pi(pn c2 + pm c3) - pn pm c2 c3
    A0_base = m2 + En*Em - En*Ei - Em*Ei  # (B,N,N)

    Facc = torch.zeros((B, N, N), device=device, dtype=dtype)

    # Scalars for clipping
    minus_one = torch.as_tensor(-1.0, device=device, dtype=dtype)
    plus_one  = torch.as_tensor( 1.0, device=device, dtype=dtype)

    for a2 in range(Ng):
        c2 = mu2[a2]          # scalar
        w2a = w2[a2]          # scalar
        s2sq = (1.0 - c2*c2)  # scalar

        # num = A0 + A1*c3, with:
        # A0 = A0_base + pi*pn*c2
        # A1 = pi*pm - pn*pm*c2
        A0 = A0_base + pi * (pn * c2)       # (B,N,N)
        A1 = pi * pm - (pnpm * c2)          # (B,N,N)

        # den^2 = (pn*pm)^2 * (1-c2^2) * (1-c3^2) = K2*(1-c3^2)
        K2 = (pnpm * pnpm) * s2sq           # (B,N,N) by broadcast

        # D(c3) = K2*(1-c3^2) - (A0 + A1*c3)^2 = a c3^2 + b c3 + c
        a = -(K2 + A1*A1)                   # (B,N,N)  (typically <0)
        b = -2.0 * A0 * A1                  # (B,N,N)
        c = (K2 - A0*A0)                    # (B,N,N)

        disc = b*b - 4.0*a*c
        mask_disc = disc > eps_disc

        sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))

        # Roots of D(c3)=0
        denom = 2.0 * a
        r1 = (-b - sqrt_disc) / denom
        r2 = (-b + sqrt_disc) / denom
        rmin = torch.minimum(r1, r2)
        rmax = torch.maximum(r1, r2)

        # Intersect with [-1,1]
        c3min = torch.maximum(rmin, minus_one)
        c3max = torch.minimum(rmax, plus_one)
        mask_int = mask_disc & (c3max > c3min)

        # Prepare safe factors
        sqrt_minus_a = torch.sqrt(torch.clamp(-a, min=0.0))
        inv_sqrt_disc = torch.where(mask_int, 1.0 / sqrt_disc, torch.zeros_like(sqrt_disc))

        # u(x) = (2ax+b)/sqrt(disc)
        def u_of(x):
            u = (2.0*a*x + b) * inv_sqrt_disc
            return torch.clamp(u, -1.0, 1.0)

        umin = u_of(c3min)
        umax = u_of(c3max)

        # Correct sign for a<0 primitive:
        # I = (asin(umin) - asin(umax)) / sqrt(-a)
        I = torch.where(
            mask_int,
            (torch.asin(umin) - torch.asin(umax)) / torch.where(mask_int, sqrt_minus_a, torch.ones_like(sqrt_minus_a)),
            torch.zeros_like(disc),
        )

        # F contribution for this c2 node
        contrib = lam2 * pnpm * I
        Facc = Facc + w2a * contrib

    # Optional: remove tiny negative roundoff only (do NOT take abs)
    F = pref * Facc
    F = torch.clamp(F, min=0.0)

    return F



@torch.no_grad()
def C_self_torch_logq(
    f, a,
    q, logq0, dlogq, log_space,
    m, lam=1.0, gchi=1.0,
    Ng=16, batch_size=16
):
    """
    Same physics as C_self_with_F_torch_batched, but for log-uniform q-grid.

    Discretization change:
      (Δp)^2 Σ_{n,m}  ->  Σ_{n,m} (Δp_n)(Δp_m)

    Interpolation change:
      f(tilde p) computed as f(tilde q) with tilde q = a * tilde p,
      using log-grid interpolation on q.
    """
    f = torch.as_tensor(f)
    device, dtype = f.device, f.dtype

    q = torch.as_tensor(q, device=device, dtype=dtype)
    a = torch.as_tensor(a, device=device, dtype=dtype)
    m = torch.as_tensor(m, device=device, dtype=dtype)

    N = q.numel()
    if f.shape != (N,):
        raise ValueError(f"f must have shape {(N,)}, got {tuple(f.shape)}")

    # Physical momenta/energies at this a
    p = p_of_a(q, a)     # (N,)
    E = energy(p, m)     # (N,)

    # --- bin widths in p for nonuniform grid ---
    # Build q-edges geometrically, then map to p-edges: p_edges = q_edges / a
    q_edges = grid_edges_from_centers_log(q)   # (N+1,)
    p_edges = q_edges / a                      # (N+1,)
    dp_vec = p_edges[1:] - p_edges[:-1]        # (N,)

    # For weighted double integral: dp_n dp_m
    dp_n = dp_vec.view(1, N, 1)   # (1,N,1)
    dp_m = dp_vec.view(1, 1, N)   # (1,1,N)

    # Precompute for sums/interp
    p2 = p*p
    p2_n = p2.view(1, N, 1)
    p2_m = p2.view(1, 1, N)

    f_n = f.view(1, N, 1)
    f_m = f.view(1, 1, N)

    # For F computation
    pn = p.view(1, N, 1)
    pm = p.view(1, 1, N)
    En = E.view(1, N, 1)
    Em = E.view(1, 1, N)

    C = torch.empty_like(f)

    # Constant prefactor 1/(2 gchi) (dp weights are inside the sum now)
    pref0 = 1.0 / (2.0 * gchi)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ib = torch.arange(start, end, device=device)
        B = ib.numel()

        pi = p[ib].view(B, 1, 1)
        Ei = E[ib].view(B, 1, 1)

        # tilde p^2 = p_n^2 + p_m^2 - p_i^2
        pi2 = p2[ib].view(B, 1, 1)
        ptil2 = p2_n + p2_m - pi2              # (B,N,N)
        phys_ptil = (ptil2 >= 0)

        ptil = torch.zeros_like(ptil2)
        ptil[phys_ptil] = torch.sqrt(ptil2[phys_ptil])

        # interpolate f(tilde p) by mapping to tilde q = a * tilde p on log-q grid
        qtil = a * ptil                        # (B,N,N)
        ftil = interp_nearest_log_uniform_vec(
            logx0=logq0, dlogx=dlogq, f=f, x=qtil,
            log_space=log_space, clamp=True
        )                                      # (B,N,N)

        # Compute F_{i n m}
        F = _F_kernel_batch_constant_M2_1d(pi, Ei, pn, En, pm, Em, m, lam, Ng, eps_disc=0.0)

        fi = f[ib].view(B, 1, 1)
        gain_loss = (f_n * f_m) - fi * ftil
        gain_loss = torch.where(phys_ptil, gain_loss, torch.zeros_like(gain_loss))
        
        ##############
        outside = phys_ptil & (ptil > p[-1])
        # weight by magnitude of contribution (pick one; this is a good start)
        W = (F.abs() * fi.abs())  # (B,N,N) broadcasted automatically if fi is (B,1,1)
        num = (W * ftil.abs()).masked_fill(~outside, 0.0).sum()
        den = (W * ftil.abs()).sum() + 1e-300
        print("weighted outside (loss) fraction:", (num/den).item())
        ##############

        # weights for nonuniform dp
        w_nm = dp_n * dp_m                     # (1,N,N), broadcasts to (B,N,N)

        C_batch = pref0 * torch.sum(w_nm * F * gain_loss, dim=(1, 2))  # (B,)
        C[start:end] = C_batch

    return C, E, p, dp_vec



    
@torch.no_grad()
def enforce_number_energy(C, p, E, dp, eps=0.0):
    dp_t = torch.as_tensor(dp, device=p.device, dtype=p.dtype)
    w = p*p * dp_t

    S0 = torch.sum(w * C)                 # energy constraint target 0
    S1 = torch.sum(w * (C / (E + eps)))   # number constraint target 0

    A00 = torch.sum(w)
    A01 = torch.sum(w * E)
    A10 = torch.sum(w / (E + eps))
    A11 = torch.sum(w * (E / (E + eps)))  # = sum(w) if eps=0

    det = A00*A11 - A01*A10
    alpha = ( S0*A11 - S1*A01) / det
    beta  = (-S0*A10 + S1*A00) / det

    return C - alpha - beta*E

@torch.no_grad()
def check_conservation_nonuniform(C, p, E, dp_vec):
    dp_vec = torch.as_tensor(dp_vec, device=p.device, dtype=p.dtype)
    w = p*p * dp_vec

    I_energy = torch.sum(w * C)
    I_number = torch.sum(w * (C / E))

    scale_energy = torch.sum(w * torch.abs(C)) + 1e-300
    scale_number = torch.sum(w * torch.abs(C / E)) + 1e-300

    return I_energy, I_number, I_energy/scale_energy, I_number/scale_number

@torch.no_grad()
def C_FI_inverse_decay_MB(p, E, T, m_med, Gamma, pref=1.0):
    """
    Unintegrated FI production term from inverse decay using detailed balance:
      C_prod(p) = pref * (m_med/E) * Gamma * f_eq_MB(E, T)

    Inputs:
      p, E: (N,) torch tensors
      T: 0-d torch tensor (SM temperature)
      m_med: 0-d torch tensor (mediator mass, here A')
      Gamma: 0-d torch tensor (decay width A'->ff)
      pref: optional overall factor (branching, multiplicity, etc.)
    Returns:
      C_FI: (N,) torch tensor
    """
    feq = f_eq_MB(E, T)
    return pref * (m_med / E) * Gamma * feq


def rhs_df_da_torch_logq(
    f, a,
    q, logq0, dlogq, log_space,
    m, lam,
    H_of_a, T_of_a=None,
    m_med=None, Gamma=None, pref_FI=1.0,
    gchi=1.0, batch_size=16
):
    C_el, E, p, dp_vec = C_self_torch_logq(
        f, a,
        q, logq0, dlogq, log_space,
        m, lam, gchi=gchi, Ng=64, batch_size=batch_size
    )

    C_tot = C_el

    if T_of_a is not None:
        Tt = torch.tensor(float(T_of_a(float(a))), device=f.device, dtype=f.dtype)
        m_med_t = torch.as_tensor(m_med, device=f.device, dtype=f.dtype)
        Gamma_t = torch.as_tensor(Gamma, device=f.device, dtype=f.dtype)

        C_tot = C_tot + C_FI_inverse_decay_MB(p=p, E=E, T=Tt, m_med=m_med_t, Gamma=Gamma_t, pref=pref_FI)

    Ht = torch.tensor(float(H_of_a(float(a))), device=f.device, dtype=f.dtype)
    a_t = torch.as_tensor(a, device=f.device, dtype=f.dtype)

    return C_tot / (a_t * Ht * E)
    



def rhs_dfFI_da_torch(
    f, a, q, dq, m, lam,
    H_of_a, T_of_a=None,
    m_med=None, Gamma=None, pref_FI=1.0,
    gchi=1.0, batch_size=16
):
    f = torch.as_tensor(f)
    device, dtype = f.device, f.dtype

    q  = torch.as_tensor(q,  device=device, dtype=dtype)
    a  = torch.as_tensor(a,  device=device, dtype=dtype)
    m  = torch.as_tensor(m,  device=device, dtype=dtype)

    # recompute physical kinematics on-the-fly
    p = q / a
    E = torch.sqrt(p*p + m*m)

    C_tot = torch.zeros_like(f)

    if T_of_a is not None:
        T  = float(T_of_a(float(a)))
        Tt = torch.tensor(T, device=device, dtype=dtype)

        if m_med is None or Gamma is None:
            raise ValueError("Provide m_med and Gamma when using T_of_a (FI term).")

        m_med_t = torch.as_tensor(m_med, device=device, dtype=dtype)
        Gamma_t = torch.as_tensor(Gamma, device=device, dtype=dtype)

        C_tot = C_tot + C_FI_inverse_decay_MB(
            p=p, E=E, T=Tt, m_med=m_med_t, Gamma=Gamma_t, pref=pref_FI
        )

    H  = float(H_of_a(float(a)))
    Ht = torch.tensor(H, device=device, dtype=dtype)

    return C_tot / (a * Ht * E)
