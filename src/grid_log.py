# src/grid_log.py
import math
import torch


# -------------------------
# Grid construction (centers)
# -------------------------

def make_log_q_grid(q_min, q_max, N, *, device=None, dtype=torch.float64, base=10.0):
    """
    Log-spaced comoving momentum grid q in [q_min, q_max] with constant spacing in log(q).

    Returns:
      q:      (N,) tensor of centers
      logq0:  scalar tensor, log(q_min) in the chosen base
      dlogq:  scalar tensor, constant spacing in log(q)
      log_space: "log10" or "ln"
    """
    if q_min <= 0:
        raise ValueError("q_min must be > 0 for log-spaced grid.")
    if q_max <= q_min:
        raise ValueError("q_max must be > q_min.")
    if N < 2:
        raise ValueError("N must be >= 2.")

    if base != 10.0:
        ln_min = math.log(q_min)
        ln_max = math.log(q_max)
        ln = torch.linspace(ln_min, ln_max, N, device=device, dtype=dtype)
        q = torch.exp(ln)
        logq0 = ln[0]
        dlogq = ln[1] - ln[0]
        return q, logq0, dlogq, "ln"

    log10_min = math.log10(q_min)
    log10_max = math.log10(q_max)
    log10 = torch.linspace(log10_min, log10_max, N, device=device, dtype=dtype)
    q = torch.pow(torch.as_tensor(10.0, device=device, dtype=dtype), log10)
    logq0 = log10[0]
    dlogq = log10[1] - log10[0]
    return q, logq0, dlogq, "log10"


# -------------------------
# Edges and widths (log-aware)
# -------------------------

def grid_edges_from_centers_log(x):
    """
    Construct bin edges for a strictly increasing positive 1D grid of centers x
    assuming x is approximately log-spaced. Uses geometric midpoints.

    Returns:
      edges: (len(x)+1,) tensor
    """
    x = torch.as_tensor(x)
    if torch.any(x <= 0):
        raise ValueError("grid_edges_from_centers_log requires x > 0.")
    if x.ndim != 1 or x.numel() < 2:
        raise ValueError("x must be a 1D tensor with at least 2 points.")
    if torch.any(x[1:] <= x[:-1]):
        raise ValueError("x must be strictly increasing.")

    edges = torch.empty(x.numel() + 1, device=x.device, dtype=x.dtype)
    edges[1:-1] = torch.sqrt(x[1:] * x[:-1])

    r0 = x[1] / x[0]
    rN = x[-1] / x[-2]
    edges[0] = x[0] / torch.sqrt(r0)
    edges[-1] = x[-1] * torch.sqrt(rN)
    return edges


def bin_widths_from_centers_log(x):
    edges = grid_edges_from_centers_log(x)
    return edges[1:] - edges[:-1]


# -------------------------
# Kinematics
# -------------------------

def p_of_a(q, a):
    q = torch.as_tensor(q)
    a = torch.as_tensor(a, device=q.device, dtype=q.dtype)
    return q / a


def q_of_a(p, a):
    p = torch.as_tensor(p)
    a = torch.as_tensor(a, device=p.device, dtype=p.dtype)
    return a * p


# -------------------------
# Interpolation on log grids
# -------------------------

def interp_linear_log_uniform_vec(logx0, dlogx, f, x, *, log_space="log10", clamp=False):
    """
    Linear interpolation for a grid uniform in log(x).

    Inputs:
      logx0, dlogx: scalars describing the log grid
      f: (N,) tensor values at centers
      x: tensor of any shape
      log_space: "log10" or "ln"
      clamp: if True, clamp x into [xmin, xmax], else return 0 outside
    """
    f = torch.as_tensor(f)
    x = torch.as_tensor(x, device=f.device, dtype=f.dtype)

    if torch.any(x <= 0):
        if not clamp:
            return torch.zeros_like(x)
        x = torch.clamp(x, min=torch.finfo(x.dtype).tiny)

    N = f.shape[0]
    logx0 = torch.as_tensor(logx0, device=f.device, dtype=f.dtype)
    dlogx = torch.as_tensor(dlogx, device=f.device, dtype=f.dtype)

    if log_space == "log10":
        lx = torch.log10(x)
        xmin = torch.pow(torch.as_tensor(10.0, device=f.device, dtype=f.dtype), logx0)
        xmax = torch.pow(torch.as_tensor(10.0, device=f.device, dtype=f.dtype), logx0 + (N - 1) * dlogx)
    elif log_space == "ln":
        lx = torch.log(x)
        xmin = torch.exp(logx0)
        xmax = torch.exp(logx0 + (N - 1) * dlogx)
    else:
        raise ValueError("log_space must be 'log10' or 'ln'.")

    x_flat = x.reshape(-1)
    lx_flat = lx.reshape(-1)
    out = torch.zeros_like(x_flat)

    eps = 10.0 * torch.finfo(x.dtype).eps * max(1.0, float(abs(xmax)))

    if clamp:
        x_clamped = torch.clamp(x_flat, xmin, xmax)
        if log_space == "log10":
            lx_flat = torch.log10(x_clamped)
        else:
            lx_flat = torch.log(x_clamped)
        inr = torch.ones_like(x_flat, dtype=torch.bool)
    else:
        inr = (x_flat >= xmin) & (x_flat <= xmax + eps)

    if not torch.any(inr):
        return out.reshape(x.shape)

    idx = torch.nonzero(inr, as_tuple=False).squeeze(1)
    lxi = lx_flat[idx]

    u = (lxi - logx0) / dlogx
    j = torch.floor(u).to(torch.long)
    j = torch.clamp(j, 0, N - 1)

    right = (j == N - 1)
    if torch.any(right):
        out[idx[right]] = f[-1]

    ok = ~right
    if torch.any(ok):
        jo = j[ok]
        t = u[ok] - jo.to(u.dtype)
        out[idx[ok]] = (1.0 - t) * f[jo] + t * f[jo + 1]

    return out.reshape(x.shape)


def interp_nearest_log_uniform_vec(logx0, dlogx, f, x, *, log_space="log10", clamp=False):
    """
    Nearest-neighbor interpolation for a grid uniform in log(x).
    """
    f = torch.as_tensor(f)
    x = torch.as_tensor(x, device=f.device, dtype=f.dtype)

    if torch.any(x <= 0):
        if not clamp:
            return torch.zeros_like(x)
        x = torch.clamp(x, min=torch.finfo(x.dtype).tiny)

    N = f.shape[0]
    logx0 = torch.as_tensor(logx0, device=f.device, dtype=f.dtype)
    dlogx = torch.as_tensor(dlogx, device=f.device, dtype=f.dtype)

    if log_space == "log10":
        lx = torch.log10(x)
        xmin = torch.pow(torch.as_tensor(10.0, device=f.device, dtype=f.dtype), logx0)
        xmax = torch.pow(torch.as_tensor(10.0, device=f.device, dtype=f.dtype), logx0 + (N - 1) * dlogx)
    elif log_space == "ln":
        lx = torch.log(x)
        xmin = torch.exp(logx0)
        xmax = torch.exp(logx0 + (N - 1) * dlogx)
    else:
        raise ValueError("log_space must be 'log10' or 'ln'.")

    x_flat = x.reshape(-1)
    lx_flat = lx.reshape(-1)

    if clamp:
        x_flat = torch.clamp(x_flat, xmin, xmax)
        if log_space == "log10":
            lx_flat = torch.log10(x_flat)
        else:
            lx_flat = torch.log(x_flat)
        inr = torch.ones_like(x_flat, dtype=torch.bool)
    else:
        eps = 10.0 * torch.finfo(x.dtype).eps * max(1.0, float(abs(xmax)))
        inr = (x_flat >= xmin) & (x_flat <= xmax + eps)

    out = torch.zeros_like(x_flat)
    if not torch.any(inr):
        return out.reshape(x.shape)

    idx = torch.nonzero(inr, as_tuple=False).squeeze(1)
    lxi = lx_flat[idx]

    u = (lxi - logx0) / dlogx
    j = torch.round(u).to(torch.long)
    j = torch.clamp(j, 0, N - 1)
    out[idx] = f[j]
    return out.reshape(x.shape)


def remap_two_bin_conservative_energy_logq(
    q_grid, f, q_query, a, m, *, clamp=False, return_meta=False
):
    """
    Two-bin remap for off-grid q_query values on a monotonic q-grid.

    For each q_query, find neighboring bins (jL, jR) and choose weights
      wL + wR = 1
      wL * E(qL,a) + wR * E(qR,a) = E(q_query,a)

    with E(q,a) = sqrt((q/a)^2 + m^2).

    This gives a smooth local remap that preserves the local particle-weight
    sum and matches the physical energy of the remapped point.

    Returns:
      f_remap with same shape as q_query

    If return_meta=True also returns:
      jL, jR, wL, wR, inside
    """
    q_grid = torch.as_tensor(q_grid)
    f = torch.as_tensor(f, device=q_grid.device, dtype=q_grid.dtype)
    q_query = torch.as_tensor(q_query, device=q_grid.device, dtype=q_grid.dtype)
    a = torch.as_tensor(a, device=q_grid.device, dtype=q_grid.dtype)
    m = torch.as_tensor(m, device=q_grid.device, dtype=q_grid.dtype)

    if q_grid.ndim != 1 or q_grid.numel() < 2:
        raise ValueError("q_grid must be 1D with at least 2 points.")
    if f.shape != q_grid.shape:
        raise ValueError(f"f must have shape {tuple(q_grid.shape)}, got {tuple(f.shape)}")
    if torch.any(q_grid <= 0):
        raise ValueError("q_grid must be positive.")
    if torch.any(q_grid[1:] <= q_grid[:-1]):
        raise ValueError("q_grid must be strictly increasing.")
    if a <= 0:
        raise ValueError("a must be > 0.")

    N = q_grid.numel()
    q_flat = q_query.reshape(-1)
    out = torch.zeros_like(q_flat)

    qmin = q_grid[0]
    qmax = q_grid[-1]

    if clamp:
        qq = torch.clamp(q_flat, qmin, qmax)
        inside = torch.ones_like(q_flat, dtype=torch.bool)
    else:
        inside = (q_flat >= qmin) & (q_flat <= qmax)
        qq = q_flat

    if not torch.any(inside):
        if return_meta:
            zlong = torch.zeros_like(q_flat, dtype=torch.long)
            z = torch.zeros_like(q_flat)
            return (
                out.reshape(q_query.shape),
                zlong.reshape(q_query.shape),
                zlong.reshape(q_query.shape),
                z.reshape(q_query.shape),
                z.reshape(q_query.shape),
                inside.reshape(q_query.shape),
            )
        return out.reshape(q_query.shape)

    idx = torch.nonzero(inside, as_tuple=False).squeeze(1)
    qqi = qq[idx]

    jR = torch.searchsorted(q_grid, qqi, right=False)
    jR = torch.clamp(jR, 1, N - 1)
    jL = jR - 1

    qL = q_grid[jL]
    qR = q_grid[jR]

    EL = torch.sqrt((qL / a) ** 2 + m ** 2)
    ER = torch.sqrt((qR / a) ** 2 + m ** 2)
    EQ = torch.sqrt((qqi / a) ** 2 + m ** 2)

    denom = ER - EL
    tiny = 10.0 * torch.finfo(q_grid.dtype).eps
    safe = torch.abs(denom) > tiny

    wR = torch.zeros_like(EQ)
    wR[safe] = (EQ[safe] - EL[safe]) / denom[safe]
    wR = torch.clamp(wR, 0.0, 1.0)
    wL = 1.0 - wR

    out[idx] = wL * f[jL] + wR * f[jR]

    if return_meta:
        jL_all = torch.zeros_like(q_flat, dtype=torch.long)
        jR_all = torch.zeros_like(q_flat, dtype=torch.long)
        wL_all = torch.zeros_like(q_flat)
        wR_all = torch.zeros_like(q_flat)

        jL_all[idx] = jL
        jR_all[idx] = jR
        wL_all[idx] = wL
        wR_all[idx] = wR

        return (
            out.reshape(q_query.shape),
            jL_all.reshape(q_query.shape),
            jR_all.reshape(q_query.shape),
            wL_all.reshape(q_query.shape),
            wR_all.reshape(q_query.shape),
            inside.reshape(q_query.shape),
        )

    return out.reshape(q_query.shape)


def is_log_uniform_grid(x, *, log_space="log10", rtol=1e-12, atol=1e-15):
    x = torch.as_tensor(x)
    if x.ndim != 1 or x.numel() < 2:
        raise ValueError("Grid must be 1D with at least 2 points.")
    if torch.any(x <= 0):
        raise ValueError("Log grid requires x > 0.")

    if log_space == "log10":
        lx = torch.log10(x)
    elif log_space == "ln":
        lx = torch.log(x)
    else:
        raise ValueError("log_space must be 'log10' or 'ln'.")

    diffs = lx[1:] - lx[:-1]
    dlogx = torch.mean(diffs)
    uniform = torch.allclose(diffs, dlogx, rtol=rtol, atol=atol)
    return bool(uniform), dlogx
