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
    """
    if q_min <= 0:
        raise ValueError("q_min must be > 0 for log-spaced grid.")
    if q_max <= q_min:
        raise ValueError("q_max must be > q_min.")
    if N < 2:
        raise ValueError("N must be >= 2.")

    # torch.logspace uses log10 endpoints, but we also return log-grid metadata for fast indexing.
    if base != 10.0:
        # build via exp to allow arbitrary base, still constant spacing in ln
        ln_min = math.log(q_min)
        ln_max = math.log(q_max)
        ln = torch.linspace(ln_min, ln_max, N, device=device, dtype=dtype)
        q = torch.exp(ln)
        logq0 = ln[0]
        dlogq = ln[1] - ln[0]          # in ln
        return q, logq0, dlogq, "ln"

    log10_min = math.log10(q_min)
    log10_max = math.log10(q_max)
    log10 = torch.linspace(log10_min, log10_max, N, device=device, dtype=dtype)
    q = torch.pow(torch.as_tensor(10.0, device=device, dtype=dtype), log10)
    logq0 = log10[0]
    dlogq = log10[1] - log10[0]       # in log10
    return q, logq0, dlogq, "log10"


# -------------------------
# Edges and widths (log-aware)
# -------------------------

def grid_edges_from_centers_log(x):
    """
    Construct bin edges for a strictly increasing positive 1D grid of centers x
    assuming x is approximately log-spaced. Uses geometric midpoints.

    edges[i+1/2] = sqrt(x[i] * x[i+1]) for interior edges.

    Returns:
      edges: (len(x)+1,) tensor
    """
    x = torch.as_tensor(x)
    if torch.any(x <= 0):
        raise ValueError("grid_edges_from_centers_log requires x > 0.")

    edges = torch.empty(x.numel() + 1, device=x.device, dtype=x.dtype)

    # interior edges: geometric mean
    edges[1:-1] = torch.sqrt(x[1:] * x[:-1])

    # extrapolate first/last edge using constant ratio assumption
    r0 = x[1] / x[0]
    rN = x[-1] / x[-2]
    edges[0] = x[0] / torch.sqrt(r0)
    edges[-1] = x[-1] * torch.sqrt(rN)
    return edges


def bin_widths_from_centers_log(x):
    """Return Δx_i for log-spaced centers using geometric-midpoint edges."""
    edges = grid_edges_from_centers_log(x)
    return edges[1:] - edges[:-1]


# -------------------------
# Kinematics
# -------------------------

def p_of_a(q, a):
    """Physical momentum p = q/a."""
    q = torch.as_tensor(q)
    a = torch.as_tensor(a, device=q.device, dtype=q.dtype)
    return q / a


def q_of_a(p, a):
    """Comoving momentum q = a p."""
    p = torch.as_tensor(p)
    a = torch.as_tensor(a, device=p.device, dtype=p.dtype)
    return a * p


# -------------------------
# Interpolation on log grids
# -------------------------

def interp_linear_log_uniform_vec(logx0, dlogx, f, x, *, log_space="log10", clamp=False):
    """
    Linear interpolation for a grid uniform in log(x).

    Assumes grid centers:
      logx_j = logx0 + j*dlogx,  j=0..N-1
      x_j    = base**logx_j  (if log_space="log10")
           or exp(logx_j)    (if log_space="ln")

    Inputs:
      logx0, dlogx: scalars (float or 0-d tensor) describing the *log* grid
      f: (N,) tensor values at centers
      x: tensor of any shape (must be >0)
      log_space: "log10" or "ln"
      clamp: if True, clamp x into [xmin,xmax] and return boundary values instead of 0 outside

    Returns:
      f(x) with same shape as x.
    """
    f = torch.as_tensor(f)
    x = torch.as_tensor(x, device=f.device, dtype=f.dtype)

    if torch.any(x <= 0):
        # for safety: outside domain => 0 (or clamp to xmin)
        if not clamp:
            return torch.zeros_like(x)
        # if clamp, clamp to a tiny positive and proceed
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

    # work flattened
    x_flat = x.reshape(-1)
    lx_flat = lx.reshape(-1)
    out = torch.zeros_like(x_flat)

    # tolerance at top end
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
        jo = j[ok]  # 0..N-2
        t = u[ok] - jo.to(u.dtype)
        out[idx[ok]] = (1.0 - t) * f[jo] + t * f[jo + 1]

    return out.reshape(x.shape)


def interp_nearest_log_uniform_vec(logx0, dlogx, f, x, *, log_space="log10", clamp=False):
    """
    Nearest-neighbor interpolation for a grid uniform in log(x).
    Same grid conventions as interp_linear_log_uniform_vec.
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


def is_log_uniform_grid(x, *, log_space="log10", rtol=1e-12, atol=1e-15):
    """
    Check whether x is uniformly spaced in log-space ("log10" or "ln").
    Returns (is_uniform, dlogx).
    """
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

