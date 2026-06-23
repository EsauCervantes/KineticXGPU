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


def interp1d_monotonic_torch(x_grid, y_grid, x, *, clamp=True):
    x_grid = torch.as_tensor(x_grid)
    y_grid = torch.as_tensor(y_grid, device=x_grid.device, dtype=x_grid.dtype)
    x = torch.as_tensor(x, device=x_grid.device, dtype=x_grid.dtype)

    if clamp:
        x_safe = torch.clamp(x, x_grid[0], x_grid[-1])
    else:
        x_safe = x

    jR = torch.searchsorted(x_grid, x_safe, right=False)
    jR = torch.clamp(jR, 1, x_grid.numel() - 1)
    jL = jR - 1

    xL = x_grid[jL]
    xR = x_grid[jR]
    yL = y_grid[jL]
    yR = y_grid[jR]

    t = (x_safe - xL) / (xR - xL)
    y = (1.0 - t) * yL + t * yR

    if not clamp:
        inside = (x >= x_grid[0]) & (x <= x_grid[-1])
        y = torch.where(inside, y, torch.zeros_like(y))

    return y
