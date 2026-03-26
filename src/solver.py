# solver.py
import torch
import math
import os
import numpy as np
import tempfile

@torch.no_grad()
def integrate_rk4_a(f0, a0, a1, n_steps, rhs):
    """
    rhs must have signature rhs(f, a) -> df/da
    """
    f = f0.clone()
    da = (a1 - a0) / n_steps
    a = torch.as_tensor(a0, device=f.device, dtype=f.dtype)

    for _ in range(n_steps):
        k1 = rhs(f, a)
        k2 = rhs(f + 0.5*da*k1, a + 0.5*da)
        k3 = rhs(f + 0.5*da*k2, a + 0.5*da)
        k4 = rhs(f + da*k3,     a + da)
        f = f + (da/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        a = a + da

    return f


@torch.no_grad()
def integrate_rk4_a_trajectory(
    f0, a0, a1, n_steps, rhs,
    store_every=1,
    print_every_pct=10,
    out_path_pt=None,      # e.g. "traj.pt"
    out_path_dat=None,     # e.g. "traj.dat"
    atomic=True,
    dat_fmt="%.18e",
):
    """
    RK4 integrator for df/da = rhs(f,a), storing snapshots of the trajectory.

    Parameters
    ----------
    f0 : tensor
        Initial distribution.
    a0, a1 : float
        Initial and final scale factors.
    n_steps : int
        Number of RK4 steps.
    rhs : callable
        Must have signature rhs(f, a) -> df/da tensor of same shape as f.
    store_every : int
        Store every `store_every` steps, plus initial and final states.
    print_every_pct : int or None
        Print progress every given percentage. Set None or <=0 to disable.
    out_path_pt : str or None
        If given, save a torch dictionary with trajectory data.
    out_path_dat : str or None
        If given, save a plain text file with rows:
            a  f[0]  f[1]  ...  f[N-1]
    atomic : bool
        If True, write via temporary file and rename.
    dat_fmt : str
        Float format for text output.

    Returns
    -------
    result : dict
        {
          "a": (M,) tensor,
          "f": (M,N) tensor,
          "a_final": scalar tensor,
          "f_final": (N,) tensor,
        }
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be > 0.")
    if store_every <= 0:
        raise ValueError("store_every must be > 0.")

    f = f0.clone()
    device, dtype = f.device, f.dtype

    da = (a1 - a0) / n_steps
    da_t = torch.as_tensor(da, device=device, dtype=dtype)
    a = torch.as_tensor(a0, device=device, dtype=dtype)

    a_hist = [a.detach().clone()]
    f_hist = [f.detach().clone()]

    if print_every_pct is not None and print_every_pct > 0:
        every = max(1, int(round(n_steps * print_every_pct / 100.0)))
    else:
        every = None

    for step in range(n_steps):
        k1 = rhs(f, a)
        k2 = rhs(f + 0.5 * da_t * k1, a + 0.5 * da_t)
        k3 = rhs(f + 0.5 * da_t * k2, a + 0.5 * da_t)
        k4 = rhs(f + da_t * k3, a + da_t)

        f = f + (da_t / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        a = a + da_t

        should_store = ((step + 1) % store_every == 0) or (step == n_steps - 1)
        if should_store:
            a_hist.append(a.detach().clone())
            f_hist.append(f.detach().clone())

        if every is not None and (((step + 1) % every == 0) or (step == n_steps - 1)):
            pct = 100.0 * (step + 1) / n_steps
            print(f"[RK4] step {step+1}/{n_steps}  ({pct:.1f}%)  a={float(a):.6e}")

    a_hist = torch.stack(a_hist, dim=0)     # (M,)
    f_hist = torch.stack(f_hist, dim=0)     # (M,N)

    result = {
        "a": a_hist,
        "f": f_hist,
        "a_final": a_hist[-1],
        "f_final": f_hist[-1],
    }

    if out_path_pt is not None:
        _torch_save_atomic(result, out_path_pt, atomic=atomic)

    if out_path_dat is not None:
        _save_trajectory_dat(a_hist, f_hist, out_path_dat, atomic=atomic, fmt=dat_fmt)

    return result


def _torch_save_atomic(obj, path, atomic=True):
    if not atomic:
        torch.save(obj, path)
        return

    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _save_trajectory_dat(a_hist, f_hist, path, atomic=True, fmt="%.18e"):
    a_np = a_hist.detach().cpu().numpy().reshape(-1, 1)   # (M,1)
    f_np = f_hist.detach().cpu().numpy()                  # (M,N)
    arr = np.concatenate([a_np, f_np], axis=1)           # (M,N+1)

    if not atomic:
        np.savetxt(path, arr, fmt=fmt)
        return

    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    os.close(fd)
    try:
        np.savetxt(tmp_path, arr, fmt=fmt)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

