# solver.py
import torch
import math
import os
import numpy as np

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
    out_path_pt=None,      # e.g. "traj.pt"  (binary, fast)
    out_path_dat=None,     # e.g. "traj.dat" (plain text, readable)
    atomic=True,
    dat_fmt="%.18e",
):
    """
    rhs: rhs(f, a) -> df/da

    Writes (overwrites) checkpoint files every time a snapshot is stored:
      - out_path_pt : torch.save({"a": (K,), "f": (K,N)})
      - out_path_dat: ASCII text with columns [a, f1, f2, ..., fN] and K rows

    atomic=True uses tmp+os.replace to avoid partially-written files.
    """
    f = f0.clone()
    da = (a1 - a0) / n_steps
    a = torch.as_tensor(a0, device=f.device, dtype=f.dtype)

    a_list = []
    f_list = []

    def _atomic_write_bytes(write_fn, path):
        if not atomic:
            write_fn(path)
            return
        tmp = path + ".tmp"
        write_fn(tmp)
        os.replace(tmp, path)

    def save_checkpoint():
        if (out_path_pt is None) and (out_path_dat is None):
            return

        # Stack on CPU for saving
        a_hist = torch.stack(a_list).cpu()   # (K,)
        f_hist = torch.stack(f_list).cpu()   # (K,N)

        if out_path_pt is not None:
            def write_pt(path):
                torch.save({"a": a_hist, "f": f_hist}, path)
            _atomic_write_bytes(write_pt, out_path_pt)

        if out_path_dat is not None:
            # Build a single matrix with shape (K, 1+N): [a | f...]
            a_np = a_hist.numpy().reshape(-1, 1)
            f_np = f_hist.numpy()
            mat = np.concatenate([a_np, f_np], axis=1)

            header = "a " + " ".join([f"f[{j}]" for j in range(f_np.shape[1])])

            def write_dat(path):
                np.savetxt(path, mat, fmt=dat_fmt, header=header, comments="")
            _atomic_write_bytes(write_dat, out_path_dat)

    def store():
        a_list.append(a.detach().cpu())
        f_list.append(f.detach().cpu())
        save_checkpoint()

    store()

    next_print = print_every_pct

    for step in range(1, n_steps + 1):
        k1 = rhs(f, a)
        k2 = rhs(f + 0.5*da*k1, a + 0.5*da)
        k3 = rhs(f + 0.5*da*k2, a + 0.5*da)
        k4 = rhs(f + da*k3,     a + da)
        f = f + (da/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        a = a + da

        pct = (100 * step) // n_steps
        if pct >= next_print:
            print(f"{int(pct)}% done!")
            next_print += print_every_pct

        if step % store_every == 0:
            store()

    a_hist = torch.stack(a_list)  # CPU
    f_hist = torch.stack(f_list)  # CPU
    return a_hist, f_hist

