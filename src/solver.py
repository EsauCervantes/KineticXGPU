import torch
import math
import os
import json
import numpy as np
import tempfile
from pathlib import Path

from functools import partial
from collision import (
    rhs_df_da_torch_logq_FI,
    rhs_df_da_torch_logq_generic,
    C_self_torch_logq,
    C_self_torch_logq_conservative_scatter,
    estimate_gamma_eff_from_current_f,
)
from grid_log import grid_edges_from_centers_log


@torch.no_grad()
def distribution_number_energy_moments(f, q, a, m):
    q = torch.as_tensor(q, device=f.device, dtype=f.dtype)
    a = torch.as_tensor(a, device=f.device, dtype=f.dtype)
    m = torch.as_tensor(m, device=f.device, dtype=f.dtype)
    q_edges = grid_edges_from_centers_log(q)
    p = q / a
    dp = (q_edges[1:] - q_edges[:-1]) / a
    E = torch.sqrt(p * p + m * m)
    w = p * p * dp
    return torch.sum(w * f), torch.sum(w * E * f), p, E, dp


@torch.no_grad()
def project_distribution_to_number_energy(
    f,
    q,
    a,
    m,
    target_number,
    target_energy,
    support_rel=1e-12,
    floor_rel=1e-30,
):
    f = torch.as_tensor(f)
    target_number = torch.as_tensor(target_number, device=f.device, dtype=f.dtype)
    target_energy = torch.as_tensor(target_energy, device=f.device, dtype=f.dtype)

    number, energy_moment, p, E, dp = distribution_number_energy_moments(f, q, a, m)
    if target_number <= 0.0 or target_energy <= 0.0:
        return torch.zeros_like(f)

    w = p * p * dp
    f_pos = torch.clamp(f, min=torch.zeros((), device=f.device, dtype=f.dtype))

    support_weight = w * f_pos
    if torch.sum(support_weight) > torch.finfo(f.dtype).tiny:
        target_emean = target_energy / target_number
        Emin = torch.min(E[support_weight > 0.0])
        Emax = torch.max(E[support_weight > 0.0])

        if target_emean >= Emin and target_emean <= Emax:
            def tilted_mean(beta_val):
                expo_val = torch.clamp(beta_val * E, min=-80.0, max=80.0)
                W_val = support_weight * torch.exp(expo_val)
                S0_val = torch.sum(W_val)
                S1_val = torch.sum(W_val * E)
                return S1_val / torch.clamp(S0_val, min=torch.finfo(f.dtype).tiny)

            beta_lo = torch.as_tensor(-1.0, device=f.device, dtype=f.dtype)
            beta_hi = torch.as_tensor(1.0, device=f.device, dtype=f.dtype)

            for _ in range(40):
                if tilted_mean(beta_lo) <= target_emean:
                    break
                beta_lo = beta_lo * 2.0

            for _ in range(40):
                if tilted_mean(beta_hi) >= target_emean:
                    break
                beta_hi = beta_hi * 2.0

            beta = torch.zeros((), device=f.device, dtype=f.dtype)
            for _ in range(80):
                beta = 0.5 * (beta_lo + beta_hi)
                mean_E = tilted_mean(beta)

                if mean_E < target_emean:
                    beta_lo = beta
                else:
                    beta_hi = beta

                if torch.abs(mean_E - target_emean) <= 1e-10 * torch.abs(target_emean):
                    break

            beta = 0.5 * (beta_lo + beta_hi)
            with torch.no_grad():
                expo = torch.clamp(beta * E, min=-80.0, max=80.0)
                W = support_weight * torch.exp(expo)
                S0 = torch.sum(W)

            if S0 > torch.finfo(f.dtype).tiny:
                amp = target_number / S0
                f_mult = f_pos * amp * torch.exp(expo)
                N_mult, E_mult, *_ = distribution_number_energy_moments(f_mult, q, a, m)
                rel_N = torch.abs(N_mult - target_number) / torch.clamp(torch.abs(target_number), min=1e-300)
                rel_E = torch.abs(E_mult - target_energy) / torch.clamp(torch.abs(target_energy), min=1e-300)
                if rel_N < 1e-8 and rel_E < 1e-8 and torch.isfinite(f_mult).all():
                    return f_mult

    R_N = number - target_number
    R_E = energy_moment - target_energy

    fmax = torch.max(torch.abs(f)).clamp(min=torch.finfo(f.dtype).tiny)
    f_rel = torch.abs(f) / fmax
    support = f_rel > support_rel

    if torch.count_nonzero(support) < 2:
        return f

    inv_pen = torch.where(
        support,
        torch.clamp(f_rel, min=floor_rel),
        torch.zeros_like(f_rel),
    )

    A00 = torch.sum(w * inv_pen)
    A01 = torch.sum(w * E * inv_pen)
    A11 = torch.sum(w * E * E * inv_pen)

    det = A00 * A11 - A01 * A01
    tiny = 100.0 * torch.finfo(f.dtype).eps
    if torch.abs(det) <= tiny * (torch.abs(A00 * A11) + 1.0):
        return f

    alpha = (R_N * A11 - R_E * A01) / det
    beta = (R_E * A00 - R_N * A01) / det
    return f - inv_pen * (alpha + beta * E)


# ============================================================
# Path / saving helpers
# ============================================================

def _make_results_path(path, results_dir="results", run_name=None):
    """
    Redirect relative output paths into a structured results directory.

    Examples
    --------
    out_path_pt="traj.pt", results_dir="results", run_name="run_001"
    -> results/run_001/traj.pt

    Absolute paths are respected.
    """
    if path is None:
        return None

    path = Path(path)

    if path.is_absolute():
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    base = Path(results_dir)
    if run_name is not None:
        base = base / str(run_name)

    base.mkdir(parents=True, exist_ok=True)
    return str(base / path)


def _save_metadata(metadata, results_dir="results", run_name=None, filename="metadata.json"):
    if metadata is None:
        return

    out = _make_results_path(
        filename,
        results_dir=results_dir,
        run_name=run_name,
    )

    with open(out, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def _torch_save_atomic(obj, path, atomic=True):
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)

    if not atomic:
        torch.save(obj, path)
        return

    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    os.close(fd)

    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _save_trajectory_dat(a_hist, f_hist, path, atomic=True, fmt="%.18e"):
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)

    a_np = a_hist.detach().cpu().numpy().reshape(-1, 1)
    f_np = f_hist.detach().cpu().numpy()
    arr = np.concatenate([a_np, f_np], axis=1)

    if not atomic:
        np.savetxt(path, arr, fmt=fmt)
        return

    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
    os.close(fd)

    try:
        np.savetxt(tmp_path, arr, fmt=fmt)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ============================================================
# RK4 in log(a), final state only
# ============================================================

@torch.no_grad()
def integrate_rk4_loga(
    f0, a0, a1, n_steps, rhs,
    clip_negative=False,
    clip_tol=0.0,
    return_clip_stats=False,
    stop_on_nonfinite=True,
    return_status=False,
):
    """
    RK4 integrator in u = log(a), for an RHS written as

        df/da = rhs(f, a)

    Internally uses

        df/du = a * rhs(f, a),   with a = exp(u)

    Parameters
    ----------
    f0 : tensor
        Initial distribution at a0.
    a0, a1 : float
        Initial and final scale factor.
    n_steps : int
        Number of uniform steps in log(a).
    rhs : callable
        Function rhs(f, a) returning df/da.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be > 0.")
    if a0 <= 0 or a1 <= 0:
        raise ValueError("a0 and a1 must be > 0 for log(a) integration.")

    f = f0.clone()
    device, dtype = f.device, f.dtype

    u0 = math.log(a0)
    u1 = math.log(a1)
    du = (u1 - u0) / n_steps

    du_t = torch.as_tensor(du, device=device, dtype=dtype)
    u = torch.as_tensor(u0, device=device, dtype=dtype)
    clip_tol_t = torch.as_tensor(clip_tol, device=device, dtype=dtype)

    if return_clip_stats:
        total_clipped_l1 = torch.zeros((), device=device, dtype=dtype)
        max_negative = torch.zeros((), device=device, dtype=dtype)
        n_clipped_steps = 0

    status = {
        "completed": True,
        "stopped_early": False,
        "reason": None,
        "step": n_steps,
        "u_last": u.detach().clone(),
        "a_last": torch.exp(u.detach().clone()),
    }

    def _stop(reason, step_idx, u_valid):
        status["completed"] = False
        status["stopped_early"] = True
        status["reason"] = reason
        status["step"] = step_idx
        status["u_last"] = u_valid.detach().clone()
        status["a_last"] = torch.exp(u_valid.detach().clone())

    def rhs_u(f_state, u_state):
        a_state = torch.exp(u_state)
        return a_state * rhs(f_state, a_state)

    for step in range(n_steps):
        f_prev = f.detach().clone()
        u_prev = u.detach().clone()

        k1 = rhs_u(f, u)
        if stop_on_nonfinite and (not torch.isfinite(k1).all()):
            f = f_prev
            u = u_prev
            _stop("nonfinite in k1", step, u_prev)
            break

        y2 = f + 0.5 * du_t * k1
        if stop_on_nonfinite and (not torch.isfinite(y2).all()):
            f = f_prev
            u = u_prev
            _stop("nonfinite in stage y2", step, u_prev)
            break

        k2 = rhs_u(y2, u + 0.5 * du_t)
        if stop_on_nonfinite and (not torch.isfinite(k2).all()):
            f = f_prev
            u = u_prev
            _stop("nonfinite in k2", step, u_prev)
            break

        y3 = f + 0.5 * du_t * k2
        if stop_on_nonfinite and (not torch.isfinite(y3).all()):
            f = f_prev
            u = u_prev
            _stop("nonfinite in stage y3", step, u_prev)
            break

        k3 = rhs_u(y3, u + 0.5 * du_t)
        if stop_on_nonfinite and (not torch.isfinite(k3).all()):
            f = f_prev
            u = u_prev
            _stop("nonfinite in k3", step, u_prev)
            break

        y4 = f + du_t * k3
        if stop_on_nonfinite and (not torch.isfinite(y4).all()):
            f = f_prev
            u = u_prev
            _stop("nonfinite in stage y4", step, u_prev)
            break

        k4 = rhs_u(y4, u + du_t)
        if stop_on_nonfinite and (not torch.isfinite(k4).all()):
            f = f_prev
            u = u_prev
            _stop("nonfinite in k4", step, u_prev)
            break

        f_new = f + (du_t / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        u_new = u + du_t

        if stop_on_nonfinite and (not torch.isfinite(f_new).all()):
            f = f_prev
            u = u_prev
            _stop("nonfinite in updated state", step, u_prev)
            break

        f = f_new
        u = u_new

        if clip_negative:
            neg_part = torch.clamp(clip_tol_t - f, min=0.0)

            if return_clip_stats:
                clipped_now = torch.sum(neg_part)
                total_clipped_l1 = total_clipped_l1 + clipped_now
                max_negative = torch.maximum(max_negative, torch.max(neg_part))

                if clipped_now.item() > 0:
                    n_clipped_steps += 1

            f = torch.clamp(f, min=clip_tol_t)

    if return_clip_stats and return_status:
        stats = {
            "total_clipped_l1": total_clipped_l1,
            "max_negative_before_clip": max_negative,
            "n_clipped_steps": n_clipped_steps,
        }
        return f, stats, status

    if return_clip_stats:
        stats = {
            "total_clipped_l1": total_clipped_l1,
            "max_negative_before_clip": max_negative,
            "n_clipped_steps": n_clipped_steps,
        }
        return f, stats

    if return_status:
        return f, status

    return f


# ============================================================
# RK4 in log(a), trajectory storage
# ============================================================

@torch.no_grad()
def integrate_rk4_loga_trajectory(
    f0, a0, a1, n_steps, rhs,
    store_every=1,
    print_every_pct=10,
    out_path_pt=None,
    out_path_dat=None,
    atomic=True,
    dat_fmt="%.18e",
    stop_on_nonfinite=True,
    results_dir="results",
    run_name=None,
    metadata=None,
):
    """
    RK4 integrator in u = log(a), storing snapshots.

    The RHS must be written as

        df/da = rhs(f, a)

    Internally evolves

        df/du = a * rhs(f, a),   with a = exp(u)

    Output paths
    ------------
    Relative output paths are redirected to:

        results_dir/run_name/out_path

    Example
    -------
    integrate_rk4_a_trajectory(
        ...,
        out_path_pt="trajectory.pt",
        out_path_dat="trajectory.dat",
        results_dir="results",
        run_name="N256_steps1000_lam5e-4",
        metadata={...},
    )

    creates:

        results/N256_steps1000_lam5e-4/trajectory.pt
        results/N256_steps1000_lam5e-4/trajectory.dat
        results/N256_steps1000_lam5e-4/metadata.json
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be > 0.")
    if store_every <= 0:
        raise ValueError("store_every must be > 0.")
    if a0 <= 0 or a1 <= 0:
        raise ValueError("a0 and a1 must be > 0 for log(a) integration.")

    f = f0.clone()
    device, dtype = f.device, f.dtype

    u0 = math.log(a0)
    u1 = math.log(a1)
    du = (u1 - u0) / n_steps

    du_t = torch.as_tensor(du, device=device, dtype=dtype)
    u = torch.as_tensor(u0, device=device, dtype=dtype)

    a_hist = [torch.exp(u.detach().clone())]
    u_hist = [u.detach().clone()]
    f_hist = [f.detach().clone()]

    if print_every_pct is not None and print_every_pct > 0:
        every = max(1, int(round(n_steps * print_every_pct / 100.0)))
    else:
        every = None

    status = {
        "completed": True,
        "stopped_early": False,
        "reason": None,
        "step": n_steps,
        "u_last": u.detach().clone(),
        "a_last": torch.exp(u.detach().clone()),
    }

    def _stop(reason, step_idx, u_valid):
        status["completed"] = False
        status["stopped_early"] = True
        status["reason"] = reason
        status["step"] = step_idx
        status["u_last"] = u_valid.detach().clone()
        status["a_last"] = torch.exp(u_valid.detach().clone())

    def rhs_u(f_state, u_state):
        a_state = torch.exp(u_state)
        return a_state * rhs(f_state, a_state)

    for step in range(n_steps):
        f_prev = f.detach().clone()
        u_prev = u.detach().clone()

        k1 = rhs_u(f, u)
        if stop_on_nonfinite and (not torch.isfinite(k1).all()):
            f = f_prev
            u = u_prev
            _stop("nonfinite in k1", step, u_prev)
            break

        y2 = f + 0.5 * du_t * k1
        if stop_on_nonfinite and (not torch.isfinite(y2).all()):
            f = f_prev
            u = u_prev
            _stop("nonfinite in stage y2", step, u_prev)
            break

        k2 = rhs_u(y2, u + 0.5 * du_t)
        if stop_on_nonfinite and (not torch.isfinite(k2).all()):
            f = f_prev
            u = u_prev
            _stop("nonfinite in k2", step, u_prev)
            break

        y3 = f + 0.5 * du_t * k2
        if stop_on_nonfinite and (not torch.isfinite(y3).all()):
            f = f_prev
            u = u_prev
            _stop("nonfinite in stage y3", step, u_prev)
            break

        k3 = rhs_u(y3, u + 0.5 * du_t)
        if stop_on_nonfinite and (not torch.isfinite(k3).all()):
            f = f_prev
            u = u_prev
            _stop("nonfinite in k3", step, u_prev)
            break

        y4 = f + du_t * k3
        if stop_on_nonfinite and (not torch.isfinite(y4).all()):
            f = f_prev
            u = u_prev
            _stop("nonfinite in stage y4", step, u_prev)
            break

        k4 = rhs_u(y4, u + du_t)
        if stop_on_nonfinite and (not torch.isfinite(k4).all()):
            f = f_prev
            u = u_prev
            _stop("nonfinite in k4", step, u_prev)
            break

        f_new = f + (du_t / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        u_new = u + du_t

        if stop_on_nonfinite and (not torch.isfinite(f_new).all()):
            f = f_prev
            u = u_prev
            _stop("nonfinite in updated state", step, u_prev)
            break

        f = f_new
        u = u_new

        should_store = ((step + 1) % store_every == 0) or (step == n_steps - 1)
        if should_store:
            a_hist.append(torch.exp(u.detach().clone()))
            u_hist.append(u.detach().clone())
            f_hist.append(f.detach().clone())

        if every is not None and (((step + 1) % every == 0) or (step == n_steps - 1)):
            pct = 100.0 * (step + 1) / n_steps
            print(
                f"[RK4 loga] step {step + 1}/{n_steps} "
                f"({pct:.1f}%)  a={float(torch.exp(u)):.6e}"
            )

    a_hist = torch.stack(a_hist, dim=0)
    u_hist = torch.stack(u_hist, dim=0)
    f_hist = torch.stack(f_hist, dim=0)

    result = {
        "u": u_hist,
        "a": a_hist,
        "f": f_hist,
        "u_final": u_hist[-1],
        "a_final": a_hist[-1],
        "f_final": f_hist[-1],
        "status": status,
    }

    out_path_pt = _make_results_path(
        out_path_pt,
        results_dir=results_dir,
        run_name=run_name,
    )
    out_path_dat = _make_results_path(
        out_path_dat,
        results_dir=results_dir,
        run_name=run_name,
    )

    if out_path_pt is not None:
        _torch_save_atomic(result, out_path_pt, atomic=atomic)

    if out_path_dat is not None:
        _save_trajectory_dat(a_hist, f_hist, out_path_dat, atomic=atomic, fmt=dat_fmt)

    _save_metadata(
        metadata,
        results_dir=results_dir,
        run_name=run_name,
    )

    return result


# ============================================================
# Backward-compatible wrappers
# ============================================================

@torch.no_grad()
def integrate_rk4_a(*args, **kwargs):
    """
    Backward-compatible wrapper.

    This now integrates uniformly in log(a), not linearly in a.
    """
    return integrate_rk4_loga(*args, **kwargs)


@torch.no_grad()
def integrate_rk4_a_trajectory(*args, **kwargs):
    """
    Backward-compatible wrapper.

    This now integrates uniformly in log(a), not linearly in a.
    """
    return integrate_rk4_loga_trajectory(*args, **kwargs)


@torch.no_grad()
def integrate_heun_adaptive_loga_trajectory(
    f0, a0, a1, rhs,
    du_init=1e-3,
    du_min=1e-10,
    du_max=0.05,
    rtol=1e-3,
    atol=1e-30,
    safety=0.8,
    growth_max=5.0,
    shrink_min=0.1,
    max_steps=100000,
    store_every_accepted=1,
    print_every_accepted=100,
    clip_negative=False,
    clip_tol=0.0,
    stop_on_nonfinite=True,
    post_step_projector=None,
    out_path_pt=None,
    out_path_dat=None,
    atomic=True,
    dat_fmt="%.18e",
    results_dir="results",
    run_name=None,
    metadata=None,
):
    if a0 <= 0 or a1 <= 0:
        raise ValueError("a0 and a1 must be > 0.")
    if du_init <= 0 or du_min <= 0 or du_max <= 0:
        raise ValueError("du_init, du_min, du_max must be > 0.")

    f = f0.clone()
    device, dtype = f.device, f.dtype

    u = torch.as_tensor(math.log(a0), device=device, dtype=dtype)
    u_end = torch.as_tensor(math.log(a1), device=device, dtype=dtype)

    du = torch.as_tensor(min(du_init, float(u_end - u)), device=device, dtype=dtype)
    du_min_t = torch.as_tensor(du_min, device=device, dtype=dtype)
    du_max_t = torch.as_tensor(du_max, device=device, dtype=dtype)

    atol_t = torch.as_tensor(atol, device=device, dtype=dtype)
    rtol_t = torch.as_tensor(rtol, device=device, dtype=dtype)
    clip_tol_t = torch.as_tensor(clip_tol, device=device, dtype=dtype)

    a_hist = [torch.exp(u.detach().clone())]
    u_hist = [u.detach().clone()]
    f_hist = [f.detach().clone()]

    n_accept = 0
    n_reject = 0

    status = {
        "completed": True,
        "stopped_early": False,
        "reason": None,
        "n_accept": 0,
        "n_reject": 0,
        "u_last": u.detach().clone(),
        "a_last": torch.exp(u.detach().clone()),
        "last_err": None,
        "last_du": None,
        "last_reject_factor": None,
        "last_proposed_du": None,
    }

    def rhs_u(f_state, u_state):
        a_state = torch.exp(u_state)
        return a_state * rhs(f_state, a_state)

    def stop(reason):
        status["completed"] = False
        status["stopped_early"] = True
        status["reason"] = reason
        status["n_accept"] = n_accept
        status["n_reject"] = n_reject
        status["u_last"] = u.detach().clone()
        status["a_last"] = torch.exp(u.detach().clone())

    for step in range(max_steps):
        if u >= u_end:
            break

        if u + du > u_end:
            du = u_end - u

        f_old = f.detach().clone()
        u_old = u.detach().clone()

        k1 = rhs_u(f, u)

        if stop_on_nonfinite and (not torch.isfinite(k1).all()):
            stop("nonfinite in k1")
            break

        f_euler = f + du * k1

        if clip_negative:
            f_euler = torch.clamp(f_euler, min=clip_tol_t)

        if post_step_projector is not None:
            f_euler = post_step_projector(
                f_old=f_old,
                f_candidate=f_euler,
                u_old=u_old,
                u_new=u + du,
                du=du,
            )
            if clip_negative:
                f_euler = torch.clamp(f_euler, min=clip_tol_t)

        if stop_on_nonfinite and (not torch.isfinite(f_euler).all()):
            stop("nonfinite in Euler predictor")
            break

        k2 = rhs_u(f_euler, u + du)

        if stop_on_nonfinite and (not torch.isfinite(k2).all()):
            stop("nonfinite in k2")
            break

        f_heun = f + 0.5 * du * (k1 + k2)

        if clip_negative:
            f_heun = torch.clamp(f_heun, min=clip_tol_t)

        if post_step_projector is not None:
            f_heun = post_step_projector(
                f_old=f_old,
                f_candidate=f_heun,
                u_old=u_old,
                u_new=u + du,
                du=du,
            )
            if clip_negative:
                f_heun = torch.clamp(f_heun, min=clip_tol_t)

        if stop_on_nonfinite and (not torch.isfinite(f_heun).all()):
            stop("nonfinite in Heun update")
            break

        scale = atol_t + rtol_t * torch.maximum(torch.abs(f_heun), torch.abs(f_old))
        err_vec = torch.abs(f_heun - f_euler) / scale
        err = torch.max(err_vec)

        if not torch.isfinite(err):
            stop("nonfinite error estimate")
            break

        if err <= 1.0:
            f = f_heun
            u = u + du
            n_accept += 1
            status["last_err"] = err.detach().clone()
            status["last_du"] = du.detach().clone()

            if (n_accept % store_every_accepted == 0) or (u >= u_end):
                a_hist.append(torch.exp(u.detach().clone()))
                u_hist.append(u.detach().clone())
                f_hist.append(f.detach().clone())

            if print_every_accepted is not None and print_every_accepted > 0:
                if (n_accept % print_every_accepted == 0) or (u >= u_end):
                    print(
                        f"[Heun adaptive loga] accept={n_accept} reject={n_reject} "
                        f"a={float(torch.exp(u)):.6e} du={float(du):.3e} err={float(err):.3e}"
                    )

            if err == 0:
                factor = growth_max
            else:
                factor = min(growth_max, max(shrink_min, safety * float(err) ** (-0.5)))

            du = torch.minimum(du_max_t, du * torch.as_tensor(factor, device=device, dtype=dtype))

        else:
            n_reject += 1
            factor = max(shrink_min, safety * float(err) ** (-0.5))
            du_old = du.detach().clone()
            du = du * torch.as_tensor(factor, device=device, dtype=dtype)
            status["last_err"] = err.detach().clone()
            status["last_du"] = du_old
            status["last_reject_factor"] = factor
            status["last_proposed_du"] = du.detach().clone()

            f = f_old
            u = u_old

            if du < du_min_t:
                stop("du below du_min")
                break

    else:
        stop("max_steps reached")

    if status["completed"]:
        status["n_accept"] = n_accept
        status["n_reject"] = n_reject
        status["u_last"] = u.detach().clone()
        status["a_last"] = torch.exp(u.detach().clone())

    a_hist = torch.stack(a_hist, dim=0)
    u_hist = torch.stack(u_hist, dim=0)
    f_hist = torch.stack(f_hist, dim=0)

    result = {
        "u": u_hist,
        "a": a_hist,
        "f": f_hist,
        "u_final": u_hist[-1],
        "a_final": a_hist[-1],
        "f_final": f_hist[-1],
        "status": status,
    }

    out_path_pt = _make_results_path(
        out_path_pt,
        results_dir=results_dir,
        run_name=run_name,
    )
    out_path_dat = _make_results_path(
        out_path_dat,
        results_dir=results_dir,
        run_name=run_name,
    )

    if out_path_pt is not None:
        _torch_save_atomic(result, out_path_pt, atomic=atomic)

    if out_path_dat is not None:
        _save_trajectory_dat(a_hist, f_hist, out_path_dat, atomic=atomic, fmt=dat_fmt)

    _save_metadata(
        metadata,
        results_dir=results_dir,
        run_name=run_name,
    )

    return result

@torch.no_grad()
def integrate_heun_adaptive_a_trajectory(*args, **kwargs):
    return integrate_heun_adaptive_loga_trajectory(*args, **kwargs)


# ============================================================
# Hybrid driver: FI-only chunks until self-scattering matters,
# then one global adaptive Heun solve from a_switch to af
# ============================================================

@torch.no_grad()
def run_hybrid_FI_then_adaptive_self(
    *,
    f0,
    a0,
    af,
    q,
    m_chi,
    H_of_a,
    T_of_a,
    m_h,
    g_trilinear,
    nX_of_a,
    Gamma_X,
    mX,
    lam_self,
    C_self_operator=C_self_torch_logq,
    batch_size=64,
    Ng=12,

    # Hybrid control
    n_windows=400,
    gamma_over_H_on=0.1,
    gamma_check_every_far=50,
    gamma_check_every_mid=20,
    gamma_check_every_near=5,
    rk4_steps_per_window=2,
    rk4_store_every_steps=None,

    # Adaptive Heun settings once self turns on
    heun_du_init=1e-3,
    heun_du_min=1e-6,
    heun_du_max=0.1,
    heun_rtol=1e-2,
    heun_atol=1e-14,
    heun_safety=0.95,
    heun_store_every_accepted=200,
    heun_print_every_accepted=50,

    # Stability
    clip_negative=True,
    clip_tol=0.0,

    # Saving
    out_path_pt=None,
    out_path_dat=None,
    results_dir="results",
    run_name=None,
    metadata=None,
    atomic=True,
    dat_fmt="%.18e",
):
    if a0 <= 0 or af <= 0:
        raise ValueError("a0 and af must be > 0.")
    if n_windows <= 0:
        raise ValueError("n_windows must be > 0.")
    if gamma_over_H_on <= 0:
        raise ValueError("gamma_over_H_on must be > 0.")
    if (
        gamma_check_every_far <= 0
        or gamma_check_every_mid <= 0
        or gamma_check_every_near <= 0
    ):
        raise ValueError("Gamma/H check intervals must be > 0.")
    if rk4_steps_per_window <= 0:
        raise ValueError("rk4_steps_per_window must be > 0.")

    if rk4_store_every_steps is None:
        rk4_store_every_steps = rk4_steps_per_window

    if rk4_store_every_steps <= 0:
        raise ValueError("rk4_store_every_steps must be > 0.")

    device, dtype = f0.device, f0.dtype

    # ------------------------------------------------------------
    # Self-scattering collision operator
    #
    # Built-in choices C_self_torch_logq and
    # C_self_torch_logq_conservative_scatter share this interface, so
    # either can be passed as C_self_operator for comparison runs.
    # ------------------------------------------------------------
    Cself_full = partial(
        C_self_operator,
        lam=lam_self,
        Ng=Ng,
        batch_size=batch_size,
        return_diagnostics=False,
    )

    # ------------------------------------------------------------
    # FI-only RHS
    # ------------------------------------------------------------
    rhs_FI = partial(
        rhs_df_da_torch_logq_FI,
        q=q,
        m_chi=m_chi,
        H_of_a=H_of_a,
        T_of_a=T_of_a,
        m_h=m_h,
        g_trilinear=g_trilinear,
        n_parent2_of_a=nX_of_a,
        Gamma_parent2=Gamma_X,
        m_h2=mX,
        m_other2=m_chi,
        multiplicity2=2.0,
        gchi=1.0,
        pref_FI=1.0,
    )

    # ------------------------------------------------------------
    # Full RHS = FI + self-scattering
    # ------------------------------------------------------------
    rhs_full = partial(
        rhs_df_da_torch_logq_generic,
        q=q,
        m_chi=m_chi,
        H_of_a=H_of_a,
        C_self_func=Cself_full,
        T_of_a=T_of_a,
        m_h=m_h,
        g_trilinear=g_trilinear,
        n_parent2_of_a=nX_of_a,
        Gamma_parent2=Gamma_X,
        m_h2=mX,
        m_other2=m_chi,
        multiplicity2=2.0,
        gchi=1.0,
        pref_FI=1.0,
    )

    def project_full_step_to_source_moments(f_old, f_candidate, u_old, u_new, du):
        a_old = torch.exp(u_old)
        a_new = torch.exp(u_new)

        # Source-only update over the same accepted trial step.  Self-scattering
        # is deliberately excluded because it must not change these moments.
        k_src_old = a_old * rhs_FI(f_old, a_old)
        k_src_new = a_new * rhs_FI(f_old, a_new)
        f_target = f_old + 0.5 * du * (k_src_old + k_src_new)

        target_N, target_E, _, _, _ = distribution_number_energy_moments(
            f_target,
            q=q,
            a=a_new,
            m=m_chi,
        )

        return project_distribution_to_number_energy(
            f=f_candidate,
            q=q,
            a=a_new,
            m=m_chi,
            target_number=target_N,
            target_energy=target_E,
        )

    u_grid = torch.linspace(
        math.log(a0),
        math.log(af),
        n_windows + 1,
        device=device,
        dtype=dtype,
    )
    a_grid = torch.exp(u_grid)

    f = f0.clone()

    a_hist = [torch.as_tensor(a0, device=device, dtype=dtype)]
    u_hist = [torch.as_tensor(math.log(a0), device=device, dtype=dtype)]
    f_hist = [f.detach().clone()]

    mode_hist = ["initial"]
    gamma_over_H_hist = [0.0]
    gamma_check_a_hist = [torch.as_tensor(a0, device=device, dtype=dtype)]

    switched = False
    a_switch = None
    heun_status = None
    rk4_status_hist = []
    last_gamma_over_H = 0.0
    next_gamma_check_iw = 0

    def gamma_check_interval(gamma_over_H):
        ratio = gamma_over_H / gamma_over_H_on

        if ratio < 1.0e-4:
            return int(gamma_check_every_far)
        if ratio < 1.0e-2:
            return int(gamma_check_every_mid)
        if ratio < 1.0e-1:
            return int(gamma_check_every_near)
        return 1

    for iw in range(n_windows):
        a_left = float(a_grid[iw].detach().cpu())
        a_right = float(a_grid[iw + 1].detach().cpu())

        checked_gamma_now = iw >= next_gamma_check_iw
        if checked_gamma_now:
            diag_self = estimate_gamma_eff_from_current_f(
                f_t=f,
                a_star=a_left,
                q=q,
                m_chi=m_chi,
                C_self_func=Cself_full,
                H_of_a=H_of_a,
            )

            last_gamma_over_H = float(diag_self["Gamma_over_H"])
            gamma_over_H_hist.append(last_gamma_over_H)
            gamma_check_a_hist.append(torch.as_tensor(a_left, device=device, dtype=dtype))
            next_gamma_check_iw = iw + gamma_check_interval(last_gamma_over_H)

        gamma_over_H = last_gamma_over_H

        if gamma_over_H < gamma_over_H_on:
            traj_rk4 = integrate_rk4_a_trajectory(
                f0=f,
                a0=a_left,
                a1=a_right,
                n_steps=rk4_steps_per_window,
                rhs=rhs_FI,
                store_every=rk4_store_every_steps,
                print_every_pct=None,
                out_path_pt=None,
                out_path_dat=None,
                results_dir=results_dir,
                run_name=None,
                metadata=None,
                stop_on_nonfinite=True,
            )

            f = traj_rk4["f_final"]
            rk4_status_hist.append(traj_rk4["status"])

            if traj_rk4["a"].numel() > 1:
                for aa, uu, ff in zip(
                    traj_rk4["a"][1:],
                    traj_rk4["u"][1:],
                    traj_rk4["f"][1:],
                ):
                    a_hist.append(aa.detach().clone())
                    u_hist.append(uu.detach().clone())
                    f_hist.append(ff.detach().clone())
                    mode_hist.append("FI_RK4")

            if clip_negative:
                f = torch.clamp(
                    f,
                    min=torch.as_tensor(clip_tol, device=device, dtype=dtype),
                )

            if (iw % 10 == 0) or (iw == n_windows - 1):
                gamma_tag = "checked" if checked_gamma_now else "cached"
                print(
                    f"[hybrid] window {iw + 1}/{n_windows} "
                    f"a={a_right:.6e} mode=FI_RK4 "
                    f"Gamma/H={gamma_over_H:.3e} ({gamma_tag}) "
                    f"finite={torch.isfinite(f).all().item()}"
                )

            if not traj_rk4["status"]["completed"]:
                print(
                    "[hybrid] RK4 stopped early: "
                    f"status={traj_rk4['status']}"
                )
                break

        else:
            switched = True
            a_switch = a_left

            print(
                f"[hybrid] switching to global adaptive Heun at "
                f"window {iw + 1}/{n_windows}, "
                f"a_switch={a_switch:.6e}, Gamma/H={gamma_over_H:.3e}, "
                f"C_self_operator={C_self_operator.__name__}"
            )

            traj_heun = integrate_heun_adaptive_a_trajectory(
                f0=f,
                a0=a_left,
                a1=af,
                rhs=rhs_full,
                du_init=heun_du_init,
                du_min=heun_du_min,
                du_max=heun_du_max,
                rtol=heun_rtol,
                atol=heun_atol,
                safety=heun_safety,
                store_every_accepted=heun_store_every_accepted,
                print_every_accepted=heun_print_every_accepted,
                out_path_pt=None,
                out_path_dat=None,
                results_dir=results_dir,
                run_name=None,
                metadata=None,
                clip_negative=clip_negative,
                clip_tol=clip_tol,
                stop_on_nonfinite=True,
                post_step_projector=project_full_step_to_source_moments,
            )

            f = traj_heun["f_final"]
            heun_status = traj_heun["status"]

            if traj_heun["a"].numel() > 1:
                for aa, uu, ff in zip(
                    traj_heun["a"][1:],
                    traj_heun["u"][1:],
                    traj_heun["f"][1:],
                ):
                    a_hist.append(aa.detach().clone())
                    u_hist.append(uu.detach().clone())
                    f_hist.append(ff.detach().clone())
                    mode_hist.append("FULL_HEUN")

            print(
                f"[hybrid] global adaptive Heun finished: "
                f"a_final={float(traj_heun['a_final']):.6e}, "
                f"finite={torch.isfinite(f).all().item()}, "
                f"status={heun_status}"
            )

            break

    if not switched:
        print(
            "[hybrid] self-scattering never crossed threshold; "
            "finished with FI-only RK4."
        )

    a_hist = torch.stack(a_hist)
    u_hist = torch.stack(u_hist)
    f_hist = torch.stack(f_hist)

    result = {
        "u": u_hist,
        "a": a_hist,
        "f": f_hist,
        "u_final": u_hist[-1],
        "a_final": a_hist[-1],
        "f_final": f_hist[-1],

        "mode_hist": mode_hist,
        "gamma_over_H_hist": gamma_over_H_hist,
        "gamma_check_a_hist": torch.stack(gamma_check_a_hist),

        "a_switch": None
        if a_switch is None
        else torch.as_tensor(a_switch, device=device, dtype=dtype),

        "heun_status": heun_status,
        "rk4_status_hist": rk4_status_hist,

        "settings": {
            "n_windows": n_windows,
            "gamma_over_H_on": gamma_over_H_on,
            "gamma_check_every_far": gamma_check_every_far,
            "gamma_check_every_mid": gamma_check_every_mid,
            "gamma_check_every_near": gamma_check_every_near,
            "rk4_steps_per_window": rk4_steps_per_window,
            "rk4_store_every_steps": rk4_store_every_steps,
            "lam_self": float(lam_self),
            "C_self_operator": C_self_operator.__name__,
            "Ng": Ng,
            "batch_size": batch_size,
            "heun_du_init": heun_du_init,
            "heun_du_min": heun_du_min,
            "heun_du_max": heun_du_max,
            "heun_rtol": heun_rtol,
            "heun_atol": heun_atol,
            "heun_safety": heun_safety,
            "heun_store_every_accepted": heun_store_every_accepted,
            "heun_project_source_moments": True,
        },
    }

    out_path_pt = _make_results_path(
        out_path_pt,
        results_dir=results_dir,
        run_name=run_name,
    )

    out_path_dat = _make_results_path(
        out_path_dat,
        results_dir=results_dir,
        run_name=run_name,
    )

    if out_path_pt is not None:
        _torch_save_atomic(result, out_path_pt, atomic=atomic)

    if out_path_dat is not None:
        _save_trajectory_dat(
            result["a"],
            result["f"],
            out_path_dat,
            atomic=atomic,
            fmt=dat_fmt,
        )

    _save_metadata(
        metadata,
        results_dir=results_dir,
        run_name=run_name,
    )

    return result
