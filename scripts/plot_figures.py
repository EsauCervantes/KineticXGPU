#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm, Normalize
from scipy.special import kve


THIS_FILE = Path(__file__).resolve()
SCRIPTS_DIR = THIS_FILE.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cosmology import VariableGCosmology
from cBE_solver import solve_condensate_N_loga_quad
from thermodynamics import Gamma_htophiphi


def project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def set_plot_style(usetex=False):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "axes.unicode_minus": False,
            "text.usetex": bool(usetex),
        }
    )


def tensor_to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def positive(values, floor=1e-300):
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values) & (values > 0.0), np.maximum(values, floor), np.nan)


def clip_for_log(values, floor=1e-5):
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values) & (values >= floor), values, np.nan)


def floor_for_log(values, floor=1e-5):
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values) & (values > 0.0), np.maximum(values, floor), np.nan)


def grid_edges_from_centers_log(q):
    q = np.asarray(q, dtype=float)
    if q.ndim != 1 or q.size < 2:
        raise ValueError("The q grid must be one-dimensional with at least two points.")

    edges = np.empty(q.size + 1, dtype=float)
    edges[1:-1] = np.sqrt(q[:-1] * q[1:])
    edges[0] = q[0] / np.sqrt(q[1] / q[0])
    edges[-1] = q[-1] * np.sqrt(q[-1] / q[-2])
    return edges


def reconstruct_q(metadata):
    return np.logspace(
        np.log10(float(metadata["qmin"])),
        np.log10(float(metadata["qmax"])),
        int(metadata["N_grid"]),
    )


def load_fbe_run(run_dir, map_location="cpu"):
    run_dir = project_path(run_dir)
    traj_path = run_dir / "trajectory.pt"
    metadata_path = run_dir / "metadata.json"

    if not traj_path.exists():
        raise FileNotFoundError(f"Missing {traj_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {metadata_path}")

    try:
        traj = torch.load(traj_path, map_location=map_location, weights_only=False)
    except TypeError:
        traj = torch.load(traj_path, map_location=map_location)

    with metadata_path.open("r") as f:
        metadata = json.load(f)

    return {
        "name": run_dir.name,
        "run_dir": run_dir,
        "traj": traj,
        "metadata": metadata,
        "q": reconstruct_q(metadata),
    }


def load_fbe_runs(runs_dir, run_prefix=None, map_location="cpu", max_runs=None):
    runs_dir = project_path(runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Missing fBE runs directory: {runs_dir}")

    runs = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_prefix and not run_dir.name.startswith(run_prefix):
            continue
        if not (run_dir / "trajectory.pt").exists():
            continue
        runs.append(load_fbe_run(run_dir, map_location=map_location))

    if not runs:
        detail = f" with prefix {run_prefix!r}" if run_prefix else ""
        raise FileNotFoundError(f"No fBE runs found in {runs_dir}{detail}.")

    runs.sort(key=lambda run: float(run["metadata"].get("lambda_self", 0.0)))
    if max_runs is not None and len(runs) > max_runs:
        idx = np.unique(np.linspace(0, len(runs) - 1, max_runs, dtype=int))
        runs = [runs[i] for i in idx]
    return runs


def load_cbe(cbe_dir):
    cbe_dir = project_path(cbe_dir)
    data_path = cbe_dir / "cbe_solution.npz"
    metadata_path = cbe_dir / "metadata.json"

    if not data_path.exists():
        return None

    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r") as f:
            metadata = json.load(f)

    return {"run_dir": cbe_dir, "data": np.load(data_path), "metadata": metadata}


def lambda_from_run(run):
    return float(run["metadata"].get("lambda_self", np.nan))


def x_from_a(a, m_chi, cosmo):
    a_arr = np.asarray(a, dtype=float)
    flat = np.atleast_1d(a_arr).reshape(-1)
    T = np.array([float(cosmo.T_of_a(float(ai))) for ai in flat], dtype=float)
    x = float(m_chi) / np.maximum(T, 1e-300)
    return x.reshape(np.atleast_1d(a_arr).shape).item() if a_arr.ndim == 0 else x.reshape(a_arr.shape)


def entropy_density(cosmo, a):
    return float(cosmo.entropy_density(float(cosmo.T_of_a(float(a)))))


def fbe_arrays(run):
    traj = run["traj"]
    if "a" not in traj or "f" not in traj:
        raise KeyError(f"{run['name']} trajectory must contain 'a' and 'f'.")
    return tensor_to_numpy(traj["a"]).astype(float), tensor_to_numpy(traj["f"]).astype(float)


def first_full_solver_index(run):
    traj = run["traj"]
    mode_hist = traj.get("mode_hist")
    if mode_hist is None:
        return 0

    for i, mode in enumerate(mode_hist):
        text = mode.decode() if isinstance(mode, bytes) else str(mode)
        if "FULL" in text or "HEUN" in text:
            return min(i, len(tensor_to_numpy(traj["a"])) - 1)
    return 0


def normalized_distribution(q, f):
    y = np.asarray(q, dtype=float) ** 3 * np.asarray(f, dtype=float)
    denom = np.trapezoid(y, x=np.log(q))
    if not np.isfinite(denom) or denom <= 0.0:
        return np.full_like(y, np.nan, dtype=float)
    y = y / denom
    return np.where(np.isfinite(y) & (y >= 0.0), y, np.nan)


def mb_distribution_from_n_T(p, m, n, T):
    p = np.asarray(p, dtype=float)
    m = float(m)
    n = float(n)
    T = float(T)
    if T <= 0.0 or n <= 0.0:
        return np.zeros_like(p)

    x = m / T
    E = np.sqrt(p * p + m * m)
    amp = n * (2.0 * np.pi**2) / max(m * m * T * kve(2, x), 1e-300)
    return amp * np.exp(-(E - m) / T)


def cbe_distribution_at_a(cbe, q, a, m_chi):
    if cbe is None:
        return None

    data = cbe["data"]
    a_cbe = np.asarray(data["a"], dtype=float)
    N_cbe = np.asarray(data["N"], dtype=float)
    T_cbe = np.asarray(data["T"], dtype=float)
    a = float(a)
    m_chi = float(m_chi)

    mask = np.isfinite(a_cbe) & np.isfinite(N_cbe) & np.isfinite(T_cbe) & (a_cbe > 0.0)
    if np.count_nonzero(mask) < 2:
        return None

    a_cbe = a_cbe[mask]
    N_cbe = N_cbe[mask]
    T_cbe = T_cbe[mask]
    order = np.argsort(a_cbe)
    a_cbe = a_cbe[order]
    N_cbe = N_cbe[order]
    T_cbe = T_cbe[order]

    N_val = interp_positive(np.array([a]), a_cbe, N_cbe)[0]
    T_val = interp_positive(np.array([a]), a_cbe, T_cbe)[0]
    if not np.isfinite(N_val):
        N_val = np.interp(np.log(a), np.log(a_cbe), N_cbe)
    if not np.isfinite(T_val):
        T_val = np.interp(np.log(a), np.log(a_cbe), T_cbe)

    n_val = float(N_val) / max(a**3, 1e-300)
    p = np.asarray(q, dtype=float) / a
    return mb_distribution_from_n_T(p=p, m=m_chi, n=n_val, T=T_val)


def moments_from_snapshot(q, f, a, m, g=1.0):
    q = np.asarray(q, dtype=float)
    f = np.asarray(f, dtype=float)
    a = float(a)
    m = float(m)

    dq = np.diff(grid_edges_from_centers_log(q))
    p = q / a
    E = np.sqrt(p * p + m * m)
    v = p / np.maximum(E, 1e-300)
    q2f_dq = q * q * f * dq
    n_integral = np.sum(q2f_dq)
    pref = float(g) / (2.0 * np.pi**2)

    n = pref * n_integral / max(a**3, 1e-300)
    rho = pref * np.sum(q2f_dq * E) / max(a**3, 1e-300)
    pressure = pref * np.sum(q2f_dq * p * p / (3.0 * np.maximum(E, 1e-300))) / max(a**3, 1e-300)

    avg_v = np.sum(q2f_dq * v) / max(n_integral, 1e-300)
    avg_v2 = np.sum(q2f_dq * v * v) / max(n_integral, 1e-300)

    return {
        "n": n,
        "N": pref * n_integral,
        "rho": rho,
        "P": pressure,
        "Tchi": pressure / max(n, 1e-300),
        "avg_v": avg_v,
        "v_rms": np.sqrt(max(avg_v2, 0.0)),
    }


def fbe_curves(run, cosmo):
    metadata = run["metadata"]
    q = run["q"]
    m = float(metadata["m_chi"])
    g = float(metadata.get("g_chi", 1.0))
    a, f = fbe_arrays(run)

    rows = [moments_from_snapshot(q, fi, ai, m, g=g) for ai, fi in zip(a, f)]
    x = np.asarray(x_from_a(a, m, cosmo), dtype=float)
    s = np.array([entropy_density(cosmo, ai) for ai in a], dtype=float)

    out = {
        "a": a,
        "x": x,
        "n": np.array([row["n"] for row in rows], dtype=float),
        "N": np.array([row["N"] for row in rows], dtype=float),
        "Y": np.array([row["n"] for row in rows], dtype=float) / np.maximum(s, 1e-300),
        "Tchi": np.array([row["Tchi"] for row in rows], dtype=float),
        "avg_v": np.array([row["avg_v"] for row in rows], dtype=float),
        "v_rms": np.array([row["v_rms"] for row in rows], dtype=float),
    }

    order = np.argsort(out["x"])
    for key in list(out):
        out[key] = np.asarray(out[key])[order]
    return out


def cbe_velocity_moments(T, m_chi, n_p=96):
    T = np.asarray(T, dtype=float)
    m = max(float(m_chi), 1e-300)
    y_nodes, y_weights = np.polynomial.laguerre.laggauss(n_p)
    avg_v = np.full_like(T, np.nan, dtype=float)
    v_rms = np.full_like(T, np.nan, dtype=float)

    for idx, T_val in np.ndenumerate(T):
        if not np.isfinite(T_val) or T_val <= 0.0:
            continue

        z = m / T_val
        k2e = kve(2, z)
        denom = m * m * T_val * max(k2e, 1e-300)
        avg_v[idx] = 2.0 * T_val**2 * (m + T_val) / denom

        E = m + T_val * y_nodes
        p2 = np.maximum(T_val * y_nodes * (2.0 * m + T_val * y_nodes), 0.0)
        p = np.sqrt(p2)
        avg_v2 = T_val * np.sum(y_weights * p2 * p / np.maximum(E, 1e-300)) / denom
        v_rms[idx] = np.sqrt(max(avg_v2, 0.0))

    return avg_v, v_rms


def cbe_curves(cbe, m_chi, cosmo):
    if cbe is None:
        return None

    data = cbe["data"]
    a = np.asarray(data["a"], dtype=float)
    N = np.asarray(data["N"], dtype=float)
    Tchi = np.asarray(data["T"], dtype=float)
    x = np.asarray(x_from_a(a, m_chi, cosmo), dtype=float)
    n = N / np.maximum(a**3, 1e-300)
    s = np.array([entropy_density(cosmo, ai) for ai in a], dtype=float)
    avg_v, v_rms = cbe_velocity_moments(Tchi, m_chi)

    out = {
        "a": a,
        "x": x,
        "N": N,
        "n": n,
        "Y": n / np.maximum(s, 1e-300),
        "Tchi": Tchi,
        "avg_v": avg_v,
        "v_rms": v_rms,
    }

    order = np.argsort(out["x"])
    for key in list(out):
        out[key] = np.asarray(out[key])[order]
    return out


def interp_positive(x_new, x_old, y_old):
    x_old = np.asarray(x_old, dtype=float)
    y_old = np.asarray(y_old, dtype=float)
    mask = np.isfinite(x_old) & np.isfinite(y_old) & (x_old > 0.0) & (y_old > 0.0)
    if np.count_nonzero(mask) < 2:
        return np.full_like(np.asarray(x_new, dtype=float), np.nan)
    return np.exp(np.interp(np.log(x_new), np.log(x_old[mask]), np.log(y_old[mask]), left=np.nan, right=np.nan))


def color_norm(values, log=True):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if vmin == vmax:
        delta = abs(vmin) * 1e-6 + 1e-12
        vmin -= delta
        vmax += delta
    if log and vmin > 0.0:
        return LogNorm(vmin=max(vmin, 1e-300), vmax=max(vmax, 1e-300))
    return Normalize(vmin=vmin, vmax=vmax)


def add_lambda_colorbar(fig, ax, cmap, norm):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.ax.set_title(r"$\lambda$", pad=8)
    return cbar


def save_figure(fig, out_dir, name, formats):
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_dir / f"{name}.{fmt}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def lambda_filename_value(lam):
    return f"{float(lam):.1e}".replace("+", "")


def plot_evolution_for_run(run, cosmo, out_dir, formats, n_snapshots, x_start=1.0):
    metadata = run["metadata"]
    q = run["q"]
    m = float(metadata["m_chi"])
    a, f = fbe_arrays(run)

    x_all = np.asarray(x_from_a(a, m, cosmo), dtype=float)
    start = first_full_solver_index(run)
    if x_start is not None:
        x_candidates = np.where(x_all >= float(x_start))[0]
        if x_candidates.size:
            start = max(start, int(x_candidates[0]))
    if start >= len(a) - 1:
        start = 0
    indices = np.unique(np.linspace(start, len(a) - 1, min(n_snapshots, len(a) - start), dtype=int))
    x_vals = np.asarray(x_from_a(a[indices], m, cosmo), dtype=float)

    cmap = plt.get_cmap("inferno_r")
    norm = color_norm(x_vals, log=True)
    fig, ax = plt.subplots(figsize=(6.7, 4.5))

    for idx, x_val in zip(indices, x_vals):
        ax.plot(
            q / m,
            normalized_distribution(q, f[idx]),
            color=cmap(norm(x_val)),
            lw=1.5,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.ax.set_title(r"$x$", pad=8)

    ax.set_xscale("log")
    ax.set_xlabel(r"$q/m_\chi$")
    ax.set_ylabel(r"$N^{-1}\,dN/d\ln q$")
    ax.set_ylim(bottom=0.0)
    ax.text(
        0.04,
        0.94,
        rf"$\lambda={lambda_from_run(run):.1e}$",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9, "pad": 3.0},
    )
    lam_text = lambda_filename_value(lambda_from_run(run))
    save_figure(fig, out_dir, f"evolution_lambda_{lam_text}", formats)


def plot_all_evolutions(runs, cosmo, out_dir, formats, n_snapshots, x_start=1.0):
    for run in runs:
        plot_evolution_for_run(run, cosmo, out_dir, formats, n_snapshots, x_start=x_start)


def plot_fig5_final_distributions(runs, cbe, cosmo, out_dir, formats):
    lambdas = np.array([lambda_from_run(run) for run in runs], dtype=float)
    cmap = plt.get_cmap("jet")
    norm = color_norm(lambdas, log=True)

    fig, ax = plt.subplots(figsize=(6.7, 4.5))
    x_final_values = []

    for run in runs:
        q = run["q"]
        metadata = run["metadata"]
        m = float(metadata["m_chi"])
        a, f = fbe_arrays(run)
        x_final_values.append(float(x_from_a(a[-1], m, cosmo)))
        ax.plot(
            q / m,
            normalized_distribution(q, f[-1]),
            color=cmap(norm(lambda_from_run(run))),
            lw=1.8,
        )

    ref_run = runs[-1]
    ref_q = ref_run["q"]
    ref_m = float(ref_run["metadata"]["m_chi"])
    ref_a, _ = fbe_arrays(ref_run)
    f_cbe = cbe_distribution_at_a(cbe, ref_q, ref_a[-1], ref_m)
    if f_cbe is not None:
        ax.plot(
            ref_q / ref_m,
            normalized_distribution(ref_q, f_cbe),
            color="black",
            ls="--",
            lw=2.0,
            label=r"cBE",
        )

    add_lambda_colorbar(fig, ax, cmap, norm)
    if f_cbe is not None:
        ax.legend(frameon=False, loc="best")
    xf = float(np.nanmedian(x_final_values))
    ax.text(
        0.04,
        0.94,
        rf"$x_f={xf:.2g}$",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9, "pad": 3.0},
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"$q/m_\chi$")
    ax.set_ylabel(r"$N^{-1}\,dN/d\ln q$")
    ax.set_ylim(bottom=0.0)
    save_figure(fig, out_dir, "final_distributions", formats)


def moments_x_min_for_run(run, x_min=None, x_min_factor=1.0 + 1.0 / 15.0):
    if x_min is not None:
        return float(x_min)
    metadata = run["metadata"]
    if "x_initial" in metadata:
        return float(x_min_factor) * float(metadata["x_initial"])
    return None


def plot_fig6_velocity_ratios(runs, cbe, cosmo, out_dir, formats, x_min=None, x_min_factor=1.0 + 1.0 / 15.0):
    if cbe is None:
        raise FileNotFoundError("Fig. 6 needs a cBE solution.")

    m = float(runs[0]["metadata"]["m_chi"])
    cbe_data = cbe_curves(cbe, m, cosmo)
    lambdas = np.array([lambda_from_run(run) for run in runs], dtype=float)
    cmap = plt.get_cmap("jet")
    norm = color_norm(lambdas, log=True)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), sharex=True)
    fig.subplots_adjust(wspace=0.28)

    for run in runs:
        curves = fbe_curves(run, cosmo)
        mask = np.isfinite(curves["x"])
        x_cut = moments_x_min_for_run(run, x_min=x_min, x_min_factor=x_min_factor)
        if x_cut is not None:
            mask &= curves["x"] >= x_cut
        color = cmap(norm(lambda_from_run(run)))
        x_plot = curves["x"][mask]
        avg_ref = interp_positive(x_plot, cbe_data["x"], cbe_data["avg_v"])
        rms_ref = interp_positive(x_plot, cbe_data["x"], cbe_data["v_rms"])
        axes[0].semilogx(x_plot, curves["avg_v"][mask] / avg_ref, color=color, lw=1.8)
        axes[1].semilogx(x_plot, curves["v_rms"][mask] / rms_ref, color=color, lw=1.8)

    axes[0].set_ylabel(r"$\langle v\rangle/\langle v\rangle_{\rm cBE}$")
    axes[1].set_ylabel(r"$\langle v^2\rangle^{1/2}/\langle v^2\rangle_{\rm cBE}^{1/2}$")
    for ax in axes:
        ax.set_xlabel(r"$x=m_\chi/T$")
        ax.axhline(1.0, color="0.35", lw=0.9, ls=":")
    add_lambda_colorbar(fig, axes.ravel().tolist(), cmap, norm)
    save_figure(fig, out_dir, "velocity_moments", formats)


def plot_fig7_temperature_ratio(runs, cbe, cosmo, out_dir, formats, x_min=None, x_min_factor=1.0 + 1.0 / 15.0):
    if cbe is None:
        raise FileNotFoundError("Fig. 7 needs a cBE solution.")

    m = float(runs[0]["metadata"]["m_chi"])
    cbe_data = cbe_curves(cbe, m, cosmo)
    lambdas = np.array([lambda_from_run(run) for run in runs], dtype=float)
    cmap = plt.get_cmap("jet")
    norm = color_norm(lambdas, log=True)

    fig, ax = plt.subplots(figsize=(6.7, 4.4))
    for run in runs:
        curves = fbe_curves(run, cosmo)
        mask = np.isfinite(curves["x"])
        x_cut = moments_x_min_for_run(run, x_min=x_min, x_min_factor=x_min_factor)
        if x_cut is not None:
            mask &= curves["x"] >= x_cut
        x_plot = curves["x"][mask]
        ref = interp_positive(x_plot, cbe_data["x"], cbe_data["Tchi"])
        ax.semilogx(x_plot, curves["Tchi"][mask] / ref, color=cmap(norm(lambda_from_run(run))), lw=1.8)

    add_lambda_colorbar(fig, ax, cmap, norm)
    ax.axhline(1.0, color="0.35", lw=0.9, ls=":")
    ax.set_xlabel(r"$x=m_\chi/T$")
    ax.set_ylabel(r"$T_\chi/T_{\chi,{\rm cBE}}$")
    save_figure(fig, out_dir, "temperature_moment", formats)


def sigma_22_contact_nr(lam, m_chi, prefactor=1.0 / (64.0 * np.pi)):
    return float(prefactor) * float(lam) ** 2 / max(float(m_chi) ** 2, 1e-300)


def relative_velocity_from_vrms(v_rms):
    return np.minimum((4.0 / np.sqrt(3.0 * np.pi)) * np.maximum(v_rms, 0.0), 2.0)


def condensate_density_from_metadata(metadata, cosmo, af):
    a0 = float(metadata.get("a0", 1.0))
    n_init = float(metadata["n_condensate_initial"])
    N_init = a0**3 * n_init
    gamma = float(metadata["gamma_condensate"])
    _, _, build_condensate = solve_condensate_N_loga_quad(
        H_of_a=cosmo.H_of_a,
        N_init=N_init,
        ai=a0,
        af=float(af),
        n_eval=4000,
        zero_rel=1e-14,
    )
    _, _, n_of_a = build_condensate(gamma)
    return n_of_a


def freeze_in_sources(a, metadata, cosmo, n_parent_of_a):
    a = np.asarray(a, dtype=float)
    T = np.array([float(cosmo.T_of_a(ai)) for ai in a], dtype=float)
    S_higgs = float(metadata["lambda_portal"]) ** 2 * np.array(
        [
            float(
                Gamma_htophiphi(
                    Ti,
                    float(metadata["m_chi"]),
                    mh=float(metadata.get("m_h", 125.0)),
                    v=float(metadata.get("v_h", 246.0)),
                )
            )
            for Ti in T
        ],
        dtype=float,
    )
    gamma_X = float(metadata["gamma_condensate"])
    multiplicity = float(metadata.get("multiplicity_condensate", 2.0))
    n_X = np.array([float(n_parent_of_a(ai)) for ai in a], dtype=float)
    S_cond = multiplicity * gamma_X * n_X
    N_X0 = float(metadata.get("a0", 1.0)) ** 3 * float(metadata["n_condensate_initial"])
    return S_higgs, S_cond, max(N_X0, 1e-300)


def plot_rates_over_H(runs, cosmo, out_dir, formats, y_floor=1e-5):
    lambdas = np.array([lambda_from_run(run) for run in runs], dtype=float)
    cmap = plt.get_cmap("jet")
    norm = color_norm(lambdas, log=True)

    fig, ax = plt.subplots(figsize=(6.9, 4.6))

    reference = runs[-1]
    reference_curves = fbe_curves(reference, cosmo)
    reference_metadata = reference["metadata"]
    n_parent = condensate_density_from_metadata(
        reference_metadata,
        cosmo=cosmo,
        af=float(np.nanmax(reference_curves["a"])),
    )
    S_higgs, S_cond, N_X0 = freeze_in_sources(reference_curves["a"], reference_metadata, cosmo, n_parent)
    H_ref = np.array([float(cosmo.H_of_a(ai)) for ai in reference_curves["a"]], dtype=float)

    ax.loglog(
        reference_curves["x"],
        floor_for_log(reference_curves["a"] ** 3 * S_higgs / np.maximum(H_ref * N_X0, 1e-300), floor=y_floor),
        color="black",
        lw=2.2,
        ls="-",
        label=r"Higgs decay FI",
    )
    ax.loglog(
        reference_curves["x"],
        floor_for_log(reference_curves["a"] ** 3 * S_cond / np.maximum(H_ref * N_X0, 1e-300), floor=y_floor),
        color="black",
        lw=2.2,
        ls="--",
        label=r"condensate FI",
    )

    for run in runs:
        curves = fbe_curves(run, cosmo)
        metadata = run["metadata"]
        H = np.array([float(cosmo.H_of_a(ai)) for ai in curves["a"]], dtype=float)
        gamma22 = curves["n"] * sigma_22_contact_nr(lambda_from_run(run), float(metadata["m_chi"])) * relative_velocity_from_vrms(curves["v_rms"])
        ax.loglog(curves["x"], floor_for_log(gamma22 / np.maximum(H, 1e-300), floor=y_floor), color=cmap(norm(lambda_from_run(run))), lw=1.8)

    ax.axhline(1.0, color="0.35", lw=0.9, ls=":")
    add_lambda_colorbar(fig, ax, cmap, norm)
    ax.legend(frameon=False, loc="best")
    ax.set_xlabel(r"$x=m_\chi/T$")
    ax.set_ylabel(r"$\mathrm{rate}/H$")
    y_max = 1e4
    for line in ax.lines:
        y = np.asarray(line.get_ydata(), dtype=float)
        if np.any(np.isfinite(y)):
            y_max = max(y_max, 10.0 ** np.ceil(np.log10(np.nanmax(y[np.isfinite(y)]))))
    ax.set_ylim(y_floor, y_max)
    save_figure(fig, out_dir, "rates", formats)


def plot_abundance_Y(runs, cbe, cosmo, out_dir, formats):
    fig, ax = plt.subplots(figsize=(6.7, 4.4))
    run = runs[-1]
    curves = fbe_curves(run, cosmo)
    y_floor = max(1e-30, 1e-8 * float(np.nanmax(curves["Y"])))
    ax.loglog(
        curves["x"],
        clip_for_log(curves["Y"], floor=y_floor),
        color="tab:blue",
        lw=1.9,
        label=rf"fBE, $\lambda={lambda_from_run(run):.1e}$",
    )

    cbe_data = cbe_curves(cbe, float(runs[0]["metadata"]["m_chi"]), cosmo)
    if cbe_data is not None:
        cbe_floor = max(1e-30, 1e-8 * float(np.nanmax(cbe_data["Y"])))
        ax.loglog(
            cbe_data["x"],
            clip_for_log(cbe_data["Y"], floor=min(y_floor, cbe_floor)),
            color="black",
            ls="--",
            lw=2.0,
            label=r"cBE",
        )

    ax.legend(frameon=False)
    ax.set_xlabel(r"$x=m_\chi/T$")
    ax.set_ylabel(r"$Y=n/s$")
    save_figure(fig, out_dir, "abundance_Y_comparison", formats)


def main():
    parser = argparse.ArgumentParser(description="Make the clean KineticXGPU figure set from saved fBE/cBE runs.")
    parser.add_argument("--runs-dir", default="results/runs/fBE")
    parser.add_argument("--cbe-dir", default="results/runs/cBE")
    parser.add_argument("--out-dir", default="results/plots")
    parser.add_argument("--run-prefix", default=None, help="Only load fBE run directories whose names start with this prefix.")
    parser.add_argument("--formats", nargs="+", default=["pdf"], choices=["pdf", "png", "svg"])
    parser.add_argument("--map-location", default="cpu")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--snapshots", type=int, default=36)
    parser.add_argument("--evolution-x-start", type=float, default=1.0)
    parser.add_argument("--moments-x-min", type=float, default=None)
    parser.add_argument("--moments-x-min-factor", type=float, default=1.0 + 1.0 / 15.0)
    parser.add_argument("--usetex", action="store_true", help="Use an external LaTeX installation for text rendering.")
    args = parser.parse_args()

    set_plot_style(usetex=args.usetex)
    runs = load_fbe_runs(
        args.runs_dir,
        run_prefix=args.run_prefix,
        map_location=args.map_location,
        max_runs=args.max_runs,
    )
    cbe = load_cbe(args.cbe_dir)
    cosmo = VariableGCosmology()
    out_dir = project_path(args.out_dir)

    plot_all_evolutions(
        runs,
        cosmo,
        out_dir,
        args.formats,
        n_snapshots=args.snapshots,
        x_start=args.evolution_x_start,
    )
    plot_fig5_final_distributions(runs, cbe, cosmo, out_dir, args.formats)
    if cbe is not None:
        plot_fig6_velocity_ratios(
            runs,
            cbe,
            cosmo,
            out_dir,
            args.formats,
            x_min=args.moments_x_min,
            x_min_factor=args.moments_x_min_factor,
        )
        plot_fig7_temperature_ratio(
            runs,
            cbe,
            cosmo,
            out_dir,
            args.formats,
            x_min=args.moments_x_min,
            x_min_factor=args.moments_x_min_factor,
        )
    plot_rates_over_H(runs, cosmo, out_dir, args.formats)
    plot_abundance_Y(runs, cbe, cosmo, out_dir, args.formats)

    print(f"Loaded {len(runs)} fBE run(s) from {project_path(args.runs_dir)}")
    if cbe is None:
        print("No cBE solution was found; cBE-dependent figures were skipped.")
    else:
        print(f"Loaded cBE solution from {project_path(args.cbe_dir)}")
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
