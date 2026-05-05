#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm, Normalize
from scipy.interpolate import PchipInterpolator
from scipy.special import kve


THIS_FILE = Path(__file__).resolve()
SCRIPTS_DIR = THIS_FILE.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cosmology import VariableGCosmology


DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "runs"


def resolve_project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_hybrid_run(run_dir, map_location="cpu"):
    run_dir = resolve_project_path(run_dir)
    traj_path = run_dir / "trajectory.pt"
    meta_path = run_dir / "metadata.json"

    if not traj_path.exists():
        raise FileNotFoundError(f"Missing {traj_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")

    try:
        traj = torch.load(traj_path, map_location=map_location, weights_only=False)
    except TypeError:
        traj = torch.load(traj_path, map_location=map_location)

    with meta_path.open("r") as f:
        metadata = json.load(f)

    return traj, metadata


def reconstruct_log_q_from_metadata(metadata):
    N = int(metadata["N_grid"])
    qmin = float(metadata["qmin"])
    qmax = float(metadata["qmax"])
    return np.logspace(np.log10(qmin), np.log10(qmax), N)


def load_all_runs_matching(pattern="hybrid_saved_*", results_dir=DEFAULT_RESULTS_DIR, map_location="cpu"):
    results_dir = resolve_project_path(results_dir)
    loaded = {}

    for run_dir in sorted(results_dir.glob(pattern)):
        if not run_dir.is_dir():
            continue

        traj_path = run_dir / "trajectory.pt"
        meta_path = run_dir / "metadata.json"

        if not traj_path.exists() or not meta_path.exists():
            continue

        traj, metadata = load_hybrid_run(run_dir, map_location=map_location)
        q = reconstruct_log_q_from_metadata(metadata)

        loaded[run_dir.name] = {
            "run_dir": run_dir,
            "traj": traj,
            "metadata": metadata,
            "q": q,
        }

    return loaded


def load_cbe_solution(run_dir=DEFAULT_RESULTS_DIR / "cbe_benchmark"):
    run_dir = resolve_project_path(run_dir)
    data_path = run_dir / "cbe_solution.npz"
    meta_path = run_dir / "metadata.json"

    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}")

    data = np.load(data_path)
    metadata = None

    if meta_path.exists():
        with meta_path.open("r") as f:
            metadata = json.load(f)

    return data, metadata


def make_cbe_interpolators(cbe_data, extrapolate_tail=False):
    a = np.asarray(cbe_data["a"], dtype=float)
    N = np.asarray(cbe_data["N"], dtype=float)
    T = np.asarray(cbe_data["T"], dtype=float)

    mask = (a > 0.0) & np.isfinite(a) & np.isfinite(N) & np.isfinite(T)
    a = a[mask]
    N = np.maximum(N[mask], 1e-300)
    T = np.maximum(T[mask], 1e-300)

    order = np.argsort(a)
    loga = np.log(a[order])
    logN = np.log(N[order])
    logT = np.log(T[order])

    a_max = float(a[order][-1])
    N_ref = float(N[order][-1])
    T_ref = float(T[order][-1])

    N_interp = PchipInterpolator(loga, logN, extrapolate=False)
    T_interp = PchipInterpolator(loga, logT, extrapolate=False)

    def Ns(a_eval):
        a_in = np.asarray(a_eval, dtype=float)
        out = np.exp(N_interp(np.log(a_in)))
        if extrapolate_tail:
            out = np.where(a_in > a_max, N_ref, out)
        return out

    def Ts(a_eval):
        a_in = np.asarray(a_eval, dtype=float)
        out = np.exp(T_interp(np.log(a_in)))
        if extrapolate_tail:
            out = np.where(a_in > a_max, T_ref * (a_max / a_in)**2, out)
        return out

    Ns.a_grid = a[order]
    Ns.N_grid = N[order]
    Ns.a_max = a_max
    Ns.extrapolate_tail = extrapolate_tail
    Ts.a_grid = a[order]
    Ts.T_grid = T[order]
    Ts.a_max = a_max
    Ts.extrapolate_tail = extrapolate_tail

    return Ns, Ts


def cbe_arrays_for_plot(cbe, a_fallback):
    if cbe is None:
        return None

    try:
        cbe_data = cbe[0]
        if "a" in cbe_data and "N" in cbe_data and "T" in cbe_data:
            return (
                np.asarray(cbe_data["a"], dtype=float),
                np.asarray(cbe_data["N"], dtype=float),
                np.asarray(cbe_data["T"], dtype=float),
            )
    except (TypeError, KeyError, IndexError):
        pass

    Ns, Ts = cbe
    if hasattr(Ns, "a_grid") and hasattr(Ns, "N_grid") and hasattr(Ts, "T_grid"):
        return np.asarray(Ns.a_grid, dtype=float), np.asarray(Ns.N_grid, dtype=float), np.asarray(Ts.T_grid, dtype=float)

    a_cbe = np.asarray(a_fallback, dtype=float)
    return a_cbe, np.asarray(Ns(a_cbe), dtype=float), np.asarray(Ts(a_cbe), dtype=float)


def positive_finite_or_nan(values, min_value=None):
    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr) & (arr > 0.0)
    if min_value is not None:
        mask &= arr >= float(min_value)
    return np.where(mask, arr, np.nan)


def tensor_to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def closest_snapshot_index(traj, a_target):
    a_arr = tensor_to_numpy(traj["a"])
    idx = int(np.argmin(np.abs(a_arr - float(a_target))))
    return idx, float(a_arr[idx])


def snapshot_indices_by_a(traj, a_values):
    return [closest_snapshot_index(traj, a)[0] for a in a_values]


def snapshot_indices_even(traj, n=8):
    a_arr = tensor_to_numpy(traj["a"])
    if len(a_arr) <= n:
        return list(range(len(a_arr)))
    return np.unique(np.linspace(0, len(a_arr) - 1, n, dtype=int)).tolist()


def get_snapshot_qspace(traj, q, idx):
    a_arr = tensor_to_numpy(traj["a"])
    f_arr = tensor_to_numpy(traj["f"])

    a = float(a_arr[idx])
    f = np.asarray(f_arr[idx], dtype=float)
    q = np.asarray(q, dtype=float)
    p = q / a

    return {
        "idx": int(idx),
        "a": a,
        "q": q,
        "p": p,
        "f": f,
        "p2f": p**2 * f,
        "q_p2f": q * p**2 * f,
        "q3f_over_a2": q**3 * f / a**2,
    }


def compute_moments_from_q_snapshot(q, f, a, m, g=1.0):
    q = np.asarray(q, dtype=float)
    f = np.asarray(f, dtype=float)
    a = float(a)
    m = float(m)
    g = float(g)

    p = q / a
    E = np.sqrt(p**2 + m**2)
    v = p / E

    pref = g / (2.0 * np.pi**2)
    n = pref * np.trapezoid(q**2 * f, q) / a**3
    rho = pref * np.trapezoid(q**2 * E * f, q) / a**3
    P = pref * np.trapezoid(q**2 * (p**2 / (3.0 * E)) * f, q) / a**3

    avg_v2 = pref * np.trapezoid(q**2 * v**2 * f, q) / a**3 / max(n, 1e-300)
    avg_p = pref * np.trapezoid(q**2 * p * f, q) / a**3 / max(n, 1e-300)
    avg_v = pref * np.trapezoid(q**2 * v * f, q) / a**3 / max(n, 1e-300)

    return {
        "a": a,
        "n": n,
        "rho": rho,
        "P": P,
        "w": P / max(rho, 1e-300),
        "Tkin": P / max(n, 1e-300),
        "avg_E": rho / max(n, 1e-300),
        "avg_p": avg_p,
        "avg_v": avg_v,
        "avg_v2": avg_v2,
        "v_rms": np.sqrt(max(avg_v2, 0.0)),
    }


def compute_moments_along_trajectory(traj, q, m, g=1.0):
    a_arr = tensor_to_numpy(traj["a"])
    f_arr = tensor_to_numpy(traj["f"])
    modes = traj.get("mode_hist", [""] * len(a_arr))

    rows = []
    for idx, (a, f) in enumerate(zip(a_arr, f_arr)):
        row = compute_moments_from_q_snapshot(q=q, f=f, a=a, m=m, g=g)
        row["idx"] = idx
        row["mode"] = modes[idx] if idx < len(modes) else ""
        rows.append(row)

    return rows


def n_eq_MB_stable(T, m):
    T = float(T)
    m = float(m)

    if T <= 0:
        return 0.0

    x = m / T
    return (m**2 * T / (2.0 * np.pi**2)) * np.exp(-x) * kve(2, x)


def p2f_MB_from_n_T(p, m, n, T):
    p = np.asarray(p, dtype=float)
    m = float(m)
    n = float(n)
    T = float(T)
    E = np.sqrt(p**2 + m**2)

    if T <= 0 or n <= 0:
        f = np.zeros_like(p)
        return f, p**2 * f

    x = m / T
    amp = n * (2.0 * np.pi**2) / max(m**2 * T * kve(2, x), 1e-300)
    f = amp * np.exp(-(E - m) / T)
    return f, p**2 * f


def infer_MB_from_snapshot(q, f, a, m, g=1.0):
    moments = compute_moments_from_q_snapshot(q=q, f=f, a=a, m=m, g=g)
    T = moments["Tkin"]
    n = moments["n"]

    neq = n_eq_MB_stable(T, m)
    logz = np.log(max(n, 1e-300)) - np.log(max(neq, 1e-300))
    z = np.exp(logz) if logz < 700 else np.inf

    p = np.asarray(q, dtype=float) / float(a)
    f_MB, p2f_MB = p2f_MB_from_n_T(p=p, m=m, n=n, T=T)

    return {
        "T": T,
        "n": n,
        "neq": neq,
        "z": z,
        "logz": logz,
        "moments": moments,
        "p": p,
        "f_MB": f_MB,
        "p2f_MB": p2f_MB,
    }


def cbe_distribution_on_p_grid(p, a, m, Ns, Ts):
    a = float(a)
    m = float(m)
    p = np.asarray(p, dtype=float)

    T_val = float(Ts(a))
    N_val = float(Ns(a))
    n_val = N_val / a**3

    f_cbe, p2f_cbe = p2f_MB_from_n_T(p=p, m=m, n=n_val, T=T_val)
    neq_val = n_eq_MB_stable(T_val, m)
    logz_val = np.log(max(n_val, 1e-300)) - np.log(max(neq_val, 1e-300))

    return {
        "T": T_val,
        "N": N_val,
        "n": n_val,
        "neq": neq_val,
        "z": np.exp(logz_val) if logz_val < 700 else np.inf,
        "logz": logz_val,
        "f": f_cbe,
        "p2f": p2f_cbe,
    }


def make_cosmology():
    return VariableGCosmology()


def entropy_density_from_cosmo(cosmo, a):
    T_sm = float(cosmo.T_of_a(float(a)))
    return float(cosmo.entropy_density(T_sm))


def x_from_a(a, m_chi, cosmo=None):
    if cosmo is None:
        cosmo = make_cosmology()

    a_arr = np.asarray(a, dtype=float)
    T_sm = np.array([float(cosmo.T_of_a(ai)) for ai in np.atleast_1d(a_arr)])
    x = float(m_chi) / np.maximum(T_sm, 1e-300)
    return x.item() if np.ndim(a_arr) == 0 else x.reshape(a_arr.shape)


def sort_xy_by_x(x, *ys):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    sorted_ys = [np.asarray(y)[order] for y in ys]
    return (x[order], *sorted_ys)


def cbe_velocity_moments_from_T(T, m, n_p=4096):
    T_arr = np.asarray(T, dtype=float)
    m = max(float(m), 1e-300)
    avg_v = np.full_like(T_arr, np.nan, dtype=float)
    v_rms = np.full_like(T_arr, np.nan, dtype=float)

    for idx, T_val in np.ndenumerate(T_arr):
        if not np.isfinite(T_val) or T_val <= 0.0:
            continue

        p_thermal_nr = np.sqrt(max(m * T_val, 0.0))
        p_max = max(80.0 * T_val, 30.0 * p_thermal_nr, 10.0 * m)
        p = np.linspace(0.0, p_max, int(n_p))
        E = np.sqrt(p**2 + m**2)
        weight = p**2 * np.exp(-(E - m) / T_val)
        denom = np.trapezoid(weight, p)

        if denom <= 0.0 or not np.isfinite(denom):
            continue

        v = p / E
        avg_v[idx] = np.trapezoid(weight * v, p) / denom
        v_rms[idx] = np.sqrt(max(np.trapezoid(weight * v**2, p) / denom, 0.0))

    if T_arr.ndim == 0:
        return {"avg_v": float(avg_v), "v_rms": float(v_rms)}
    return {"avg_v": avg_v, "v_rms": v_rms}


def mb_mean_energy_from_T(T, m):
    T = np.asarray(T, dtype=float)
    m = float(m)
    z = m / np.maximum(T, 1e-300)
    out = np.empty_like(T, dtype=float)

    nr = z > 100.0
    out[nr] = m + 1.5 * T[nr] + 1.875 * T[nr] ** 2 / max(m, 1e-300)

    rel = ~nr
    if np.any(rel):
        k1 = kve(1, z[rel])
        k2 = kve(2, z[rel])
        out[rel] = m * k1 / np.maximum(k2, 1e-300) + 3.0 * T[rel]

    return out.item() if np.ndim(T) == 0 else out


def cbe_w_from_T(T, m):
    T = np.asarray(T, dtype=float)
    ebar = mb_mean_energy_from_T(T, m)
    return T / np.maximum(ebar, 1e-300)


def compute_abundance_along_trajectory(run, cosmo=None, Y_obs=None):
    if cosmo is None:
        cosmo = make_cosmology()

    metadata = run["metadata"]
    moments = compute_moments_along_trajectory(
        traj=run["traj"],
        q=run["q"],
        m=float(metadata["m_chi"]),
        g=1.0,
    )

    a = np.array([row["a"] for row in moments])
    n = np.array([row["n"] for row in moments])
    s = np.array([entropy_density_from_cosmo(cosmo, ai) for ai in a])
    Y = n / np.maximum(s, 1e-300)

    out = {"a": a, "Y": Y, "moments": moments}
    if Y_obs is not None:
        out["omega_ratio"] = Y / max(float(Y_obs), 1e-300)
    return out


def compute_cbe_abundance(cbe_solution, cosmo=None, Y_obs=None):
    if cosmo is None:
        cosmo = make_cosmology()

    cbe_data, cbe_metadata = cbe_solution
    a = np.asarray(cbe_data["a"], dtype=float)
    N = np.asarray(cbe_data["N"], dtype=float)
    s = np.array([entropy_density_from_cosmo(cosmo, ai) for ai in a])
    Y = (N / np.maximum(a**3, 1e-300)) / np.maximum(s, 1e-300)

    if Y_obs is None and cbe_metadata is not None:
        Y_obs = cbe_metadata.get("Y_obs")

    out = {"a": a, "Y": Y}
    if Y_obs is not None:
        out["omega_ratio"] = Y / max(float(Y_obs), 1e-300)
    return out


def color_values_for_snapshots(snapshots, metadata, color_by="x", cosmo=None):
    if cosmo is None:
        cosmo = make_cosmology()

    m = float(metadata["m_chi"])
    values = []

    for snap in snapshots:
        a = snap["a"]
        T_sm = float(cosmo.T_of_a(a))

        if color_by == "x":
            values.append(m / max(T_sm, 1e-300))
        elif color_by == "T":
            values.append(T_sm)
        elif color_by == "a":
            values.append(a)
        else:
            raise ValueError("color_by must be one of: 'x', 'T', 'a'.")

    return np.asarray(values)


def quantity_label(quantity):
    labels = {
        "f": r"$f(q)$",
        "p2f": r"$p^2 f(q)$",
        "q_p2f": r"$q\,p^2 f(q)=q^3 f/a^2$",
        "q3f_over_a2": r"$q^3 f(q)/a^2$",
    }
    return labels.get(quantity, quantity)


def plot_snapshots_evolution(
    run,
    indices=None,
    a_values=None,
    n_snapshots=8,
    max_curves=None,
    quantity="q_p2f",
    color_by="x",
    x_axis="q",
    yscale="linear",
    xscale="log",
    cmap="inferno",
    cosmo=None,
    ax=None,
):
    traj = run["traj"]
    q = run["q"]
    metadata = run["metadata"]

    if indices is None:
        if a_values is not None:
            indices = snapshot_indices_by_a(traj, a_values)
        else:
            if max_curves is not None:
                n_snapshots = int(max_curves)
            indices = snapshot_indices_even(traj, n=n_snapshots)

    snapshots = [get_snapshot_qspace(traj, q, idx) for idx in indices]
    color_values = color_values_for_snapshots(snapshots, metadata, color_by=color_by, cosmo=cosmo)

    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 5.0))

    norm = LogNorm(vmin=max(np.nanmin(color_values), 1e-300), vmax=np.nanmax(color_values)) if np.all(color_values > 0) else Normalize()
    cmap_obj = plt.get_cmap(cmap)

    for snap, color_value in zip(snapshots, color_values):
        x = snap[x_axis]
        y = snap[quantity]
        ax.plot(x, y, color=cmap_obj(norm(color_value)), linewidth=1.7)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    label = {"x": r"$x=m_\chi/T_{\rm SM}$", "T": r"$T_{\rm SM}$", "a": r"$a$"}[color_by]
    plt.colorbar(sm, ax=ax, label=label)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(r"$q$" if x_axis == "q" else r"$p=q/a$")
    ax.set_ylabel(quantity_label(quantity))
    ax.set_title(metadata.get("run_name", "fBE trajectory"))
    ax.grid(True, which="both", alpha=0.3)

    return ax


def tail_diagnostics(run, idx=-1, rel_floor=1e-12):
    snap = get_snapshot_qspace(run["traj"], run["q"], idx)
    y = np.asarray(snap["q_p2f"], dtype=float)
    ymax = np.nanmax(np.abs(y))
    active = np.isfinite(y) & (np.abs(y) > rel_floor * max(ymax, 1e-300))

    if not np.any(active):
        active_qmax = np.nan
        active_fraction_of_grid = 0.0
    else:
        active_qmax = float(np.max(snap["q"][active]))
        active_fraction_of_grid = active_qmax / float(np.max(snap["q"]))

    return {
        "idx": snap["idx"],
        "a": snap["a"],
        "q_grid_max": float(np.max(snap["q"])),
        "active_qmax": active_qmax,
        "active_qmax_over_grid_qmax": active_fraction_of_grid,
        "negative_points": int(np.sum(snap["f"] < 0.0)),
        "finite": bool(np.all(np.isfinite(snap["f"]))),
        "y_abs_max": float(ymax),
    }


def get_largest_lambda_run(loaded_runs):
    return max(
        loaded_runs.items(),
        key=lambda item: float(item[1]["metadata"].get("lambda_self", item[1]["metadata"].get("lam_self", 0.0))),
    )


def sorted_runs_by_lambda(loaded_runs):
    return sorted(
        loaded_runs.items(),
        key=lambda item: float(item[1]["metadata"].get("lambda_self", item[1]["metadata"].get("lam_self", 0.0))),
    )


def plot_final_runs_with_inferred_MB(
    loaded_runs,
    cbe=None,
    quantity="q_p2f",
    yscale="linear",
    xscale="log",
    cmap="inferno",
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 5.0))

    sorted_items = sorted_runs_by_lambda(loaded_runs)
    colors = plt.get_cmap(cmap)(np.linspace(0.18, 0.88, len(sorted_items)))

    for color, (name, run) in zip(colors, sorted_items):
        snap = get_snapshot_qspace(run["traj"], run["q"], idx=-1)
        lam = float(run["metadata"].get("lambda_self", run["metadata"].get("lam_self", np.nan)))
        ax.plot(snap["q"], snap[quantity], color=color, linewidth=1.8, label=fr"$\lambda_\mathrm{{self}}={lam:.1e}$")

    _, ref = get_largest_lambda_run(loaded_runs)
    snap_ref = get_snapshot_qspace(ref["traj"], ref["q"], idx=-1)
    mb = infer_MB_from_snapshot(
        q=ref["q"],
        f=snap_ref["f"],
        a=snap_ref["a"],
        m=float(ref["metadata"]["m_chi"]),
        g=1.0,
    )

    if quantity == "f":
        y_mb = mb["f_MB"]
    elif quantity == "p2f":
        y_mb = mb["p2f_MB"]
    elif quantity == "q_p2f":
        y_mb = snap_ref["q"] * mb["p2f_MB"]
    elif quantity == "q3f_over_a2":
        y_mb = snap_ref["q"] * mb["p2f_MB"]
    else:
        raise ValueError(f"Unknown quantity: {quantity}")

    ax.plot(snap_ref["q"], y_mb, "--", color="black", linewidth=2.2, label="inferred MB")

    cbe_out = None
    if cbe is not None:
        Ns, Ts = cbe
        cbe_out = cbe_distribution_on_p_grid(
            p=snap_ref["p"],
            a=snap_ref["a"],
            m=float(ref["metadata"]["m_chi"]),
            Ns=Ns,
            Ts=Ts,
        )

        if quantity == "f":
            y_cbe = cbe_out["f"]
        elif quantity == "p2f":
            y_cbe = cbe_out["p2f"]
        elif quantity in {"q_p2f", "q3f_over_a2"}:
            y_cbe = snap_ref["q"] * cbe_out["p2f"]
        else:
            raise ValueError(f"Unknown quantity: {quantity}")

        ax.plot(snap_ref["q"], y_cbe, ":", color="black", linewidth=2.2, label="cBE MB")

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(quantity_label(quantity))
    ax.set_title("Final fBE spectra")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)

    return ax, {"inferred_MB": mb, "cBE": cbe_out}


def plot_final_against_cbe(run, cbe, quantity="q_p2f", idx=-1, yscale="linear"):
    Ns, Ts = cbe
    metadata = run["metadata"]
    snap = get_snapshot_qspace(run["traj"], run["q"], idx=idx)
    mb = infer_MB_from_snapshot(
        q=run["q"],
        f=snap["f"],
        a=snap["a"],
        m=float(metadata["m_chi"]),
        g=1.0,
    )
    cbe_out = cbe_distribution_on_p_grid(
        p=snap["p"],
        a=snap["a"],
        m=float(metadata["m_chi"]),
        Ns=Ns,
        Ts=Ts,
    )

    if quantity == "f":
        y = snap["f"]
        y_mb = mb["f_MB"]
        y_cbe = cbe_out["f"]
        x = snap["p"]
        xlabel = r"$p$"
    elif quantity == "p2f":
        y = snap["p2f"]
        y_mb = mb["p2f_MB"]
        y_cbe = cbe_out["p2f"]
        x = snap["p"]
        xlabel = r"$p$"
    elif quantity in {"q_p2f", "q3f_over_a2"}:
        y = snap[quantity]
        y_mb = snap["q"] * mb["p2f_MB"]
        y_cbe = snap["q"] * cbe_out["p2f"]
        x = snap["q"]
        xlabel = r"$q$"
    else:
        raise ValueError(f"Unknown quantity: {quantity}")

    fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True)
    ax.plot(x, y, label="fBE")
    ax.plot(x, y_mb, "--", label="inferred MB")
    ax.plot(x, y_cbe, ":", linewidth=2.2, label="cBE MB")
    ax.set_xscale("log")
    ax.set_yscale(yscale)
    ax.set_ylabel(quantity_label(quantity))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)

    ax_ratio.plot(x, y / np.maximum(y_cbe, 1e-300), label="fBE / cBE")
    ax_ratio.plot(x, y_mb / np.maximum(y_cbe, 1e-300), "--", label="inferred MB / cBE")
    ax_ratio.axhline(1.0, color="black", linewidth=1.0, alpha=0.6)
    ax_ratio.set_xscale("log")
    ax_ratio.set_xlabel(xlabel)
    ax_ratio.set_ylabel("ratio")
    ax_ratio.grid(True, which="both", alpha=0.3)
    ax_ratio.legend(frameon=False)

    print("fBE inferred MB:")
    print(f"  a      = {snap['a']:.6e}")
    print(f"  Tkin   = {mb['T']:.6e}")
    print(f"  n      = {mb['n']:.6e}")
    print(f"  log z  = {mb['logz']:.6e}")
    print("cBE:")
    print(f"  T      = {cbe_out['T']:.6e}")
    print(f"  N      = {cbe_out['N']:.6e}")
    print(f"  n      = {cbe_out['n']:.6e}")
    print(f"  log z  = {cbe_out['logz']:.6e}")

    return fig, {"snapshot": snap, "inferred_MB": mb, "cBE": cbe_out}


def plot_moment_evolution(run, cosmo=None, ax=None, min_density=1e-250, min_temperature=1e-30):
    metadata = run["metadata"]
    moments = compute_moments_along_trajectory(
        traj=run["traj"],
        q=run["q"],
        m=float(metadata["m_chi"]),
        g=1.0,
    )

    a = np.array([row["a"] for row in moments])
    n = np.array([row["n"] for row in moments])
    rho = np.array([row["rho"] for row in moments])
    Tkin = np.array([row["Tkin"] for row in moments])
    vrms = np.array([row["v_rms"] for row in moments])

    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 5.0))

    ax.loglog(a, positive_finite_or_nan(n, min_density), label=r"$n$")
    ax.loglog(a, positive_finite_or_nan(rho, min_density), label=r"$\rho$")
    ax.loglog(a, positive_finite_or_nan(Tkin, min_temperature), label=r"$T_\mathrm{kin}=P/n$")

    if cosmo is not None:
        T_sm = np.array([float(cosmo.T_of_a(ai)) for ai in a])
        ax.loglog(a, positive_finite_or_nan(T_sm, min_temperature), "--", label=r"$T_\mathrm{SM}$")

    ax.set_xlabel(r"$a$")
    ax.set_ylabel("moments / temperature")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)

    return ax, moments


def plot_moment_evolution_x(run, cosmo=None, ax=None, min_density=1e-250, min_temperature=1e-30):
    if cosmo is None:
        cosmo = make_cosmology()

    metadata = run["metadata"]
    moments = compute_moments_along_trajectory(
        traj=run["traj"],
        q=run["q"],
        m=float(metadata["m_chi"]),
        g=1.0,
    )

    a = np.array([row["a"] for row in moments])
    x = x_from_a(a, float(metadata["m_chi"]), cosmo)
    n = np.array([row["n"] for row in moments])
    rho = np.array([row["rho"] for row in moments])
    Tkin = np.array([row["Tkin"] for row in moments])
    T_sm = np.array([float(cosmo.T_of_a(ai)) for ai in a])
    x, n, rho, Tkin, T_sm = sort_xy_by_x(x, n, rho, Tkin, T_sm)

    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 5.0))

    ax.loglog(x, positive_finite_or_nan(n, min_density), label=r"$n$")
    ax.loglog(x, positive_finite_or_nan(rho, min_density), label=r"$\rho$")
    ax.loglog(x, positive_finite_or_nan(Tkin, min_temperature), label=r"$T_\mathrm{kin}=P/n$")
    ax.loglog(x, positive_finite_or_nan(T_sm, min_temperature), "--", label=r"$T_\mathrm{SM}$")

    ax.set_xlabel(r"$x=m_\chi/T_\mathrm{SM}$")
    ax.set_ylabel("moments / temperature")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)

    return ax, moments


def plot_temperature_fugacity_velocity(
    run,
    cbe=None,
    cosmo=None,
    min_temperature=1e-30,
    min_v=1e-12,
    velocity_quantity="avg_v",
):
    if velocity_quantity not in {"avg_v", "v_rms"}:
        raise ValueError("velocity_quantity must be either 'avg_v' or 'v_rms'.")

    metadata = run["metadata"]
    moments = compute_moments_along_trajectory(
        traj=run["traj"],
        q=run["q"],
        m=float(metadata["m_chi"]),
        g=1.0,
    )

    a = np.array([row["a"] for row in moments])
    Tkin = np.array([row["Tkin"] for row in moments])
    n = np.array([row["n"] for row in moments])
    velocity = np.array([row[velocity_quantity] for row in moments])
    velocity_label = r"$\langle v\rangle$" if velocity_quantity == "avg_v" else r"$v_\mathrm{rms}$"

    neq = np.array([n_eq_MB_stable(T, float(metadata["m_chi"])) for T in Tkin])
    logz = np.log(np.maximum(n, 1e-300)) - np.log(np.maximum(neq, 1e-300))

    fig, axs = plt.subplots(3, 1, figsize=(7.2, 9.0), sharex=True)

    axs[0].loglog(a, positive_finite_or_nan(Tkin, min_temperature), label=r"fBE $T_\mathrm{kin}$")
    axs[1].semilogx(a, logz, label=r"fBE $\log z$")
    axs[2].loglog(a, positive_finite_or_nan(velocity, min_v), label=fr"fBE {velocity_label}")

    if cbe is not None:
        a_cbe, N_cbe, T_cbe = cbe_arrays_for_plot(cbe, a)
        n_cbe = N_cbe / a_cbe**3
        neq_cbe = np.array([n_eq_MB_stable(T, float(metadata["m_chi"])) for T in T_cbe])
        logz_cbe = np.log(np.maximum(n_cbe, 1e-300)) - np.log(np.maximum(neq_cbe, 1e-300))
        velocity_cbe = cbe_velocity_moments_from_T(T_cbe, float(metadata["m_chi"]))[velocity_quantity]

        axs[0].loglog(a_cbe, positive_finite_or_nan(T_cbe, min_temperature), "--", label=r"cBE $T_s$")
        axs[1].semilogx(a_cbe, logz_cbe, "--", label=r"cBE $\log z$")
        axs[2].loglog(a_cbe, positive_finite_or_nan(velocity_cbe, min_v), "--", label=fr"cBE {velocity_label}")

    if cosmo is not None:
        T_sm = np.array([float(cosmo.T_of_a(ai)) for ai in a])
        axs[0].loglog(a, positive_finite_or_nan(T_sm, min_temperature), ":", label=r"$T_\mathrm{SM}$")

    axs[0].set_ylabel("temperature")
    axs[1].set_ylabel(r"$\log z$")
    axs[2].set_ylabel(velocity_label)
    axs[2].set_xlabel(r"$a$")

    for ax in axs:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False)

    return fig, {"moments": moments, "logz": logz, "velocity": velocity, "velocity_quantity": velocity_quantity}


def plot_temperature_fugacity_velocity_x(
    run,
    cbe=None,
    cosmo=None,
    min_temperature=1e-30,
    min_v=1e-12,
    velocity_quantity="avg_v",
):
    if cosmo is None:
        cosmo = make_cosmology()
    if velocity_quantity not in {"avg_v", "v_rms"}:
        raise ValueError("velocity_quantity must be either 'avg_v' or 'v_rms'.")

    metadata = run["metadata"]
    m_chi = float(metadata["m_chi"])
    moments = compute_moments_along_trajectory(
        traj=run["traj"],
        q=run["q"],
        m=m_chi,
        g=1.0,
    )

    a = np.array([row["a"] for row in moments])
    x = x_from_a(a, m_chi, cosmo)
    Tkin = np.array([row["Tkin"] for row in moments])
    n = np.array([row["n"] for row in moments])
    velocity = np.array([row[velocity_quantity] for row in moments])
    T_sm = np.array([float(cosmo.T_of_a(ai)) for ai in a])
    neq = np.array([n_eq_MB_stable(T, m_chi) for T in Tkin])
    logz = np.log(np.maximum(n, 1e-300)) - np.log(np.maximum(neq, 1e-300))
    x, Tkin, logz, velocity, T_sm = sort_xy_by_x(x, Tkin, logz, velocity, T_sm)
    velocity_label = r"$\langle v\rangle$" if velocity_quantity == "avg_v" else r"$v_\mathrm{rms}$"

    fig, axs = plt.subplots(3, 1, figsize=(7.2, 9.0), sharex=True)

    axs[0].loglog(x, positive_finite_or_nan(Tkin, min_temperature), label=r"fBE $T_\mathrm{kin}$")
    axs[1].semilogx(x, logz, label=r"fBE $\log z$")
    axs[2].loglog(x, positive_finite_or_nan(velocity, min_v), label=fr"fBE {velocity_label}")

    if cbe is not None:
        a_cbe, N_cbe, T_cbe = cbe_arrays_for_plot(cbe, a)
        x_cbe = x_from_a(a_cbe, m_chi, cosmo)
        n_cbe = N_cbe / a_cbe**3
        neq_cbe = np.array([n_eq_MB_stable(T, m_chi) for T in T_cbe])
        logz_cbe = np.log(np.maximum(n_cbe, 1e-300)) - np.log(np.maximum(neq_cbe, 1e-300))
        velocity_cbe = cbe_velocity_moments_from_T(T_cbe, m_chi)[velocity_quantity]
        x_cbe, T_cbe, logz_cbe, velocity_cbe = sort_xy_by_x(x_cbe, T_cbe, logz_cbe, velocity_cbe)

        axs[0].loglog(x_cbe, positive_finite_or_nan(T_cbe, min_temperature), "--", label=r"cBE $T_s$")
        axs[1].semilogx(x_cbe, logz_cbe, "--", label=r"cBE $\log z$")
        axs[2].loglog(x_cbe, positive_finite_or_nan(velocity_cbe, min_v), "--", label=fr"cBE {velocity_label}")

    axs[0].loglog(x, positive_finite_or_nan(T_sm, min_temperature), ":", label=r"$T_\mathrm{SM}$")

    axs[0].set_ylabel("temperature")
    axs[1].set_ylabel(r"$\log z$")
    axs[2].set_ylabel(velocity_label)
    axs[2].set_xlabel(r"$x=m_\chi/T_\mathrm{SM}$")

    for ax in axs:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=False)

    return fig, {"moments": moments, "logz": logz, "velocity": velocity, "velocity_quantity": velocity_quantity}


def plot_omega_evolution(run, cbe_solution=None, cosmo=None, Y_obs=None, ax=None, ymin=1e-1):
    if cosmo is None:
        cosmo = make_cosmology()

    if Y_obs is None and cbe_solution is not None and cbe_solution[1] is not None:
        Y_obs = cbe_solution[1].get("Y_obs")

    fbe = compute_abundance_along_trajectory(run, cosmo=cosmo, Y_obs=Y_obs)
    cbe = compute_cbe_abundance(cbe_solution, cosmo=cosmo, Y_obs=Y_obs) if cbe_solution is not None else None

    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 5.0))

    y_key = "omega_ratio" if Y_obs is not None else "Y"
    ylabel = r"$\Omega/\Omega_\mathrm{obs}$" if Y_obs is not None else r"$Y=n/s$"

    ax.loglog(fbe["a"], positive_finite_or_nan(fbe[y_key]), label="fBE")
    if cbe is not None:
        ax.loglog(cbe["a"], positive_finite_or_nan(cbe[y_key]), "--", label="cBE")
    if Y_obs is not None:
        ax.axhline(1.0, color="black", linewidth=1.0, alpha=0.55)
    if ymin is not None:
        ax.set_ylim(bottom=float(ymin))

    ax.set_xlabel(r"$a$")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)

    return ax, {"fBE": fbe, "cBE": cbe, "Y_obs": Y_obs}


def plot_omega_evolution_x(run, cbe_solution=None, cosmo=None, Y_obs=None, ax=None, ymin=1e-1):
    if cosmo is None:
        cosmo = make_cosmology()

    metadata = run["metadata"]
    m_chi = float(metadata["m_chi"])

    if Y_obs is None and cbe_solution is not None and cbe_solution[1] is not None:
        Y_obs = cbe_solution[1].get("Y_obs")

    fbe = compute_abundance_along_trajectory(run, cosmo=cosmo, Y_obs=Y_obs)
    cbe = compute_cbe_abundance(cbe_solution, cosmo=cosmo, Y_obs=Y_obs) if cbe_solution is not None else None

    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 5.0))

    y_key = "omega_ratio" if Y_obs is not None else "Y"
    ylabel = r"$\Omega/\Omega_\mathrm{obs}$" if Y_obs is not None else r"$Y=n/s$"

    x_fbe = x_from_a(fbe["a"], m_chi, cosmo)
    x_fbe, y_fbe = sort_xy_by_x(x_fbe, fbe[y_key])
    ax.loglog(x_fbe, positive_finite_or_nan(y_fbe), label="fBE")

    if cbe is not None:
        x_cbe = x_from_a(cbe["a"], m_chi, cosmo)
        x_cbe, y_cbe = sort_xy_by_x(x_cbe, cbe[y_key])
        ax.loglog(x_cbe, positive_finite_or_nan(y_cbe), "--", label="cBE")
    if Y_obs is not None:
        ax.axhline(1.0, color="black", linewidth=1.0, alpha=0.55)
    if ymin is not None:
        ax.set_ylim(bottom=float(ymin))

    ax.set_xlabel(r"$x=m_\chi/T_\mathrm{SM}$")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)

    return ax, {"fBE": fbe, "cBE": cbe, "Y_obs": Y_obs}


def plot_equation_of_state(run, cbe=None, ax=None):
    metadata = run["metadata"]
    moments = compute_moments_along_trajectory(
        traj=run["traj"],
        q=run["q"],
        m=float(metadata["m_chi"]),
        g=1.0,
    )

    a = np.array([row["a"] for row in moments])
    w = np.array([row["w"] for row in moments])

    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 5.0))

    ax.semilogx(a, w, label=r"fBE $w=P/\rho$")

    if cbe is not None:
        a_cbe, _, T_cbe = cbe_arrays_for_plot(cbe, a)
        w_cbe = cbe_w_from_T(T_cbe, float(metadata["m_chi"]))
        ax.semilogx(a_cbe, w_cbe, "--", label=r"cBE $w=P/\rho$")

    ax.axhline(1.0 / 3.0, color="black", linewidth=1.0, alpha=0.45, linestyle=":")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.3)
    ax.set_xlabel(r"$a$")
    ax.set_ylabel(r"$w=P/\rho$")
    ax.set_ylim(bottom=-0.02, top=0.38)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)

    return ax, {"moments": moments}


def plot_equation_of_state_x(run, cbe=None, cosmo=None, ax=None):
    if cosmo is None:
        cosmo = make_cosmology()

    metadata = run["metadata"]
    m_chi = float(metadata["m_chi"])
    moments = compute_moments_along_trajectory(
        traj=run["traj"],
        q=run["q"],
        m=m_chi,
        g=1.0,
    )

    a = np.array([row["a"] for row in moments])
    x = x_from_a(a, m_chi, cosmo)
    w = np.array([row["w"] for row in moments])
    x, w = sort_xy_by_x(x, w)

    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 5.0))

    ax.semilogx(x, w, label=r"fBE $w=P/\rho$")

    if cbe is not None:
        a_cbe, _, T_cbe = cbe_arrays_for_plot(cbe, a)
        x_cbe = x_from_a(a_cbe, m_chi, cosmo)
        w_cbe = cbe_w_from_T(T_cbe, m_chi)
        x_cbe, w_cbe = sort_xy_by_x(x_cbe, w_cbe)
        ax.semilogx(x_cbe, w_cbe, "--", label=r"cBE $w=P/\rho$")

    ax.axhline(1.0 / 3.0, color="black", linewidth=1.0, alpha=0.45, linestyle=":")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.3)
    ax.set_xlabel(r"$x=m_\chi/T_\mathrm{SM}$")
    ax.set_ylabel(r"$w=P/\rho$")
    ax.set_ylim(bottom=-0.02, top=0.38)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)

    return ax, {"moments": moments}


def scattering_switch_summary(run, window=4):
    metadata = run["metadata"]
    traj = run["traj"]
    a = tensor_to_numpy(traj["a"])
    modes = traj.get("mode_hist", [""] * len(a))
    moments = compute_moments_along_trajectory(
        traj=traj,
        q=run["q"],
        m=float(metadata["m_chi"]),
        g=1.0,
    )

    switch_idx = None
    for idx in range(1, len(modes)):
        if modes[idx] == "FULL_HEUN" and modes[idx - 1] != "FULL_HEUN":
            switch_idx = idx
            break

    lo = 0 if switch_idx is None else max(0, switch_idx - window)
    hi = len(a) if switch_idx is None else min(len(a), switch_idx + window + 1)

    rows = []
    for idx in range(lo, hi):
        rows.append(
            {
                "idx": idx,
                "a": float(a[idx]),
                "mode": modes[idx],
                "n": moments[idx]["n"],
                "Tkin": moments[idx]["Tkin"],
                "avg_v": moments[idx]["avg_v"],
                "v_rms": moments[idx]["v_rms"],
            }
        )

    return {
        "run_name": metadata.get("run_name", ""),
        "switch_idx": switch_idx,
        "a_switch": None if switch_idx is None else float(a[switch_idx - 1]),
        "a_first_full": None if switch_idx is None else float(a[switch_idx]),
        "heun_status": traj.get("heun_status"),
        "settings": traj.get("settings", {}),
        "rows": rows,
    }


def print_scattering_switch_summary(run, window=4):
    summary = scattering_switch_summary(run, window=window)
    print(f"run: {summary['run_name']}")
    print(f"switch_idx: {summary['switch_idx']}")
    print(f"a_switch: {summary['a_switch']}")
    print(f"a_first_full: {summary['a_first_full']}")
    print(f"heun_status: {summary['heun_status']}")
    print("idx  a  mode  Tkin  <v>  v_rms  n")
    for row in summary["rows"]:
        print(
            f"{row['idx']:4d} "
            f"{row['a']:.8e} "
            f"{row['mode']:>9s} "
            f"{row['Tkin']:.8e} "
            f"{row['avg_v']:.8e} "
            f"{row['v_rms']:.8e} "
            f"{row['n']:.8e}"
        )
    return summary


def plot_scattering_switch_diagnostics(run, cbe=None, cosmo=None, velocity_quantity="avg_v"):
    if velocity_quantity not in {"avg_v", "v_rms"}:
        raise ValueError("velocity_quantity must be either 'avg_v' or 'v_rms'.")

    metadata = run["metadata"]
    traj = run["traj"]
    a = tensor_to_numpy(traj["a"])
    modes = traj.get("mode_hist", [""] * len(a))
    moments = compute_moments_along_trajectory(
        traj=traj,
        q=run["q"],
        m=float(metadata["m_chi"]),
        g=1.0,
    )

    Tkin = np.array([row["Tkin"] for row in moments])
    velocity = np.array([row[velocity_quantity] for row in moments])
    n = np.array([row["n"] for row in moments])
    velocity_label = r"$\langle v\rangle$" if velocity_quantity == "avg_v" else r"$v_\mathrm{rms}$"

    switch_idx = None
    for idx in range(1, len(modes)):
        if modes[idx] == "FULL_HEUN" and modes[idx - 1] != "FULL_HEUN":
            switch_idx = idx
            break

    fig, axs = plt.subplots(3, 1, figsize=(7.2, 9.0), sharex=True)
    axs[0].loglog(a, positive_finite_or_nan(Tkin, 1e-30), label=r"fBE $T_\mathrm{kin}$")
    axs[1].loglog(a, positive_finite_or_nan(velocity, 1e-12), label=fr"fBE {velocity_label}")
    axs[2].loglog(a, positive_finite_or_nan(n, 1e-250), label=r"fBE $n$")

    if cbe is not None:
        a_cbe, N_cbe, T_cbe = cbe_arrays_for_plot(cbe, a)
        n_cbe = N_cbe / a_cbe**3
        velocity_cbe = cbe_velocity_moments_from_T(T_cbe, float(metadata["m_chi"]))[velocity_quantity]
        axs[0].loglog(a_cbe, positive_finite_or_nan(T_cbe, 1e-30), "--", label=r"cBE $T_s$")
        axs[1].loglog(a_cbe, positive_finite_or_nan(velocity_cbe, 1e-12), "--", label=fr"cBE {velocity_label}")
        axs[2].loglog(a_cbe, positive_finite_or_nan(n_cbe, 1e-250), "--", label=r"cBE $n$")

    if cosmo is not None:
        T_sm = np.array([float(cosmo.T_of_a(ai)) for ai in a])
        axs[0].loglog(a, positive_finite_or_nan(T_sm, 1e-30), ":", label=r"$T_\mathrm{SM}$")

    if switch_idx is not None:
        a_switch = float(a[switch_idx - 1])
        a_first_full = float(a[switch_idx])
        for ax in axs:
            ax.axvline(a_switch, color="black", linewidth=1.0, alpha=0.6, label="switch")
            ax.axvline(a_first_full, color="tab:red", linewidth=1.0, alpha=0.6, linestyle=":", label="first stored FULL")

    axs[0].set_ylabel("temperature")
    axs[1].set_ylabel(velocity_label)
    axs[2].set_ylabel(r"$n$")
    axs[2].set_xlabel(r"$a$")

    for ax in axs:
        ax.grid(True, which="both", alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), frameon=False)

    return fig, scattering_switch_summary(run)


def plot_scattering_switch_diagnostics_x(run, cbe=None, cosmo=None, velocity_quantity="avg_v"):
    if cosmo is None:
        cosmo = make_cosmology()
    if velocity_quantity not in {"avg_v", "v_rms"}:
        raise ValueError("velocity_quantity must be either 'avg_v' or 'v_rms'.")

    metadata = run["metadata"]
    m_chi = float(metadata["m_chi"])
    traj = run["traj"]
    a = tensor_to_numpy(traj["a"])
    x = x_from_a(a, m_chi, cosmo)
    modes = traj.get("mode_hist", [""] * len(a))
    moments = compute_moments_along_trajectory(
        traj=traj,
        q=run["q"],
        m=m_chi,
        g=1.0,
    )

    Tkin = np.array([row["Tkin"] for row in moments])
    velocity = np.array([row[velocity_quantity] for row in moments])
    n = np.array([row["n"] for row in moments])
    T_sm = np.array([float(cosmo.T_of_a(ai)) for ai in a])
    x, Tkin, velocity, n, T_sm = sort_xy_by_x(x, Tkin, velocity, n, T_sm)
    velocity_label = r"$\langle v\rangle$" if velocity_quantity == "avg_v" else r"$v_\mathrm{rms}$"

    switch_idx = None
    for idx in range(1, len(modes)):
        if modes[idx] == "FULL_HEUN" and modes[idx - 1] != "FULL_HEUN":
            switch_idx = idx
            break

    fig, axs = plt.subplots(3, 1, figsize=(7.2, 9.0), sharex=True)
    axs[0].loglog(x, positive_finite_or_nan(Tkin, 1e-30), label=r"fBE $T_\mathrm{kin}$")
    axs[1].loglog(x, positive_finite_or_nan(velocity, 1e-12), label=fr"fBE {velocity_label}")
    axs[2].loglog(x, positive_finite_or_nan(n, 1e-250), label=r"fBE $n$")

    if cbe is not None:
        a_cbe, N_cbe, T_cbe = cbe_arrays_for_plot(cbe, a)
        x_cbe = x_from_a(a_cbe, m_chi, cosmo)
        n_cbe = N_cbe / a_cbe**3
        velocity_cbe = cbe_velocity_moments_from_T(T_cbe, m_chi)[velocity_quantity]
        x_cbe, T_cbe, velocity_cbe, n_cbe = sort_xy_by_x(x_cbe, T_cbe, velocity_cbe, n_cbe)
        axs[0].loglog(x_cbe, positive_finite_or_nan(T_cbe, 1e-30), "--", label=r"cBE $T_s$")
        axs[1].loglog(x_cbe, positive_finite_or_nan(velocity_cbe, 1e-12), "--", label=fr"cBE {velocity_label}")
        axs[2].loglog(x_cbe, positive_finite_or_nan(n_cbe, 1e-250), "--", label=r"cBE $n$")

    axs[0].loglog(x, positive_finite_or_nan(T_sm, 1e-30), ":", label=r"$T_\mathrm{SM}$")

    if switch_idx is not None:
        x_switch = float(x_from_a(a[switch_idx - 1], m_chi, cosmo))
        x_first_full = float(x_from_a(a[switch_idx], m_chi, cosmo))
        for ax in axs:
            ax.axvline(x_switch, color="black", linewidth=1.0, alpha=0.6, label="switch")
            ax.axvline(x_first_full, color="tab:red", linewidth=1.0, alpha=0.6, linestyle=":", label="first stored FULL")

    axs[0].set_ylabel("temperature")
    axs[1].set_ylabel(velocity_label)
    axs[2].set_ylabel(r"$n$")
    axs[2].set_xlabel(r"$x=m_\chi/T_\mathrm{SM}$")

    for ax in axs:
        ax.grid(True, which="both", alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), frameon=False)

    return fig, scattering_switch_summary(run)


def print_final_velocity_summary(run):
    metadata = run["metadata"]
    snap = get_snapshot_qspace(run["traj"], run["q"], idx=-1)
    mom = compute_moments_from_q_snapshot(
        q=run["q"],
        f=snap["f"],
        a=snap["a"],
        m=float(metadata["m_chi"]),
        g=1.0,
    )

    print("Final velocity / moment summary:")
    print(f"  a       = {snap['a']:.6e}")
    print(f"  n       = {mom['n']:.6e}")
    print(f"  rho     = {mom['rho']:.6e}")
    print(f"  P       = {mom['P']:.6e}")
    print(f"  w=P/rho = {mom['w']:.6e}")
    print(f"  Tkin    = {mom['Tkin']:.6e}")
    print(f"  avg_p   = {mom['avg_p']:.6e}")
    print(f"  avg_v   = {mom['avg_v']:.6e}")
    print(f"  v_rms   = {mom['v_rms']:.6e}")
    return mom


def main():
    parser = argparse.ArgumentParser(description="Load and plot fBE solver runs.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--pattern", default="hybrid_saved_*")
    parser.add_argument("--run", default=None, help="Run directory name for snapshot evolution plot.")
    parser.add_argument("--quantity", default="q_p2f")
    parser.add_argument("--color-by", choices=["x", "T", "a"], default="x")
    parser.add_argument("--yscale", choices=["linear", "log"], default="linear")
    parser.add_argument(
        "--velocity",
        choices=["avg_v", "v_rms"],
        default="avg_v",
        help="Velocity moment for the thermo plot. avg_v uses <p/E>; v_rms uses sqrt(<(p/E)^2>).",
    )
    parser.add_argument(
        "--plot",
        choices=[
            "snapshots",
            "final",
            "moments",
            "moments_x",
            "cbe",
            "thermo",
            "thermo_x",
            "omega",
            "omega_x",
            "w",
            "w_x",
            "tail",
        ],
        default="snapshots",
    )
    parser.add_argument("--cbe-dir", default=str(DEFAULT_RESULTS_DIR / "cbe_benchmark"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    loaded = load_all_runs_matching(pattern=args.pattern, results_dir=args.results_dir)
    if not loaded:
        raise FileNotFoundError(f"No runs matching {args.pattern!r} in {args.results_dir}")

    cosmo = make_cosmology()
    cbe = None
    cbe_solution = None
    if args.plot in {"cbe", "thermo", "thermo_x", "final", "omega", "omega_x", "w", "w_x"}:
        try:
            cbe_solution = load_cbe_solution(args.cbe_dir)
            cbe_data, _ = cbe_solution
            cbe = make_cbe_interpolators(cbe_data)
        except FileNotFoundError:
            cbe = None
            cbe_solution = None

    if args.plot == "final":
        plot_final_runs_with_inferred_MB(loaded, cbe=cbe, quantity=args.quantity, yscale=args.yscale)
    else:
        run_name = args.run or sorted(loaded)[-1]
        run = loaded[run_name]

        if args.plot == "snapshots":
            plot_snapshots_evolution(
                run,
                quantity=args.quantity,
                color_by=args.color_by,
                yscale=args.yscale,
                cosmo=cosmo,
            )
        elif args.plot == "moments":
            plot_moment_evolution(run, cosmo=cosmo)
        elif args.plot == "moments_x":
            plot_moment_evolution_x(run, cosmo=cosmo)
        elif args.plot == "cbe":
            if cbe is None:
                raise FileNotFoundError(f"Missing cBE solution in {args.cbe_dir}")
            plot_final_against_cbe(run, cbe=cbe, quantity=args.quantity, yscale=args.yscale)
        elif args.plot == "thermo":
            plot_temperature_fugacity_velocity(run, cbe=cbe, cosmo=cosmo, velocity_quantity=args.velocity)
        elif args.plot == "thermo_x":
            plot_temperature_fugacity_velocity_x(run, cbe=cbe, cosmo=cosmo, velocity_quantity=args.velocity)
        elif args.plot == "omega":
            plot_omega_evolution(run, cbe_solution=cbe_solution, cosmo=cosmo)
        elif args.plot == "omega_x":
            plot_omega_evolution_x(run, cbe_solution=cbe_solution, cosmo=cosmo)
        elif args.plot == "w":
            plot_equation_of_state(run, cbe=cbe)
        elif args.plot == "w_x":
            plot_equation_of_state_x(run, cbe=cbe, cosmo=cosmo)
        elif args.plot == "tail":
            diag = tail_diagnostics(run)
            for key, value in diag.items():
                print(f"{key}: {value}")
            return

    plt.tight_layout()

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=250)
        print(f"Saved: {out}")

    if args.show or not args.output:
        plt.show()


if __name__ == "__main__":
    main()
