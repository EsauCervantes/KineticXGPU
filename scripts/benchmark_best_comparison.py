# benchmark_best_comparison.py
#
# Compare the projected dense KineticXGPU 2->2 operator ("this work") against
# the local BEST checkout.  The default BEST baseline is its semi-analytical
# 2->2 evaluator, because the generic Vegas mode is much noisier and usually
# needs substantial tuning of neval/nitn.

import argparse
import csv
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch

logging.getLogger("fontTools").setLevel(logging.ERROR)


THIS_FILE = Path(__file__).resolve()
SCRIPTS_DIR = THIS_FILE.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_BEST_DIR = PROJECT_ROOT / "external" / "BEST"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from grid_log import make_log_q_grid
from collision import C_quantum


RESULTS_DIR = PROJECT_ROOT / "results"
BENCHMARK_DIR = RESULTS_DIR / "CPU-GPU_benchmarks"
PLOTS_DIR = RESULTS_DIR / "plots"


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def get_mpi_info_if_available():
    rank_keys = ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "MPI_RANKID")
    size_keys = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "MPI_WORLD_SIZE")

    rank = 0
    size = 1
    for key in rank_keys:
        if key in os.environ:
            rank = int(os.environ[key])
            break
    for key in size_keys:
        if key in os.environ:
            size = int(os.environ[key])
            break
    return rank, size


def median(values):
    values = sorted(values)
    return values[len(values) // 2]


def latex_scientific(value, precision=2):
    value = float(value)
    if value == 0.0 or not math.isfinite(value):
        return f"{value:.{precision}g}"

    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    mantissa_text = f"{mantissa:.{precision}f}".rstrip("0").rstrip(".")
    if mantissa_text == "1":
        return rf"10^{{{exponent}}}"
    if mantissa_text == "-1":
        return rf"-10^{{{exponent}}}"
    return rf"{mantissa_text}\times 10^{{{exponent}}}"


def set_paper_plot_style(font_size=12, serif_font="Latin Modern Roman"):
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": [
                serif_font,
                "Latin Modern Roman",
                "Computer Modern Roman",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "custom",
            "mathtext.rm": serif_font,
            "mathtext.it": f"{serif_font}:italic",
            "mathtext.bf": f"{serif_font}:bold",
            "mathtext.cal": serif_font,
            "mathtext.sf": serif_font,
            "mathtext.tt": "Latin Modern Mono",
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "legend.fontsize": font_size - 1,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "figure.titlesize": font_size,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def validation_this_work_label(scale=None):
    return r"${\rm KineticXGPU}$"


def add_validation_legend(ax):
    y0 = 0.86
    dy = 0.085
    x_line0 = 0.035
    x_line1 = 0.105
    x_text = 0.13

    ax.plot(
        [x_line0, x_line1],
        [y0, y0],
        transform=ax.transAxes,
        color="black",
        lw=1.8,
        clip_on=False,
        zorder=5,
    )
    ax.text(
        x_text,
        y0,
        "BEST",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=11,
        zorder=5,
    )
    ax.text(
        x_text + 0.115,
        y0,
        "(semi-analytical)",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8,
        zorder=5,
    )

    y1 = y0 - dy
    ax.plot(
        [x_line0, x_line1],
        [y1, y1],
        transform=ax.transAxes,
        color="tab:blue",
        lw=1.5,
        ls="--",
        clip_on=False,
        zorder=5,
    )
    ax.text(
        x_text,
        y1,
        "KineticXGPU",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=11,
        zorder=5,
    )


def two_decimal_tick_label(value, pos=None):
    value = float(value)
    if abs(value) < 5.0e-3:
        value = 0.0
    return f"{value:.2f}"


def tensor_to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu())
    return float(x)


def make_two_bump_torch(
    q,
    q1,
    q2,
    sigma1=0.18,
    sigma2=0.22,
    A1=1.0,
    A2=0.5,
):
    logq = torch.log(q)
    bump1 = A1 * torch.exp(-0.5 * ((logq - math.log(q1)) / sigma1) ** 2)
    bump2 = A2 * torch.exp(-0.5 * ((logq - math.log(q2)) / sigma2) ** 2)
    return bump1 + bump2


def make_two_bump_numpy(
    r,
    q1,
    q2,
    sigma1=0.18,
    sigma2=0.22,
    A1=1.0,
    A2=0.5,
):
    logq = math.log(r)
    bump1 = A1 * math.exp(-0.5 * ((logq - math.log(q1)) / sigma1) ** 2)
    bump2 = A2 * math.exp(-0.5 * ((logq - math.log(q2)) / sigma2) ** 2)
    return bump1 + bump2


def estimate_bytes_per_batch(
    *,
    N,
    batch_size,
    dtype,
    Ng,
    safety_tensor_count=None,
):
    bytes_per_float = torch.tensor([], dtype=dtype).element_size()
    n_tensors = 18 if safety_tensor_count is None else safety_tensor_count
    angular_factor = max(1.0, 0.35 * Ng)
    effective_tensors = n_tensors + angular_factor
    return int(effective_tensors * batch_size * N * N * bytes_per_float)


def memory_aware_batch_size_cuda(
    *,
    N,
    dtype,
    Ng,
    memory_fraction=0.60,
    max_batch_size=128,
    safety_tensor_count=None,
):
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    usable_bytes = int(memory_fraction * free_bytes)

    candidates = []
    b = min(max_batch_size, N)
    while b >= 1:
        candidates.append(b)
        b //= 2

    if 1 not in candidates:
        candidates.append(1)

    for b in candidates:
        needed = estimate_bytes_per_batch(
            N=N,
            batch_size=b,
            dtype=dtype,
            Ng=Ng,
            safety_tensor_count=safety_tensor_count,
        )
        if needed <= usable_bytes:
            return {
                "batch_size": b,
                "batch_mode": "memory_aware_cuda",
                "estimated_batch_bytes": needed,
                "cuda_free_bytes": free_bytes,
                "cuda_total_bytes": total_bytes,
            }

    needed = estimate_bytes_per_batch(
        N=N,
        batch_size=1,
        dtype=dtype,
        Ng=Ng,
        safety_tensor_count=safety_tensor_count,
    )
    return {
        "batch_size": 1,
        "batch_mode": "memory_aware_cuda_minimum",
        "estimated_batch_bytes": needed,
        "cuda_free_bytes": free_bytes,
        "cuda_total_bytes": total_bytes,
    }


def choose_this_work_batch_size(
    *,
    N,
    device,
    dtype,
    Ng,
    fixed_batch_size,
    memory_aware,
    memory_fraction,
    max_batch_size,
    safety_tensor_count,
):
    if memory_aware and device.type == "cuda":
        return memory_aware_batch_size_cuda(
            N=N,
            dtype=dtype,
            Ng=Ng,
            memory_fraction=memory_fraction,
            max_batch_size=max_batch_size,
            safety_tensor_count=safety_tensor_count,
        )

    return {
        "batch_size": min(int(fixed_batch_size), N),
        "batch_mode": "fixed",
        "estimated_batch_bytes": None,
        "cuda_free_bytes": None,
        "cuda_total_bytes": None,
    }


def resolve_output_path(out_name):
    out = Path(out_name)
    if out.is_absolute():
        return out
    if out.parent == Path("."):
        return BENCHMARK_DIR / out
    return PROJECT_ROOT / out


def import_best(best_dir):
    best_dir = Path(best_dir).expanduser()
    if not best_dir.is_absolute():
        best_dir = PROJECT_ROOT / best_dir

    if not best_dir.is_dir():
        raise FileNotFoundError(
            f"BEST checkout not found: {best_dir}. "
            "Clone BEST separately and pass --best-dir /path/to/BEST."
        )
    if str(best_dir) not in sys.path:
        sys.path.insert(0, str(best_dir))

    try:
        from besthep import BEST
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise ModuleNotFoundError(
            f"Could not import BEST because dependency '{missing}' is missing. "
            "Install BEST dependencies in this Python environment first: "
            "numpy scipy mpi4py vegas."
        ) from exc

    return BEST


def constant_matrix_element(momenta, coupling):
    # BEST convention: coupling is the amplitude parameter and |M|^2=coupling^2.
    if momenta is not None and getattr(momenta, "ndim", 0) == 3:
        import numpy as np

        return np.full(momenta.shape[2], coupling**2)
    return coupling**2


def make_this_work_state(
    *,
    N,
    device,
    dtype,
    q_min,
    q_max,
    mass,
    q1,
    q2,
    A1,
    A2,
):
    q, _, _, _ = make_log_q_grid(
        q_min,
        q_max,
        N,
        device=device,
        dtype=dtype,
        base=10.0,
    )
    a = torch.tensor(1.0, device=device, dtype=dtype)
    m = torch.tensor(mass, device=device, dtype=dtype)
    f = make_two_bump_torch(q, q1=q1, q2=q2, A1=A1, A2=A2)
    f = torch.clamp(f, min=torch.finfo(dtype).tiny)
    return q, a, m, f


def call_this_work_operator(
    *,
    f,
    a,
    q,
    m,
    lam,
    Ng,
    batch_size,
    stat,
    enforce_self_projection,
    return_diagnostics=False,
):
    return C_quantum(
        f=f,
        a=a,
        q=q,
        m=m,
        lam=lam,
        Ng=Ng,
        batch_size=batch_size,
        statistics=stat,
        enforce_self_projection=enforce_self_projection,
        return_diagnostics=return_diagnostics,
    )


def run_this_work(
    *,
    N,
    device,
    dtype,
    q_min,
    q_max,
    mass,
    lam,
    Ng,
    batch_info,
    repeats,
    warmup,
    q1,
    q2,
    A1,
    A2,
    stat,
    enforce_self_projection,
    diagnostics,
):
    batch_size = int(batch_info["batch_size"])
    q, a, m, f = make_this_work_state(
        N=N,
        device=device,
        dtype=dtype,
        q_min=q_min,
        q_max=q_max,
        mass=mass,
        q1=q1,
        q2=q2,
        A1=A1,
        A2=A2,
    )

    for _ in range(warmup):
        _ = call_this_work_operator(
            f=f,
            a=a,
            q=q,
            m=m,
            lam=lam,
            Ng=Ng,
            batch_size=batch_size,
            stat=stat,
            enforce_self_projection=enforce_self_projection,
            return_diagnostics=False,
        )
        synchronize(device)

    times = []
    last_C = None

    for _ in range(repeats):
        synchronize(device)
        t0 = time.perf_counter()
        last_C, _, _, _ = call_this_work_operator(
            f=f,
            a=a,
            q=q,
            m=m,
            lam=lam,
            Ng=Ng,
            batch_size=batch_size,
            stat=stat,
            enforce_self_projection=enforce_self_projection,
            return_diagnostics=False,
        )
        synchronize(device)
        times.append(time.perf_counter() - t0)

    row = {
        "implementation": "this_work",
        "best_mode": "",
        "device": device.type,
        "N": N,
        "shape": "two_bump",
        "time_median_s": median(times),
        "time_min_s": min(times),
        "time_max_s": max(times),
        "finite_C": bool(torch.isfinite(last_C).all().item()),
        "Ng": Ng,
        "batch_size": batch_size,
        "batch_mode": batch_info["batch_mode"],
        "estimated_batch_bytes": batch_info["estimated_batch_bytes"],
        "cuda_free_bytes": batch_info["cuda_free_bytes"],
        "cuda_total_bytes": batch_info["cuda_total_bytes"],
        "lam": lam,
        "m": mass,
        "q_min": q_min,
        "q_max": q_max,
        "q1": q1,
        "q2": q2,
        "A1": A1,
        "A2": A2,
        "projected": bool(enforce_self_projection),
        "this_work_stat": stat,
        "dtype": "float64" if dtype == torch.float64 else "float32",
    }

    if diagnostics:
        C, _, _, _, diag = call_this_work_operator(
            f=f,
            a=a,
            q=q,
            m=m,
            lam=lam,
            Ng=Ng,
            batch_size=batch_size,
            stat=stat,
            enforce_self_projection=enforce_self_projection,
            return_diagnostics=True,
        )
        row["finite_C_diagnostics"] = bool(torch.isfinite(C).all().item())
        for key, value in diag.items():
            row[key] = tensor_to_float(value)

    return row


def build_best_solver(
    *,
    BEST,
    N,
    q_min,
    q_max,
    mass,
    lam,
    q1,
    q2,
    A1,
    A2,
    stat,
    neval,
    nitn,
    delta_width,
    max_rel_change,
    adapt_width,
    max_rel_err,
    min_rel_err,
    n_r_parallel,
):
    solver = BEST(
        q_min=q_min,
        q_max=q_max,
        n_grid=N,
        max_rel_change=max_rel_change,
        adapt_width=adapt_width,
        max_rel_err=max_rel_err,
        min_rel_err=min_rel_err,
        n_r_parallel=n_r_parallel,
    )
    solver.initialize_species(
        "phi",
        lambda r: make_two_bump_numpy(r, q1=q1, q2=q2, A1=A1, A2=A2),
        stat=stat,
        mass=mass,
        dof=1,
    )
    solver.add_process(
        "phi_2to2",
        ["phi", "phi"],
        ["phi", "phi"],
        constant_matrix_element,
        coupling=lam,
        neval=neval,
        nitn=nitn,
        delta_width=delta_width,
    )
    return solver


def time_best_analytical(
    *,
    solver,
    n_F,
    repeats,
    warmup,
    include_cold,
):
    import numpy as np

    rows = []

    if include_cold:
        solver._analytical_integrators = {}
        t0 = time.perf_counter()
        rates = solver._compute_rates_all_species("phi_2to2", n_F=n_F, M_squared=None)
        cold_time = time.perf_counter() - t0
        if solver.world_rank == 0:
            rows.append(
                {
                    "implementation": "BEST",
                    "best_mode": "analytical_cold",
                    "device": "cpu",
                    "time_median_s": cold_time,
                    "time_min_s": cold_time,
                    "time_max_s": cold_time,
                    "finite_C": bool(np.isfinite(rates["phi"]).all()),
                }
            )

    for _ in range(warmup):
        _ = solver._compute_rates_all_species("phi_2to2", n_F=n_F, M_squared=None)

    times = []
    rates = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        rates = solver._compute_rates_all_species("phi_2to2", n_F=n_F, M_squared=None)
        times.append(time.perf_counter() - t0)

    if solver.world_rank == 0:
        timed_mode = "analytical_warm" if (warmup > 0 or include_cold) else "analytical_cold"
        rows.append(
            {
                "implementation": "BEST",
                "best_mode": timed_mode,
                "device": "cpu",
                "time_median_s": median(times),
                "time_min_s": min(times),
                "time_max_s": max(times),
                "finite_C": bool(np.isfinite(rates["phi"]).all()),
            }
        )

    return rows, rates


def time_best_vegas(*, solver, repeats, warmup):
    import numpy as np

    for _ in range(warmup):
        _ = solver._compute_rates_vegas(["phi_2to2"], t=0.0)

    times = []
    rates = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        rates, _, _ = solver._compute_rates_vegas(["phi_2to2"], t=0.0)
        times.append(time.perf_counter() - t0)

    if solver.world_rank != 0:
        return [], rates

    return [
        {
            "implementation": "BEST",
            "best_mode": "vegas",
            "device": "cpu",
            "time_median_s": median(times),
            "time_min_s": min(times),
            "time_max_s": max(times),
            "finite_C": bool(np.isfinite(rates["phi"]).all()),
        }
    ], rates


def run_best(
    *,
    N,
    args,
):
    BEST = import_best(args.best_dir)
    solver = build_best_solver(
        BEST=BEST,
        N=N,
        q_min=args.q_min,
        q_max=args.q_max,
        mass=args.m,
        lam=args.lam,
        q1=args.q1,
        q2=args.q2,
        A1=args.A1,
        A2=args.A2,
        stat=args.best_stat,
        neval=args.best_neval,
        nitn=args.best_nitn,
        delta_width=args.best_delta_width,
        max_rel_change=args.best_max_rel_change,
        adapt_width=not args.best_no_adapt_width,
        max_rel_err=args.best_max_rel_err,
        min_rel_err=args.best_min_rel_err,
        n_r_parallel=args.best_n_r_parallel,
    )

    if args.best_mode == "analytical":
        n_F = N if args.best_nF is None else int(args.best_nF)
        rows, rates = time_best_analytical(
            solver=solver,
            n_F=n_F,
            repeats=args.repeats,
            warmup=args.best_warmup,
            include_cold=args.best_include_cold,
        )
    elif args.best_mode == "vegas":
        rows, rates = time_best_vegas(
            solver=solver,
            repeats=args.repeats,
            warmup=args.best_warmup,
        )
    else:
        raise ValueError(f"Unknown BEST mode: {args.best_mode}")

    for row in rows:
        row.update(
            {
                "N": N,
                "shape": "two_bump",
                "Ng": "",
                "batch_size": "",
                "lam": args.lam,
                "m": args.m,
                "q_min": args.q_min,
                "q_max": args.q_max,
                "q1": args.q1,
                "q2": args.q2,
                "A1": args.A1,
                "A2": args.A2,
                "projected": "",
                "this_work_stat": args.this_work_stat,
                "best_neval": args.best_neval,
                "best_nitn": args.best_nitn,
                "best_nF": N if args.best_nF is None else int(args.best_nF),
                "best_delta_width": args.best_delta_width,
                "best_stat": args.best_stat,
                "best_n_r_parallel": args.best_n_r_parallel,
                "best_max_rel_err": args.best_max_rel_err,
                "best_min_rel_err": args.best_min_rel_err,
            }
        )

    payload = None
    if getattr(solver, "world_rank", 0) == 0 and rates is not None and "phi" in rates:
        payload = {
            "N": N,
            "q": solver.r_grids["phi"].copy(),
            "C_best": rates["phi"].copy(),
            "best_mode": rows[-1]["best_mode"] if rows else args.best_mode,
            "best_nF": N if args.best_nF is None else int(args.best_nF),
        }

    return rows, payload


def save_rows(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_this_work_validation_on_best_grid(best_payload, args):
    import numpy as np

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device("cpu")
    q = torch.as_tensor(best_payload["q"], device=device, dtype=dtype)
    a = torch.tensor(1.0, device=device, dtype=dtype)
    m = torch.tensor(args.m, device=device, dtype=dtype)
    f = make_two_bump_torch(q, q1=args.q1, q2=args.q2, A1=args.A1, A2=args.A2)
    f = torch.clamp(f, min=torch.finfo(dtype).tiny)

    C, E, p, dp = call_this_work_operator(
        f=f,
        a=a,
        q=q,
        m=m,
        lam=args.lam,
        Ng=args.Ng,
        batch_size=min(int(args.batch_size), int(best_payload["N"])),
        stat=args.this_work_stat,
        enforce_self_projection=args.enforce_self_projection,
        return_diagnostics=False,
    )

    return {
        "q": q.detach().cpu().double().numpy(),
        "C_this": C.detach().cpu().double().numpy(),
        "E": E.detach().cpu().double().numpy(),
        "p": p.detach().cpu().double().numpy(),
        "dp": dp.detach().cpu().double().numpy(),
        "f": f.detach().cpu().double().numpy(),
    }


def validation_metrics(q, C_this, C_best, p, E, dp):
    import numpy as np

    C_this = np.asarray(C_this, dtype=float)
    C_best = np.asarray(C_best, dtype=float)
    w = np.asarray(p, dtype=float) ** 2 * np.asarray(dp, dtype=float)

    denom_best = np.sum(w * C_best * C_best)
    denom_this = np.sum(w * C_this * C_this)
    cross = np.sum(w * C_this * C_best)

    rel_l2 = np.sqrt(
        np.sum(w * (C_this - C_best) ** 2) / max(denom_best, 1e-300)
    )

    alpha_best_over_this = cross / max(denom_this, 1e-300)
    C_this_scaled = alpha_best_over_this * C_this
    rel_l2_scaled = np.sqrt(
        np.sum(w * (C_this_scaled - C_best) ** 2) / max(denom_best, 1e-300)
    )

    cosine = cross / np.sqrt(max(denom_this * denom_best, 1e-300))
    max_abs_best = np.max(np.abs(C_best))
    max_abs_diff = np.max(np.abs(C_this - C_best))
    max_abs_diff_scaled = np.max(np.abs(C_this_scaled - C_best))

    I_N_this = np.sum(w * C_this)
    I_E_this = np.sum(w * E * C_this)
    I_N_best = np.sum(w * C_best)
    I_E_best = np.sum(w * E * C_best)
    scale_N_this = np.sum(np.abs(w * C_this))
    scale_E_this = np.sum(np.abs(w * E * C_this))
    scale_N_best = np.sum(np.abs(w * C_best))
    scale_E_best = np.sum(np.abs(w * E * C_best))

    return {
        "rel_l2": rel_l2,
        "alpha_best_over_this": alpha_best_over_this,
        "rel_l2_after_alpha": rel_l2_scaled,
        "cosine_weighted": cosine,
        "max_abs_diff_over_max_abs_best": max_abs_diff / max(max_abs_best, 1e-300),
        "max_abs_diff_after_alpha_over_max_abs_best": (
            max_abs_diff_scaled / max(max_abs_best, 1e-300)
        ),
        "I_number_this": I_N_this,
        "I_energy_this": I_E_this,
        "I_number_best": I_N_best,
        "I_energy_best": I_E_best,
        "rel_number_moment_this": I_N_this / max(scale_N_this, 1e-300),
        "rel_energy_moment_this": I_E_this / max(scale_E_this, 1e-300),
        "rel_number_moment_best": I_N_best / max(scale_N_best, 1e-300),
        "rel_energy_moment_best": I_E_best / max(scale_E_best, 1e-300),
    }


def save_validation_outputs(best_payloads, args, plot_path):
    if not best_payloads:
        return

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LinearLocator

    set_paper_plot_style()

    validation_dir = plot_path.parent / f"validation_{plot_path.stem}"
    validation_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    this_work_plot_scale = float(args.validation_this_work_scale)

    for payload in best_payloads:
        N = int(payload["N"])
        this = compute_this_work_validation_on_best_grid(payload, args)
        q = this["q"]
        C_this = this["C_this"]
        C_best = np.asarray(payload["C_best"], dtype=float)
        metrics = validation_metrics(
            q=q,
            C_this=C_this,
            C_best=C_best,
            p=this["p"],
            E=this["E"],
            dp=this["dp"],
        )

        row = {
            "N": N,
            "dtype": args.dtype,
            "Ng": args.Ng,
            "this_work_stat": args.this_work_stat,
            "best_stat": args.best_stat,
            "projected": bool(args.enforce_self_projection),
            "best_mode": payload["best_mode"],
            "best_nF": payload["best_nF"],
            **metrics,
        }
        summary_rows.append(row)

        np.savez(
            validation_dir / f"collision_validation_N{N}.npz",
            q=q,
            f=this["f"],
            C_this=C_this,
            C_best=C_best,
            C_this_scaled=this_work_plot_scale * C_this,
            cumulative_number_this=np.cumsum(this["p"] * this["p"] * this["dp"] * C_this),
            cumulative_number_best=np.cumsum(this["p"] * this["p"] * this["dp"] * C_best),
            cumulative_energy_this=np.cumsum(this["p"] * this["p"] * this["dp"] * this["E"] * C_this),
            cumulative_energy_best=np.cumsum(this["p"] * this["p"] * this["dp"] * this["E"] * C_best),
            p=this["p"],
            E=this["E"],
            dp=this["dp"],
            validation_this_work_scale=this_work_plot_scale,
        )

        denom = max(np.max(np.abs(C_best)), 1e-300)
        C_this_plot = this_work_plot_scale * C_this
        residual = (C_this_plot - C_best) / denom
        w = this["p"] * this["p"] * this["dp"]
        cum_N_this = np.cumsum(w * C_this)
        cum_N_best = np.cumsum(w * C_best)
        cum_E_this = np.cumsum(w * this["E"] * C_this)
        cum_E_best = np.cumsum(w * this["E"] * C_best)

        scale_N_this = max(np.sum(np.abs(w * C_this)), 1e-300)
        scale_N_best = max(np.sum(np.abs(w * C_best)), 1e-300)
        scale_E_this = max(np.sum(np.abs(w * this["E"] * C_this)), 1e-300)
        scale_E_best = max(np.sum(np.abs(w * this["E"] * C_best)), 1e-300)

        fig, (ax0, ax1) = plt.subplots(
            2,
            1,
            figsize=(6.6, 5.8),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )

        ax0.plot(q, C_best, color="black", lw=1.8)
        ax0.plot(
            q,
            C_this_plot,
            color="tab:blue",
            lw=1.5,
            ls="--",
        )
        ax0.set_xscale("log")
        ax0.set_yscale("symlog", linthresh=1e-12)
        ax0.set_ylabel(r"$C(q)$")
        add_validation_legend(ax0)
        ax0.set_title(None)
        ax0.text(
            0.035,
            0.69,
            rf"$N_{{\rm grid}}={N}$" + "\n" + rf"$N_\mu={int(args.Ng)}$",
            transform=ax0.transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "0.85", "linewidth": 0.5, "alpha": 0.75, "pad": 2.5},
        )

        ax1.axhline(0.0, color="0.6", lw=0.8)
        ax1.plot(q, residual, color="tab:blue", lw=1.2, ls="--")
        ax1.set_xscale("log")
        ax1.set_xlabel(r"$q$")
        ax1.set_ylabel(r"$\Delta C/C_{\rm max}$")
        ax1.yaxis.set_major_locator(LinearLocator(numticks=4))
        ax1.yaxis.set_major_formatter(FuncFormatter(two_decimal_tick_label))

        fig.tight_layout()
        out_pdf = validation_dir / f"collision_validation_N{N}.pdf"
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)

        fig_C, ax_C = plt.subplots(figsize=(6.4, 4.2))
        ax_C.plot(q, C_best, color="black", lw=1.8, label="BEST (semi-analytical)")
        ax_C.plot(
            q,
            C_this_plot,
            color="tab:blue",
            lw=1.5,
            ls="--",
            label=validation_this_work_label(this_work_plot_scale),
        )
        ax_C.set_xscale("log")
        ax_C.set_yscale("symlog", linthresh=1e-12)
        ax_C.set_xlabel(r"$q$")
        ax_C.set_ylabel(r"$C(q)$")
        ax_C.set_title(None)
        ax_C.text(
            0.035,
            0.79,
            rf"$N_{{\rm grid}}={N}$",
            transform=ax_C.transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "0.85", "linewidth": 0.5, "alpha": 0.75, "pad": 2.5},
        )
        ax_C.legend(frameon=False)
        fig_C.tight_layout()
        fig_C.savefig(validation_dir / f"collision_terms_N{N}.pdf", bbox_inches="tight")
        plt.close(fig_C)

        fig_mom, (ax_N, ax_E) = plt.subplots(
            2,
            1,
            figsize=(6.4, 5.2),
            sharex=True,
        )
        ax_N.axhline(0.0, color="0.6", lw=0.8)
        ax_N.plot(
            q,
            cum_N_best / scale_N_best,
            color="black",
            lw=1.8,
            label=rf"${{\rm BEST}}:\ {latex_scientific(metrics['rel_number_moment_best'], precision=2)}$",
        )
        ax_N.plot(
            q,
            cum_N_this / scale_N_this,
            color="tab:blue",
            lw=1.5,
            ls="--",
            label=rf"${{\rm this\ work}}:\ {latex_scientific(metrics['rel_number_moment_this'], precision=2)}$",
        )
        ax_N.set_xscale("log")
        ax_N.set_ylabel(r"$N_C(<q)/\sum |q^2\Delta q\,C|$")
        ax_N.legend(frameon=False, fontsize=8)

        ax_E.axhline(0.0, color="0.6", lw=0.8)
        ax_E.plot(
            q,
            cum_E_best / scale_E_best,
            color="black",
            lw=1.8,
            label=rf"${{\rm BEST}}:\ {latex_scientific(metrics['rel_energy_moment_best'], precision=2)}$",
        )
        ax_E.plot(
            q,
            cum_E_this / scale_E_this,
            color="tab:blue",
            lw=1.5,
            ls="--",
            label=rf"${{\rm this\ work}}:\ {latex_scientific(metrics['rel_energy_moment_this'], precision=2)}$",
        )
        ax_E.set_xscale("log")
        ax_E.set_xlabel(r"$q$")
        ax_E.set_ylabel(r"$E_C(<q)/\sum |q^2\Delta q\,E\,C|$")
        ax_E.legend(frameon=False, fontsize=8)

        fig_mom.tight_layout()
        fig_mom.savefig(validation_dir / f"weighted_moments_N{N}.pdf", bbox_inches="tight")
        plt.close(fig_mom)

    summary_path = validation_dir / "validation_summary.csv"
    save_rows(summary_rows, summary_path)
    print(f"Saved validation summary: {summary_path}")
    print(f"Saved validation plots:   {validation_dir}")


def make_plot(csv_path, pdf_path, legend_layout="right"):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )

    this_work = df[df["implementation"] == "this_work"].sort_values("N")
    best = df[
        (df["implementation"] == "BEST")
        & (df["best_mode"].fillna("") == "analytical_warm")
    ].sort_values("N")
    if best.empty:
        best = df[df["implementation"] == "BEST"].sort_values("N")

    def first_unique(group, column):
        if column not in group.columns:
            return None
        vals = [v for v in group[column].dropna().unique()]
        if len(vals) == 1:
            return vals[0]
        return None

    def dtype_label(dtype_value):
        if dtype_value is None:
            return ""
        key = str(dtype_value).lower()
        if key in ("float32", "torch.float32", "float"):
            return r"${\rm FP32}$"
        if key in ("float64", "torch.float64", "double"):
            return r"${\rm FP64}$"
        if key:
            return rf"{dtype_value}"
        return ""

    def this_work_label(group, device_name):
        precision = dtype_label(first_unique(group, "dtype"))
        if str(device_name) == "cuda":
            device_label = "GPU"
        elif str(device_name) == "cpu":
            device_label = "CPU"
        else:
            device_label = str(device_name)
        if precision:
            return rf"KineticXGPU ({device_label}, {precision})"
        return rf"KineticXGPU ({device_label})"

    def add_plot_legend(ax, layout):
        if layout == "center":
            return ax.legend(frameon=False, loc="center", ncol=1)
        if layout == "right":
            return ax.legend(
                frameon=False,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
                ncol=1,
            )
        if layout == "top":
            return ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
        raise ValueError("legend_layout must be 'center', 'right', or 'top'.")

    def dtype_sort_key(value):
        key = str(value).lower()
        order = {"float32": 0, "torch.float32": 0, "float": 0, "float64": 1, "torch.float64": 1, "double": 1, "": 2}
        return (order.get(key, 10), key)

    def iter_this_work_groups(frame):
        if frame.empty:
            return
        frame = frame.copy()
        if "dtype" not in frame.columns:
            frame["_plot_dtype"] = ""
        else:
            frame["_plot_dtype"] = frame["dtype"].fillna("").astype(str)

        device_order = ["cuda", "cpu"]
        extra_devices = sorted(
            dev for dev in frame["device"].dropna().unique()
            if dev not in device_order
        )
        for device_name in device_order + extra_devices:
            device_group = frame[frame["device"] == device_name]
            if device_group.empty:
                continue
            dtype_values = sorted(device_group["_plot_dtype"].dropna().unique(), key=dtype_sort_key)
            for dtype_value in dtype_values:
                group = device_group[device_group["_plot_dtype"] == dtype_value].sort_values("N")
                if group.empty:
                    continue
                yield device_name, dtype_value, group

    def derived_path(kind):
        return pdf_path.with_name(f"{kind}_{pdf_path.name}")

    runtime_path = derived_path("runtime_vs_N")
    speedup_path = derived_path("speedup_vs_N")

    # ------------------------------------------------------------
    # Runtime plot: absolute cost per collision call.
    # ------------------------------------------------------------
    fig_runtime, ax_runtime = plt.subplots(figsize=(6.4, 4.4))

    if not this_work.empty:
        for device_name, dtype_value, group in iter_this_work_groups(this_work):
            marker = "^" if str(dtype_value).lower() in ("float32", "torch.float32", "float") else "o"
            if str(device_name) == "cpu":
                marker = "o"
            ax_runtime.loglog(
                group["N"],
                group["time_median_s"],
                marker=marker,
                lw=1.8,
                label=this_work_label(group, device_name),
            )
    if not best.empty:
        best_label = "BEST (analytical)"
        ax_runtime.loglog(
            best["N"],
            best["time_median_s"],
            marker="s",
            lw=1.8,
            label=best_label,
        )

    ax_runtime.set_xlabel(r"$N_{\rm grid}$")
    ax_runtime.set_ylabel(r"runtime per collision call [s]")
    add_plot_legend(ax_runtime, legend_layout)
    fig_runtime.tight_layout()
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    fig_runtime.savefig(runtime_path)
    plt.close(fig_runtime)
    print(f"Saved runtime plot: {runtime_path}")

    # ------------------------------------------------------------
    # Speedup plot: old-style t_CPU/t_GPU for this work, plus an
    # explicitly labeled BEST CPU / this work GPU comparison.
    # ------------------------------------------------------------
    gpu_groups = [
        (dtype_value, group)
        for device_name, dtype_value, group in iter_this_work_groups(this_work)
        if str(device_name) == "cuda"
    ]
    if not gpu_groups:
        print("Skipping speedup plot: no this work GPU rows found.")
        return

    fig_speedup, ax_speedup = plt.subplots(figsize=(6.4, 4.4))
    plotted_speedup = False

    cpu_groups = {
        dtype_value: group
        for device_name, dtype_value, group in iter_this_work_groups(this_work)
        if str(device_name) == "cpu"
    }
    for dtype_value, ref in gpu_groups:
        this_work_cpu = cpu_groups.get(dtype_value)
        if this_work_cpu is None or this_work_cpu.empty:
            continue
        merged = pd.merge(
            this_work_cpu[["N", "time_median_s"]],
            ref[["N", "time_median_s"]],
            on="N",
            suffixes=("_cpu", "_gpu"),
        ).sort_values("N")
        if merged.empty:
            continue
        speedup = merged["time_median_s_cpu"] / merged["time_median_s_gpu"]
        ax_speedup.semilogx(
            merged["N"],
            speedup,
            marker="o",
            lw=1.8,
            label=rf"KineticXGPU {dtype_label(dtype_value).lstrip(', ')}: $t_{{\rm CPU}}/t_{{\rm GPU}}$",
        )
        plotted_speedup = True

    if not best.empty:
        for dtype_value, ref in gpu_groups:
            merged = pd.merge(
                best[["N", "time_median_s"]],
                ref[["N", "time_median_s"]],
                on="N",
                suffixes=("_best_cpu", "_this_work_gpu"),
            ).sort_values("N")
            if merged.empty:
                continue
            speedup = merged["time_median_s_best_cpu"] / merged["time_median_s_this_work_gpu"]
            ax_speedup.semilogx(
                merged["N"],
                speedup,
                marker="s",
                lw=1.8,
                label=rf"BEST / KineticXGPU GPU {dtype_label(dtype_value)}",
            )
            plotted_speedup = True

    if not plotted_speedup:
        plt.close(fig_speedup)
        print("Skipping speedup plot: no matching CPU/GPU or BEST/GPU rows found.")
        return

    ax_speedup.axhline(1.0, color="0.6", lw=0.8)
    ax_speedup.set_xlabel(r"$N_{\rm grid}$")
    ax_speedup.set_ylabel(r"speedup factor")
    ax_speedup.set_yscale("log")
    add_plot_legend(ax_speedup, legend_layout)
    fig_speedup.tight_layout()
    speedup_path.parent.mkdir(parents=True, exist_ok=True)
    fig_speedup.savefig(speedup_path)
    plt.close(fig_speedup)
    print(f"Saved speedup plot: {speedup_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the dense KineticXGPU 2->2 operator against local BEST."
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--include-this-work-cpu",
        action="store_true",
        help=(
            "Also benchmark the projected dense operator on CPU. "
            "Useful for plotting this work CPU/GPU speedup together with BEST CPU."
        ),
    )
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--N-list", type=int, nargs="+", default=[32, 48, 64, 96, 128])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--Ng", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--memory-aware-batch-size",
        action="store_true",
        help="For this work on CUDA, choose the largest estimated safe batch size.",
    )
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.70,
        help="Fraction of currently free CUDA memory used by the batch estimator.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=1024,
        help="Maximum candidate batch size for memory-aware mode. Effective cap is N.",
    )
    parser.add_argument(
        "--safety-tensor-count",
        type=int,
        default=None,
        help="Override the estimated count of large B x N x N tensors.",
    )
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--m", type=float, default=1.0)
    parser.add_argument("--q-min", type=float, default=1e-3)
    parser.add_argument("--q-max", type=float, default=1e2)
    parser.add_argument("--q1", type=float, default=0.3)
    parser.add_argument("--q2", type=float, default=8.0)
    parser.add_argument("--A1", type=float, default=1.0)
    parser.add_argument("--A2", type=float, default=0.5)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument(
        "--this-work-stat",
        choices=["classical", "boson", "fermion"],
        default="boson",
        help="Statistical factors for the KineticXGPU dense operator.",
    )
    parser.add_argument(
        "--enforce-self-projection",
        action="store_true",
        help=(
            "Apply the KineticXGPU number/energy conservation projection. "
            "Leave off for raw BEST-vs-this-work validation."
        ),
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Do not save C(q) validation plots and validation summary CSV.",
    )
    parser.add_argument(
        "--validation-this-work-scale",
        type=float,
        default=2.0,
        help=(
            "Fixed display factor applied to the KineticXGPU validation C(p) curve. "
            "This is only for validation plots/arrays, not timing."
        ),
    )

    parser.add_argument("--skip-best", action="store_true")
    parser.add_argument(
        "--best-dir",
        default=str(DEFAULT_BEST_DIR),
        help="Path to a local BEST checkout. The KineticXGPU repo does not vendor BEST.",
    )
    parser.add_argument("--best-mode", choices=["analytical", "vegas"], default="analytical")
    parser.add_argument("--best-warmup", type=int, default=1)
    parser.add_argument("--best-include-cold", action="store_true")
    parser.add_argument(
        "--best-nF",
        type=int,
        default=None,
        help=(
            "Internal BEST analytical integration grid. Default is N. "
            "Lower values are faster but no longer equal-resolution."
        ),
    )
    parser.add_argument("--best-neval", type=int, default=int(1e5))
    parser.add_argument("--best-nitn", type=int, default=2)
    parser.add_argument("--best-delta-width", type=float, default=0.01)
    parser.add_argument("--best-max-rel-change", type=float, default=0.3)
    parser.add_argument("--best-max-rel-err", type=float, default=0.1)
    parser.add_argument("--best-min-rel-err", type=float, default=0.01)
    parser.add_argument(
        "--best-n-r-parallel",
        type=int,
        default=None,
        help=(
            "BEST MPI r-grid groups. Default lets BEST use one r-group per MPI rank. "
            "Smaller values put more ranks on each Vegas integral."
        ),
    )
    parser.add_argument("--best-no-adapt-width", action="store_true")
    parser.add_argument(
        "--best-stat",
        default="boson",
        help=(
            "BEST species statistics. Use 'boson' for the analytical BEST path. "
            "The generic Vegas path can also accept a classical label, but the "
            "analytical helper treats non-boson as fermion in the current BEST code."
        ),
    )

    parser.add_argument(
        "--out",
        default="best_comparison_two_bump.csv",
        help="CSV output path. Relative filenames go to results/CPU-GPU_benchmarks.",
    )
    parser.add_argument(
        "--plot-out",
        default="results/plots/best_comparison_two_bump.pdf",
        help="PDF plot path.",
    )
    parser.add_argument("--no-plot", action="store_true")

    args = parser.parse_args()

    mpi_rank, mpi_size = get_mpi_info_if_available()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    this_work_devices = [args.device]
    if args.include_this_work_cpu and "cpu" not in this_work_devices:
        this_work_devices.append("cpu")

    out_path = resolve_output_path(args.out)
    plot_path = resolve_output_path(args.plot_out)

    rows = []
    validation_payloads = []
    if mpi_rank == 0:
        print(f"Project root: {PROJECT_ROOT}")
        print(f"Saving CSV:   {out_path}")
        print(f"Saving plot:  {plot_path}")
        print(f"BEST dir:     {Path(args.best_dir).expanduser()}")
        print(f"MPI ranks:    {mpi_size}")
        print(
            "this work:    dense operator on "
            f"{', '.join(this_work_devices)} ({args.dtype}), "
            f"stat={args.this_work_stat}, "
            f"projection={args.enforce_self_projection}"
        )

    for N in args.N_list:
        if mpi_rank == 0:
            for device_name in this_work_devices:
                device = torch.device(device_name)
                batch_info = choose_this_work_batch_size(
                    N=N,
                    device=device,
                    dtype=dtype,
                    Ng=args.Ng,
                    fixed_batch_size=args.batch_size,
                    memory_aware=args.memory_aware_batch_size,
                    memory_fraction=args.memory_fraction,
                    max_batch_size=args.max_batch_size,
                    safety_tensor_count=args.safety_tensor_count,
                )
                row = run_this_work(
                    N=N,
                    device=device,
                    dtype=dtype,
                    q_min=args.q_min,
                    q_max=args.q_max,
                    mass=args.m,
                    lam=args.lam,
                    Ng=args.Ng,
                    batch_info=batch_info,
                    repeats=args.repeats,
                    warmup=args.warmup,
                    q1=args.q1,
                    q2=args.q2,
                    A1=args.A1,
                    A2=args.A2,
                    stat=args.this_work_stat,
                    enforce_self_projection=args.enforce_self_projection,
                    diagnostics=args.diagnostics,
                )
                rows.append(row)
                estimated_mb = (
                    None
                    if row["estimated_batch_bytes"] is None
                    else row["estimated_batch_bytes"] / 1024**2
                )
                estimated_str = "n/a" if estimated_mb is None else f"{estimated_mb:.1f} MB"
                print(
                    f"N={N:4d} | this work {device_name:4s} | "
                    f"batch={row['batch_size']:4d} | "
                    f"mode={row['batch_mode']:18s} | "
                    f"est_mem={estimated_str:>10s} | "
                    f"time={row['time_median_s']:.4e} s | finite={row['finite_C']}"
                )

        if not args.skip_best:
            best_rows, best_payload = run_best(N=N, args=args)
            rows.extend(best_rows)
            if mpi_rank == 0 and best_payload is not None:
                validation_payloads.append(best_payload)
            if mpi_rank == 0:
                for brow in best_rows:
                    print(
                        f"N={N:4d} | BEST {brow['best_mode']:16s} | "
                        f"time={brow['time_median_s']:.4e} s | finite={brow['finite_C']}"
                    )

    if mpi_rank != 0:
        return

    save_rows(rows, out_path)
    print(f"Saved CSV: {out_path}")

    if not args.no_plot:
        make_plot(out_path, plot_path)

    if not args.skip_validation:
        save_validation_outputs(validation_payloads, args, plot_path)


if __name__ == "__main__":
    main()
