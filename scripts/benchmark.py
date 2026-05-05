# benchmark.py

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import torch

# ============================================================
# Paths
# ============================================================

THIS_FILE = Path(__file__).resolve()
SCRIPTS_DIR = THIS_FILE.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if not SRC_DIR.is_dir():
    raise FileNotFoundError(f"Expected source directory not found: {SRC_DIR}")

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from grid_log import make_log_q_grid
from collision import (
    C_self_torch_logq,
    C_self_torch_logq_conservative_scatter,
)


RESULTS_DIR = PROJECT_ROOT / "results"
BENCHMARK_DIR = RESULTS_DIR / "CPU-GPU_benchmarks"

BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Utilities
# ============================================================

def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def make_mb_shape(q, a, m, T, z=1.0):
    p = q / a
    E = torch.sqrt(p * p + m * m)
    return z * torch.exp(-E / T)


def make_two_bump_shape(
    q,
    q1,
    q2,
    sigma1=0.18,
    sigma2=0.22,
    A1=1.0,
    A2=0.6,
):
    logq = torch.log(q)

    bump1 = A1 * torch.exp(
        -0.5 * ((logq - math.log(q1)) / sigma1) ** 2
    )

    bump2 = A2 * torch.exp(
        -0.5 * ((logq - math.log(q2)) / sigma2) ** 2
    )

    return bump1 + bump2


def make_hot_tail_shape(q, a, m, T, q_tail, sigma=0.25, eps=2.0):
    f_mb = make_mb_shape(q=q, a=a, m=m, T=T, z=1.0)
    logq = torch.log(q)

    tail = eps * torch.exp(
        -0.5 * ((logq - math.log(q_tail)) / sigma) ** 2
    )

    return f_mb * (1.0 + tail)


def tensor_to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu())
    return float(x)


def resolve_benchmark_output(out_name):
    out_path = Path(out_name)

    if out_path.is_absolute():
        return out_path

    if out_path.parent == Path("."):
        return BENCHMARK_DIR / out_path

    return PROJECT_ROOT / out_path


# ============================================================
# Adaptive batch-size logic
# ============================================================

def rule_based_batch_size(N, device_type):
    if device_type == "cpu":
        if N <= 128:
            return 32
        if N <= 256:
            return 16
        if N <= 384:
            return 8
        return 4

    if device_type == "cuda":
        if N <= 96:
            return 64
        if N <= 160:
            return 32
        if N <= 256:
            return 16
        if N <= 384:
            return 8
        return 4


def estimate_bytes_per_batch(
    *,
    N,
    batch_size,
    dtype,
    operator_name,
    Ng,
    safety_tensor_count=None,
):
    """
    Rough memory estimate for the temporary tensors inside one collision batch.

    Most large intermediates scale as

        batch_size x N x N

    Some angular-kernel intermediates also involve Ng x batch_size x N x N,
    but many are reduced quickly. This estimate is intentionally conservative.

    It is not exact. It is a guardrail.
    """
    bytes_per_float = torch.tensor([], dtype=dtype).element_size()

    # Approximate number of B x N x N tensors alive at peak.
    # Conservative operator has extra scatter/indexing intermediates.
    if safety_tensor_count is not None:
        n_tensors = safety_tensor_count
    else:
        if operator_name == "conservative_scatter":
            n_tensors = 24
        else:
            n_tensors = 18

    # Angular part can temporarily be heavier. Add an Ng-dependent allowance.
    # This is deliberately cautious.
    angular_factor = max(1.0, 0.35 * Ng)

    effective_tensors = n_tensors + angular_factor

    return int(effective_tensors * batch_size * N * N * bytes_per_float)


def memory_aware_batch_size_cuda(
    *,
    N,
    dtype,
    operator_name,
    Ng,
    memory_fraction=0.60,
    max_batch_size=128,
    min_batch_size=1,
    safety_tensor_count=None,
):
    """
    Pick the largest batch size that approximately fits in available CUDA memory.

    Uses torch.cuda.mem_get_info(), then applies a safety fraction.

    This is approximate because PyTorch/CUDA allocation and temporary tensor
    lifetimes are not exactly captured by the simple estimate.
    """
    free_bytes, total_bytes = torch.cuda.mem_get_info()

    usable_bytes = int(memory_fraction * free_bytes)

    candidates = []
    b = min(max_batch_size, N)

    while b >= min_batch_size:
        candidates.append(b)
        b //= 2

    if min_batch_size not in candidates:
        candidates.append(min_batch_size)

    for b in candidates:
        needed = estimate_bytes_per_batch(
            N=N,
            batch_size=b,
            dtype=dtype,
            operator_name=operator_name,
            Ng=Ng,
            safety_tensor_count=safety_tensor_count,
        )

        if needed <= usable_bytes:
            return b, needed, free_bytes, total_bytes

    return min_batch_size, estimate_bytes_per_batch(
        N=N,
        batch_size=min_batch_size,
        dtype=dtype,
        operator_name=operator_name,
        Ng=Ng,
        safety_tensor_count=safety_tensor_count,
    ), free_bytes, total_bytes


def choose_batch_size(
    *,
    N,
    device,
    dtype,
    operator_name,
    Ng,
    fixed_batch_size,
    auto_batch_size,
    memory_aware_batch_size,
    memory_fraction,
    max_batch_size,
    safety_tensor_count,
):
    """
    Choose batch size according to user options.
    """
    if memory_aware_batch_size:
        if device.type != "cuda":
            # CPU memory-aware estimate is less reliable and not worth
            # complicating here. Use rule-based fallback.
            b = rule_based_batch_size(N, device.type)
            return {
                "batch_size": b,
                "batch_mode": "rule_based_cpu_fallback",
                "estimated_batch_bytes": None,
                "cuda_free_bytes": None,
                "cuda_total_bytes": None,
            }

        b, needed, free_bytes, total_bytes = memory_aware_batch_size_cuda(
            N=N,
            dtype=dtype,
            operator_name=operator_name,
            Ng=Ng,
            memory_fraction=memory_fraction,
            max_batch_size=max_batch_size,
            min_batch_size=1,
            safety_tensor_count=safety_tensor_count,
        )

        return {
            "batch_size": b,
            "batch_mode": "memory_aware_cuda",
            "estimated_batch_bytes": needed,
            "cuda_free_bytes": free_bytes,
            "cuda_total_bytes": total_bytes,
        }

    if auto_batch_size:
        b = rule_based_batch_size(N, device.type)
        return {
            "batch_size": b,
            "batch_mode": "rule_based",
            "estimated_batch_bytes": None,
            "cuda_free_bytes": None,
            "cuda_total_bytes": None,
        }

    return {
        "batch_size": fixed_batch_size,
        "batch_mode": "fixed",
        "estimated_batch_bytes": None,
        "cuda_free_bytes": None,
        "cuda_total_bytes": None,
    }


# ============================================================
# Benchmark core
# ============================================================

def run_one_operator(
    *,
    op_name,
    op,
    f,
    a,
    q,
    logq0,
    dlogq,
    log_space,
    m,
    lam,
    Ng,
    batch_size,
    repeats,
    warmup,
    device,
    apply_conservation_projection,
):
    # Warmup
    for _ in range(warmup):
        _ = op(
            f=f,
            a=a,
            q=q,
            logq0=logq0,
            dlogq=dlogq,
            log_space=log_space,
            m=m,
            lam=lam,
            Ng=Ng,
            batch_size=batch_size,
            use_cbe_shape=False,
            apply_conservation_projection=apply_conservation_projection,
            return_diagnostics=True,
        )
        synchronize(device)

    times = []
    last_diag = None
    last_C = None

    for _ in range(repeats):
        synchronize(device)
        t0 = time.perf_counter()

        C, E, p, dp, diag = op(
            f=f,
            a=a,
            q=q,
            logq0=logq0,
            dlogq=dlogq,
            log_space=log_space,
            m=m,
            lam=lam,
            Ng=Ng,
            batch_size=batch_size,
            use_cbe_shape=False,
            apply_conservation_projection=apply_conservation_projection,
            return_diagnostics=True,
        )

        synchronize(device)
        t1 = time.perf_counter()

        times.append(t1 - t0)
        last_diag = diag
        last_C = C

    times_sorted = sorted(times)
    median_time = times_sorted[len(times_sorted) // 2]

    row = {
        "operator": op_name,
        "time_median_s": median_time,
        "time_min_s": min(times),
        "time_max_s": max(times),
        "finite_C": bool(torch.isfinite(last_C).all().item()),
        "apply_conservation_projection": bool(apply_conservation_projection),
    }

    for key, value in last_diag.items():
        row[key] = tensor_to_float(value)

    return row


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark isolated self-scattering collision operators."
    )

    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")

    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)

    parser.add_argument("--Ng", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument(
        "--auto-batch-size",
        action="store_true",
        help="Choose batch_size by a simple rule depending on N_grid and device.",
    )

    parser.add_argument(
        "--memory-aware-batch-size",
        action="store_true",
        help=(
            "On CUDA, choose the largest approximate batch_size fitting in "
            "available GPU memory. On CPU, falls back to rule-based batch size."
        ),
    )

    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.60,
        help="Fraction of currently free CUDA memory allowed for one batch estimate.",
    )

    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=128,
        help="Maximum batch size allowed in adaptive/memory-aware modes.",
    )

    parser.add_argument(
        "--safety-tensor-count",
        type=int,
        default=None,
        help=(
            "Override the estimated number of large B x N x N tensors used in "
            "the memory estimate. Leave unset for operator-dependent defaults."
        ),
    )

    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--m", type=float, default=1.0)
    parser.add_argument("--a", type=float, default=1.0)

    parser.add_argument(
        "--q-min-factor",
        type=float,
        default=1e-3,
        help="Set q_min = q_min_factor * m.",
    )

    parser.add_argument(
        "--q-max-factor",
        type=float,
        default=1e2,
        help="Set q_max = q_max_factor * m.",
    )

    parser.add_argument(
        "--N-list",
        type=int,
        nargs="+",
        default=[32, 48, 64, 96, 128],
        help="List of grid sizes N_grid to benchmark.",
    )

    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output CSV filename. If omitted, a default filename is created "
            "inside results/CPU-GPU_benchmarks."
        ),
    )

    parser.add_argument(
        "--apply-conservation-projection",
        action="store_true",
        help="Apply conservation projection where supported.",
    )

    args = parser.parse_args()

    if args.auto_batch_size and args.memory_aware_batch_size:
        raise ValueError("Use only one of --auto-batch-size or --memory-aware-batch-size.")

    if args.q_min_factor <= 0.0:
        raise ValueError("--q-min-factor must be positive.")

    if args.q_max_factor <= args.q_min_factor:
        raise ValueError("--q-max-factor must be larger than --q-min-factor.")

    if args.memory_fraction <= 0.0 or args.memory_fraction > 1.0:
        raise ValueError("--memory-fraction must be in the interval (0, 1].")

    if args.max_batch_size <= 0:
        raise ValueError("--max-batch-size must be positive.")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    if args.out is None:
        projection_tag = (
            "projection_on"
            if args.apply_conservation_projection
            else "projection_off"
        )

        qrange_tag = (
            f"qmin{args.q_min_factor:.0e}_"
            f"qmax{args.q_max_factor:.0e}"
        )

        if args.memory_aware_batch_size:
            batch_tag = f"batch_memoryaware_max{args.max_batch_size}"
        elif args.auto_batch_size:
            batch_tag = "batch_auto"
        else:
            batch_tag = f"batch{args.batch_size}"

        out_name = (
            f"collision_benchmark_{args.device}_{args.dtype}_"
            f"Ng{args.Ng}_{batch_tag}_"
            f"{qrange_tag}_{projection_tag}.csv"
        )
    else:
        out_name = args.out

    out_path = resolve_benchmark_output(out_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    operators = {
        "semi_linearized": C_self_torch_logq,
        "conservative_scatter": C_self_torch_logq_conservative_scatter,
    }

    rows = []

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Saving benchmark CSV to: {out_path}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"q_min_factor: {args.q_min_factor}")
    print(f"q_max_factor: {args.q_max_factor}")

    if args.memory_aware_batch_size:
        print("Batch mode: memory-aware")
        print(f"memory_fraction: {args.memory_fraction}")
        print(f"max_batch_size: {args.max_batch_size}")
    elif args.auto_batch_size:
        print("Batch mode: rule-based auto")
    else:
        print(f"Batch mode: fixed batch_size={args.batch_size}")

    for N in args.N_list:
        q_min = args.q_min_factor * args.m
        q_max = args.q_max_factor * args.m

        q, logq0, dlogq, log_space = make_log_q_grid(
            q_min,
            q_max,
            N,
            device=device,
            dtype=dtype,
            base=10.0,
        )

        a = torch.tensor(args.a, device=device, dtype=dtype)
        m = torch.tensor(args.m, device=device, dtype=dtype)
        T = torch.tensor(args.m, device=device, dtype=dtype)

        shapes = {
            "MB_T_eq_m": make_mb_shape(
                q=q,
                a=a,
                m=m,
                T=T,
                z=1.0,
            ),
            "two_bump": make_two_bump_shape(
                q=q,
                q1=0.3 * args.m,
                q2=8.0 * args.m,
                A1=1.0,
                A2=0.5,
            ),
            "hot_tail": make_hot_tail_shape(
                q=q,
                a=a,
                m=m,
                T=T,
                q_tail=10.0 * args.m,
            ),
        }

        for shape_name, f in shapes.items():
            # Avoid exact zeros, especially because the conservative operator
            # may use interpolation involving f_tilde.
            f = torch.clamp(f, min=torch.finfo(dtype).tiny)

            for op_name, op in operators.items():
                batch_info = choose_batch_size(
                    N=N,
                    device=device,
                    dtype=dtype,
                    operator_name=op_name,
                    Ng=args.Ng,
                    fixed_batch_size=args.batch_size,
                    auto_batch_size=args.auto_batch_size,
                    memory_aware_batch_size=args.memory_aware_batch_size,
                    memory_fraction=args.memory_fraction,
                    max_batch_size=args.max_batch_size,
                    safety_tensor_count=args.safety_tensor_count,
                )

                batch_size_this = int(batch_info["batch_size"])

                row = run_one_operator(
                    op_name=op_name,
                    op=op,
                    f=f,
                    a=a,
                    q=q,
                    logq0=logq0,
                    dlogq=dlogq,
                    log_space=log_space,
                    m=m,
                    lam=args.lam,
                    Ng=args.Ng,
                    batch_size=batch_size_this,
                    repeats=args.repeats,
                    warmup=args.warmup,
                    device=device,
                    apply_conservation_projection=args.apply_conservation_projection,
                )

                row.update(
                    {
                        "device": args.device,
                        "dtype": args.dtype,
                        "N": N,
                        "shape": shape_name,
                        "Ng": args.Ng,
                        "batch_size": batch_size_this,
                        "batch_mode": batch_info["batch_mode"],
                        "estimated_batch_bytes": batch_info["estimated_batch_bytes"],
                        "cuda_free_bytes": batch_info["cuda_free_bytes"],
                        "cuda_total_bytes": batch_info["cuda_total_bytes"],
                        "memory_fraction": args.memory_fraction,
                        "max_batch_size": args.max_batch_size,
                        "lam": args.lam,
                        "m": args.m,
                        "a": args.a,
                        "q_min": q_min,
                        "q_max": q_max,
                        "q_min_factor": args.q_min_factor,
                        "q_max_factor": args.q_max_factor,
                    }
                )

                rows.append(row)

                estimated_mb = (
                    None
                    if row["estimated_batch_bytes"] is None
                    else row["estimated_batch_bytes"] / 1024**2
                )

                estimated_mb_str = (
                    "n/a"
                    if estimated_mb is None
                    else f"{estimated_mb:.1f} MB"
                )

                print(
                    f"N={N:4d} | "
                    f"batch={batch_size_this:3d} | "
                    f"mode={row['batch_mode']:20s} | "
                    f"est_mem={estimated_mb_str:>10s} | "
                    f"q=[{q_min:.1e}, {q_max:.1e}] | "
                    f"shape={shape_name:10s} | "
                    f"op={op_name:22s} | "
                    f"time={row['time_median_s']:.4e} s | "
                    f"rel_E={row.get('rel_energy', float('nan')):.3e} | "
                    f"rel_N={row.get('rel_number', float('nan')):.3e} | "
                    f"outside={row.get('outside_weight_fraction', float('nan')):.3e}"
                )

    fieldnames = sorted({key for row in rows for key in row.keys()})

    with out_path.open("w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved benchmark CSV:\n{out_path}")


if __name__ == "__main__":
    main()
