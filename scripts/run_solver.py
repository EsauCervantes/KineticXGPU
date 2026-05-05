#!/usr/bin/env python3

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


THIS_FILE = Path(__file__).resolve()
SCRIPTS_DIR = THIS_FILE.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cBE_solver import solve_condensate_N_loga_quad, solve_free_in_loga_with_abundance
from collision import C_self_torch_logq, C_self_torch_logq_conservative_scatter
from cosmology import VariableGCosmology
from grid_log import make_log_q_grid
from solver import run_hybrid_FI_then_adaptive_self


SELF_OPERATORS = {
    "dense": C_self_torch_logq,
    "conservative": C_self_torch_logq_conservative_scatter,
}


def load_config(path):
    with Path(path).open("r") as f:
        return json.load(f)


def as_torch_dtype(name):
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError("dtype must be 'float32' or 'float64'.")


def choose_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return device


def config_get(config, section, key, default=None):
    return config.get(section, {}).get(key, default)


def make_cosmology(config, device=None, dtype=None):
    cosmo = VariableGCosmology()

    if config_get(config, "cosmology", "use_torch_tables", False):
        cosmo.build_torch_tables(device=device, dtype=dtype)

    return cosmo


def make_condensate(config, cosmo, af):
    physics = config["physics"]
    cbe = config.get("cbe", {})

    a0 = float(physics["a0"])
    n_init = float(physics["n_condensate_initial"])
    N_init = a0**3 * n_init

    _, _, build_condensate = solve_condensate_N_loga_quad(
        H_of_a=cosmo.H_of_a,
        N_init=N_init,
        ai=a0,
        af=af,
        n_eval=int(cbe.get("condensate_n_eval", 4000)),
        zero_rel=float(cbe.get("condensate_zero_rel", 1e-14)),
    )

    gamma = float(physics["gamma_condensate"])
    _, N_of_a, n_of_a = build_condensate(gamma)
    return N_of_a, n_of_a


def make_run_name(prefix, q, lam_self, Ng, batch_size, n_windows, rk4_steps_per_window, af):
    return (
        f"{prefix}_"
        f"N{q.numel()}_"
        f"lamself{lam_self:.1e}_"
        f"Ng{Ng}_"
        f"batch{batch_size}_"
        f"nw{n_windows}_"
        f"rk4spw{rk4_steps_per_window}_"
        f"af{af:.1e}"
    )


def make_common_metadata(config, q, lam_self=None, run_name=None):
    physics = config["physics"]
    collision = config.get("collision", {})
    hybrid = config.get("hybrid", {})
    heun = config.get("heun", {})

    metadata = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "a0": float(physics["a0"]),
        "af": float(physics["af"]),
        "N_grid": int(q.numel()),
        "qmin": float(q[0].detach().cpu()),
        "qmax": float(q[-1].detach().cpu()),
        "m_chi": float(physics["m_chi"]),
        "m_h": float(physics["m_h"]),
        "m_condensate": float(physics["m_condensate"]),
        "gamma_condensate": float(physics["gamma_condensate"]),
        "n_condensate_initial": float(physics["n_condensate_initial"]),
        "lambda_portal": float(physics["lambda_portal"]),
        "v_h": float(physics["v_h"]),
        "g_trilinear": float(physics["lambda_portal"]) * float(physics["v_h"]),
        "multiplicity_condensate": float(physics.get("multiplicity_condensate", 2.0)),
        "self_operator": collision.get("operator", "dense"),
        "Ng": int(collision.get("Ng", 12)),
        "batch_size": int(collision.get("batch_size", 16)),
        "gamma_over_H_on": float(hybrid.get("gamma_over_H_on", 0.1)),
        "n_windows": int(hybrid.get("n_windows", 400)),
        "rk4_steps_per_window": int(hybrid.get("rk4_steps_per_window", 1)),
        "rk4_store_every_steps": hybrid.get("rk4_store_every_steps", None),
        "heun_store_every_accepted": int(heun.get("store_every_accepted", 20)),
        "heun_print_every_accepted": int(heun.get("print_every_accepted", 30)),
    }

    if lam_self is not None:
        metadata["lambda_self"] = float(lam_self)

    return metadata


def run_fbe(config, save_override=None):
    device = choose_device(config.get("device", "auto"))
    dtype = as_torch_dtype(config.get("dtype", "float64"))
    save = bool(config.get("save", True)) if save_override is None else bool(save_override)

    physics = config["physics"]
    grid = config["grid"]
    collision = config.get("collision", {})
    hybrid = config.get("hybrid", {})
    heun = config.get("heun", {})
    stability = config.get("stability", {})

    cosmo = make_cosmology(config, device=device, dtype=dtype)
    H_of_a = cosmo.H_of_a_torch if config_get(config, "cosmology", "use_torch_tables", False) else cosmo.H_of_a
    T_of_a = cosmo.T_of_a_torch if config_get(config, "cosmology", "use_torch_tables", False) else cosmo.T_of_a

    _, nX_of_a = make_condensate(config, cosmo, af=float(physics["af"]))

    q, _, _, _ = make_log_q_grid(
        float(grid["q_min"]),
        float(grid["q_max"]),
        int(grid["N"]),
        device=device,
        dtype=dtype,
        base=10.0,
    )

    operator_name = collision.get("operator", "dense")
    if operator_name not in SELF_OPERATORS:
        raise ValueError(f"Unknown collision.operator={operator_name!r}. Use one of {sorted(SELF_OPERATORS)}.")

    C_self_operator = SELF_OPERATORS[operator_name]
    lam_values = collision.get("lambda_self_values")
    if lam_values is None:
        lam_values = [float(collision.get("lambda_self", 5e-4))]

    results = {}
    f0_base = torch.zeros_like(q, device=device, dtype=dtype)

    for lam_self in lam_values:
        lam_self = float(lam_self)
        batch_size = int(collision.get("batch_size", 16))
        Ng = int(collision.get("Ng", 12))
        n_windows = int(hybrid.get("n_windows", 400))
        rk4_steps_per_window = int(hybrid.get("rk4_steps_per_window", 1))
        af = float(physics["af"])

        run_name = config.get("run_name")
        if run_name is None:
            run_name = make_run_name(
                prefix=config.get("run_name_prefix", "hybrid_saved"),
                q=q,
                lam_self=lam_self,
                Ng=Ng,
                batch_size=batch_size,
                n_windows=n_windows,
                rk4_steps_per_window=rk4_steps_per_window,
                af=af,
            )

        metadata = make_common_metadata(config, q, lam_self=lam_self, run_name=run_name)

        print("=" * 80)
        print(f"Starting fBE run: {run_name}")
        print(f"device={device}, dtype={dtype}, operator={operator_name}, save={save}")
        print("=" * 80)

        traj = run_hybrid_FI_then_adaptive_self(
            f0=f0_base.clone(),
            a0=float(physics["a0"]),
            af=af,
            q=q,
            m_chi=float(physics["m_chi"]),
            H_of_a=H_of_a,
            T_of_a=T_of_a,
            m_h=float(physics["m_h"]),
            g_trilinear=float(physics["lambda_portal"]) * float(physics["v_h"]),
            nX_of_a=nX_of_a,
            Gamma_X=float(physics["gamma_condensate"]),
            mX=float(physics["m_condensate"]),
            lam_self=lam_self,
            C_self_operator=C_self_operator,
            batch_size=batch_size,
            Ng=Ng,
            n_windows=n_windows,
            gamma_over_H_on=float(hybrid.get("gamma_over_H_on", 0.1)),
            gamma_check_every_far=int(hybrid.get("gamma_check_every_far", 50)),
            gamma_check_every_mid=int(hybrid.get("gamma_check_every_mid", 20)),
            gamma_check_every_near=int(hybrid.get("gamma_check_every_near", 5)),
            rk4_steps_per_window=rk4_steps_per_window,
            rk4_store_every_steps=hybrid.get("rk4_store_every_steps", None),
            heun_du_init=float(heun.get("du_init", 1e-3)),
            heun_du_min=float(heun.get("du_min", 1e-6)),
            heun_du_max=float(heun.get("du_max", 0.1)),
            heun_rtol=float(heun.get("rtol", 1e-2)),
            heun_atol=float(heun.get("atol", 1e-14)),
            heun_safety=float(heun.get("safety", 0.95)),
            heun_store_every_accepted=int(heun.get("store_every_accepted", 200)),
            heun_print_every_accepted=int(heun.get("print_every_accepted", 50)),
            clip_negative=bool(stability.get("clip_negative", True)),
            clip_tol=float(stability.get("clip_tol", 0.0)),
            out_path_pt="trajectory.pt" if save else None,
            out_path_dat="trajectory.dat" if save else None,
            results_dir=config.get("results_dir", "results"),
            run_name=run_name if save else None,
            metadata=metadata if save else None,
        )

        f_final = traj["f_final"]
        print("-" * 80)
        print(f"Finished: {run_name}")
        print(f"finite={torch.isfinite(f_final).all().item()}")
        print(f"min={float(f_final.min().detach().cpu()):.6e}")
        print(f"max={float(f_final.max().detach().cpu()):.6e}")
        if save:
            print(f"Saved to: {config.get('results_dir', 'results')}/{run_name}/")

        results[run_name] = {
            "finite": bool(torch.isfinite(f_final).all().item()),
            "min": float(f_final.min().detach().cpu()),
            "max": float(f_final.max().detach().cpu()),
            "saved": save,
            "run_name": run_name,
        }

        del traj

    return results


def run_cbe(config, save_override=None):
    save = bool(config.get("save", True)) if save_override is None else bool(save_override)
    physics = config["physics"]
    cbe = config.get("cbe", {})

    cbe_config = dict(config)
    cbe_config["cosmology"] = dict(config.get("cosmology", {}))
    cbe_config["cosmology"]["use_torch_tables"] = False
    cosmo = make_cosmology(cbe_config)
    af = float(cbe.get("af", physics.get("af", 30000.0)))
    _, nX_of_a = make_condensate(config, cosmo, af=af)

    sol, Ns, Ts, Y_sol, Y_obs, ratio = solve_free_in_loga_with_abundance(
        cosmo=cosmo,
        ms=float(physics["m_chi"]),
        lhs=float(physics["lambda_portal"]),
        xi_inf=float(physics.get("xi_inf", 1e-4)),
        ai=float(physics["a0"]),
        af=af,
        mh=float(physics["m_h"]),
        v=float(physics["v_h"]),
        nX_of_a=nX_of_a,
        Gamma_X=float(physics["gamma_condensate"]),
        mX=float(physics["m_condensate"]),
        m_other=float(physics["m_chi"]),
        multiplicity_X=float(physics.get("multiplicity_condensate", 2.0)),
        method=cbe.get("method", "Radau"),
        rtol=float(cbe.get("rtol", 1e-6)),
        atol=float(cbe.get("atol", 1e-9)),
        max_step_u=float(cbe.get("max_step_u", 0.001)),
        a_match=cbe.get("a_match", None),
    )

    print("cBE result:")
    print(f"  Y_solver = {Y_sol:.6e}")
    print(f"  Y_obs    = {Y_obs:.6e}")
    print(f"  ratio    = {ratio:.6e}")

    if save:
        out_dir = PROJECT_ROOT / config.get("results_dir", "results") / config.get("run_name", "cbe_benchmark")
        out_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "mode": "cbe",
            "Y_solver": float(Y_sol),
            "Y_obs": float(Y_obs),
            "ratio": float(ratio),
            "af": af,
            "physics": physics,
            "cbe": cbe,
        }

        a_eval = np.logspace(np.log10(float(physics["a0"])), np.log10(af), 500)
        N_eval = np.array([Ns(a) for a in a_eval])
        T_eval = np.array([Ts(a) for a in a_eval])

        np.savez(out_dir / "cbe_solution.npz", a=a_eval, N=N_eval, T=T_eval)
        with (out_dir / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

        print(f"Saved to: {out_dir}")

    return sol, Ns, Ts, Y_sol, Y_obs, ratio


def main():
    parser = argparse.ArgumentParser(description="Run KineticXGPU cBE/fBE solvers from a JSON config.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    for mode in ("fbe", "cbe"):
        sub = subparsers.add_parser(mode)
        sub.add_argument("--config", default="configs/fbe_benchmark.json")
        save_group = sub.add_mutually_exclusive_group()
        save_group.add_argument("--save", action="store_true", default=None)
        save_group.add_argument("--no-save", action="store_false", dest="save", default=None)

    args = parser.parse_args()
    config = load_config(PROJECT_ROOT / args.config)

    if args.mode == "fbe":
        run_fbe(config, save_override=args.save)
    elif args.mode == "cbe":
        run_cbe(config, save_override=args.save)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
