"""Command line helpers for the thin KineticXGPU API."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .operators import ContactSelfCollisionOperator


def _dtype_from_name(name):
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError("dtype must be float32 or float64.")


def _device_from_name(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return device


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate the KineticXGPU contact self-collision operator."
    )
    parser.add_argument("--input", required=True, help="Input .npz with arrays f and q.")
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument("--a", type=float, default=1.0, help="Scale factor.")
    parser.add_argument("--mass", type=float, required=True, help="Particle mass.")
    parser.add_argument("--coupling", type=float, required=True, help="Contact coupling.")
    parser.add_argument(
        "--statistics",
        default="classical",
        choices=["classical", "boson", "fermion"],
    )
    parser.add_argument("--Ng", type=int, default=16, help="Angular quadrature order.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument(
        "--no-projection",
        action="store_true",
        help="Disable the self-collision moment projection.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    data = np.load(args.input)
    if "f" not in data or "q" not in data:
        raise ValueError("Input .npz must contain arrays named 'f' and 'q'.")

    device = _device_from_name(args.device)
    dtype = _dtype_from_name(args.dtype)
    operator = ContactSelfCollisionOperator(
        q=data["q"],
        mass=args.mass,
        coupling=args.coupling,
        statistics=args.statistics,
        Ng=args.Ng,
        batch_size=args.batch_size,
        enforce_self_projection=not args.no_projection,
        device=device,
        dtype=dtype,
    )

    C, E, p, dp = operator.evaluate(data["f"], a=args.a)
    metadata = {
        "a": args.a,
        "mass": args.mass,
        "coupling": args.coupling,
        "statistics": args.statistics,
        "Ng": args.Ng,
        "batch_size": args.batch_size,
        "device": str(device),
        "dtype": args.dtype,
        "enforce_self_projection": not args.no_projection,
    }

    np.savez(
        args.output,
        C=C.detach().cpu().numpy(),
        E=E.detach().cpu().numpy(),
        p=p.detach().cpu().numpy(),
        dp=dp.detach().cpu().numpy(),
        q=operator.q.detach().cpu().numpy(),
        metadata=json.dumps(metadata),
    )


if __name__ == "__main__":
    main()
