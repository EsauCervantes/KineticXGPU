#!/usr/bin/env python3

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from collision import C_MB, C_quantum
from kineticxgpu import ContactSelfCollisionOperator, make_log_grid


def compare_classical():
    q = make_log_grid(1e-3, 3.0, 12, dtype=torch.float64)
    f = 1e-7 * torch.exp(-q)
    params = {
        "a": 1.3,
        "m": 0.5,
        "lam": 2.0e-3,
        "Ng": 4,
        "batch_size": 4,
        "return_diagnostics": False,
        "enforce_self_projection": True,
    }

    C_ref, *_ = C_MB(f=f, q=q, **params)
    op = ContactSelfCollisionOperator(
        q=q,
        mass=params["m"],
        coupling=params["lam"],
        statistics="classical",
        Ng=params["Ng"],
        batch_size=params["batch_size"],
        enforce_self_projection=params["enforce_self_projection"],
    )
    C_api, *_ = op.evaluate(f, a=params["a"])
    torch.testing.assert_close(C_api, C_ref, rtol=1e-12, atol=1e-30)


def compare_quantum():
    q = make_log_grid(1e-3, 3.0, 12, dtype=torch.float64)
    f = 1e-7 * torch.exp(-q)
    params = {
        "a": 1.3,
        "m": 0.5,
        "lam": 2.0e-3,
        "Ng": 4,
        "batch_size": 4,
        "return_diagnostics": False,
        "enforce_self_projection": True,
        "statistics": "boson",
    }

    C_ref, *_ = C_quantum(f=f, q=q, **params)
    op = ContactSelfCollisionOperator(
        q=q,
        mass=params["m"],
        coupling=params["lam"],
        statistics=params["statistics"],
        Ng=params["Ng"],
        batch_size=params["batch_size"],
        enforce_self_projection=params["enforce_self_projection"],
    )
    C_api, *_ = op.evaluate(f, a=params["a"])
    torch.testing.assert_close(C_api, C_ref, rtol=1e-12, atol=1e-30)


if __name__ == "__main__":
    compare_classical()
    compare_quantum()
    print("API smoke test passed.")
