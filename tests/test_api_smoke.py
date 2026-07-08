#!/usr/bin/env python3

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from collision import C_MB, C_quantum, F_analytical, F_contact, _leggauss_torch
from kineticxgpu import ContactSelfCollisionOperator, make_log_grid


def check_kernel_backends_finite():
    device = "cpu"
    dtype = torch.float64
    m = torch.tensor(1.0, device=device, dtype=dtype)
    lam = torch.tensor(0.3, device=device, dtype=dtype)

    pi = torch.tensor([0.4, 1.2], device=device, dtype=dtype).view(-1, 1, 1)
    pn = torch.tensor([0.5, 1.4], device=device, dtype=dtype).view(1, -1, 1)
    pm = torch.tensor([0.6, 1.6], device=device, dtype=dtype).view(1, 1, -1)

    Ei = torch.sqrt(pi * pi + m * m)
    En = torch.sqrt(pn * pn + m * m)
    Em = torch.sqrt(pm * pm + m * m)
    mu2, w2 = _leggauss_torch(64, device, dtype)

    F_ana = F_analytical(pi, Ei, pn, En, pm, Em, m, lam)
    F_quad = F_contact(pi, Ei, pn, En, pm, Em, m, lam, mu2, w2)

    assert torch.isfinite(F_ana).all()
    assert torch.isfinite(F_quad).all()


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
    C_default, *_ = C_MB(f=f, q=q, **params)
    quad_params = {**params, "Ng": 64}
    C_quad, *_ = C_MB(f=f, q=q, kernel_backend="quadrature", **quad_params)
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
    assert torch.isfinite(C_ref).all()
    assert torch.isfinite(C_quad).all()
    torch.testing.assert_close(C_default, C_ref, rtol=0.0, atol=0.0)
    torch.testing.assert_close(C_api, C_ref, rtol=1e-12, atol=1e-30)
    scale = torch.clamp(torch.max(torch.abs(C_ref)), min=1e-300)
    rel_max = torch.max(torch.abs(C_quad - C_ref)) / scale
    assert rel_max < 5e-1


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
    C_quad, *_ = C_quantum(f=f, q=q, kernel_backend="quadrature", **params)
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
    assert torch.isfinite(C_ref).all()
    assert torch.isfinite(C_quad).all()
    torch.testing.assert_close(C_api, C_ref, rtol=1e-12, atol=1e-30)


if __name__ == "__main__":
    check_kernel_backends_finite()
    compare_classical()
    compare_quantum()
    print("API smoke test passed.")
