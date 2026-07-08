"""Thin wrappers around the existing KineticXGPU numerical operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from collision import C_MB, C_quantum, resolve_contact_kernel_backend
from grid_log import make_log_q_grid

Statistics = Literal["classical", "boson", "fermion"]
KernelBackend = Literal["analytic", "quadrature"]


def _torch_dtype(dtype):
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    key = str(dtype).lower()
    if key in ("float32", "torch.float32"):
        return torch.float32
    if key in ("float64", "torch.float64"):
        return torch.float64
    raise ValueError("dtype must be float32 or float64.")


def make_log_grid(q_min, q_max, n, *, device=None, dtype=torch.float64, base=10.0):
    """Return a logarithmic comoving momentum grid."""
    q, *_ = make_log_q_grid(
        q_min=q_min,
        q_max=q_max,
        N=n,
        device=device,
        dtype=_torch_dtype(dtype),
        base=base,
    )
    return q


@dataclass
class ContactSelfCollisionOperator:
    """Callable contact self-collision operator.

    This class does not reimplement the collision integral. It only stores
    parameters and forwards calls to the existing ``C_MB`` or ``C_quantum``
    implementation.
    """

    q: object
    mass: float
    coupling: float
    statistics: Statistics = "classical"
    kernel_backend: KernelBackend = "analytic"
    Ng: int = 16
    batch_size: int = 16
    enforce_self_projection: bool = True
    device: object | None = None
    dtype: object | None = torch.float64

    def __post_init__(self):
        dtype = _torch_dtype(self.dtype)
        self.q = torch.as_tensor(self.q, device=self.device, dtype=dtype)
        self.dtype = self.q.dtype
        self.device = self.q.device

        stat = str(self.statistics).strip().lower().replace("-", "_")
        aliases = {
            "mb": "classical",
            "maxwell_boltzmann": "classical",
            "maxwell": "classical",
            "bose": "boson",
            "be": "boson",
            "bose_einstein": "boson",
            "fermi": "fermion",
            "fd": "fermion",
            "fermi_dirac": "fermion",
        }
        self.statistics = aliases.get(stat, stat)
        if self.statistics not in ("classical", "boson", "fermion"):
            raise ValueError("statistics must be classical, boson, or fermion.")
        self.kernel_backend = resolve_contact_kernel_backend(self.kernel_backend)[0]

    def evaluate(self, f, *, a=1.0, return_diagnostics=False):
        """Evaluate the self-collision term for a distribution ``f``."""
        f_t = torch.as_tensor(f, device=self.device, dtype=self.dtype)
        if self.statistics == "classical":
            return C_MB(
                f=f_t,
                a=a,
                q=self.q,
                m=self.mass,
                lam=self.coupling,
                Ng=self.Ng,
                batch_size=self.batch_size,
                return_diagnostics=return_diagnostics,
                enforce_self_projection=self.enforce_self_projection,
                kernel_backend=self.kernel_backend,
            )

        return C_quantum(
            f=f_t,
            a=a,
            q=self.q,
            m=self.mass,
            lam=self.coupling,
            Ng=self.Ng,
            batch_size=self.batch_size,
            return_diagnostics=return_diagnostics,
            enforce_self_projection=self.enforce_self_projection,
            statistics=self.statistics,
            kernel_backend=self.kernel_backend,
        )
