#!/usr/bin/env python3

import torch

from kineticxgpu import ContactSelfCollisionOperator, make_log_grid


def main():
    q = make_log_grid(1e-3, 10.0, 32, dtype=torch.float64)
    f = 1e-6 * torch.exp(-q)

    collision = ContactSelfCollisionOperator(
        q=q,
        mass=1.0,
        coupling=1e-3,
        statistics="classical",
        Ng=8,
        batch_size=8,
    )

    C, E, p, dp = collision.evaluate(f, a=1.0)
    print(f"C shape: {tuple(C.shape)}")
    print(f"max |C|: {torch.max(torch.abs(C)).item():.6e}")


if __name__ == "__main__":
    main()
