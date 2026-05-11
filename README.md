# KineticXGPU
### GPU-Accelerated Boltzmann Collision Operator for Cosmological Kinetic Theory

**KineticXGPU achieves up to 9.5× GPU speedup** over CPU for evaluating discretized Boltzmann collision operators — enabling faster iterative studies of thermalization and non-equilibrium dynamics in the early universe.

Built with PyTorch. Designed for researchers working on dark matter at the phase-space level, or any problem where the collision term is the computational bottleneck.

---

## Benchmark Results

| Grid size \(N\) | CPU runtime [ms] | GPU runtime [ms] | GPU speedup |
|----------------:|-----------------:|-----------------:|------------:|
| 32              | 13.6             | 6.5              | 2.10×       |
| 48              | 44.5             | 10.2             | 4.36×       |
| 64              | 112.5            | 15.5             | 7.24×       |
| 80              | 190.7            | 27.8             | 6.86×       |
| 96              | 346.8            | 41.9             | 8.28×       |
| 112             | 565.1            | 66.5             | 8.50×       |
| 128             | 862.0            | 94.7             | 9.10×       |
| 160             | 1709.7           | 180.9            | 9.45×       |
| 192             | 3016.9           | 322.2            | 9.36×       |
| 224             | 4716.6           | 505.8            | 9.33×       |
| 256             | 7036.9           | 739.3            | 9.52×       |

Runtimes are wall-clock medians over multiple runs (float32, Ng=12 angular quadrature, batch size 16). Speedup grows with grid size, reaching ~9.5× at N=256.

**Hardware:**
| Component | Specification |
|:--|:--|
| CPU | Intel® Core™ i7-10750H @ 2.60GHz, 6 cores / 12 threads |
| GPU | NVIDIA Quadro T2000 Mobile / Max-Q |
| CUDA version | 12.2 |

---

## What This Solves

The collision term C[f] in df/dt = C[f] is typically the computational bottleneck in Boltzmann solvers. For 2→2 self-scattering, the discretized operator takes a bilinear form over momentum grids — a structure naturally suited to GPU tensor operations. KineticXGPU exploits this with PyTorch, making grid refinement studies that were previously slow practical to run iteratively.

---

## Features

- PyTorch implementation of a discretized collision operator C[f] for self-scattering (`collision.py`)
- CPU and GPU execution modes
- Timing benchmarks and speedup measurements
- Diagnostics for thermalization toward Maxwell–Boltzmann distributions
- Conservative deposition variant for exact discrete number and energy conservation
- Adaptive solver (`solver.py`). Different strategies are being explored right now to prevent the constant CPU-GPU overhead from time-step integration. 

---

## Notebooks

Explore results and diagnostics:

```text
notebooks/plots.ipynb
```

---

## Benchmarking Methodology

Performance is measured as:
- Wall-clock time per collision-operator evaluation
- Median over multiple runs
- Explicit CUDA synchronization when benchmarking GPU
- Fixed angular quadrature and batch size

Benchmark settings:
| Setting | Value |
|:--|:--|
| Precision | `float32` |
| Angular quadrature | Ng = 12 |
| Batch size | 16 |
| Momentum range | qmin=1e-3, qmax=1e2 |
| Self-coupling | lambda = 1 |
| Conservation projection | Enabled |
