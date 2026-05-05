# KineticXGPU  
### GPU-Accelerated Boltzmann Collision Operator with Neural Surrogate

KineticXGPU is a PyTorch-based implementation of a Boltzmann collision operator designed for high-performance kinetic simulations. The project focuses on:

- GPU acceleration of collision term evaluation  
- Benchmarking CPU vs GPU performance  
- Studying relaxation toward Maxwell–Boltzmann equilibrium
  
This repository is part of ongoing work on accelerating kinetic simulations relevant to non-equilibrium dynamics in the early universe.

---

## Motivation

The collision term is typically the computational bottleneck in Boltzmann equation solvers. The Boltzmann equation has the structure \( df/dt = C[f] \), where \( C[f] \) is an integral operator acting on the distribution function \( f \). For self-scattering collisions, after discretizing the momentum grid as \( f_i = f(p_i) \), the self-scattering collision operator can be written schematically as a double sum over \( f \), resembling a bilinear form for \(2 \to 2\) self-scattering. This structure can be optimized using GPU tensor operations, which is the aim of the code.

This project explores:

- Efficient tensorized implementations in PyTorch  
- GPU acceleration strategies  
- Performance benchmarking and scalability analysis  

The objective is to enable faster iterative studies of kinetic and thermalization processes.

---

## Features

- PyTorch implementation of a discretized collision operator \( C[f] \)  
- CPU and GPU execution modes  
- Timing benchmarks and speedup measurements  
- Diagnostics for thermalization toward Maxwell–Boltzmann distributions  
- Modular structure for future neural surrogate integration  

---

## Notebooks

Explore notebooks:

```text
notebooks/01_benchmark.ipynb
notebooks/02_thermalization.ipynb
```

---

## Benchmarking Methodology

Performance is measured as:

- Wall-clock time per collision-operator evaluation  
- Median over multiple runs  
- Explicit CUDA synchronization when benchmarking GPU  
- Fixed angular quadrature and batch size  

The table below summarizes the benchmark results generated from the CSV files in
`results/CPU-GPU_benchmarks/`. Each entry reports the average, over the three
test distributions (`MB_T_eq_m`, `two_bump`, and `hot_tail`), of the median
runtime for one dense self-scattering collision-operator evaluation.

Benchmark settings:

| Setting | Value |
|:--|:--|
| Precision | `float32` |
| Angular quadrature | \(N_g = 12\) |
| Batch size | 16 |
| Momentum range | \(q/m \in [10^{-3}, 10^{2}]\) |
| Self-coupling | \(\lambda = 1\) |
| Conservation projection | Enabled |

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

The code also contains a conservative deposition variant of the self-scattering
operator for tests where exact discrete number and energy conservation are the
primary concern.

Hardware details for this benchmark:

| Component | Specification |
|:--|:--|
| CPU | Intel® Core™ i7-10750H CPU @ 2.60GHz, 6 cores / 12 threads |
| GPU | NVIDIA Quadro T2000 Mobile / Max-Q |
| CUDA version | 12.2 |

---

## Thermalization Test

A non-equilibrium initial distribution is evolved under the collision operator to test relaxation toward a Maxwell–Boltzmann distribution.

Diagnostics include:

- Convergence of distribution shape  
- Moment evolution, e.g. \( \langle p^2 \rangle \), \( \langle p^4 \rangle \)  
- Relative deviation from analytic Maxwell–Boltzmann form  

Output plots are saved under:

```text
results/thermalization/
```

---

## Neural Surrogate — Work in Progress

Planned extensions:

- Train neural network surrogate for \( C[f] \)  
- Compare surrogate vs exact operator runtime  
- Error diagnostics on distribution moments  
- Investigate operator-learning approaches, such as MLP, DeepONet, and FNO  

---
