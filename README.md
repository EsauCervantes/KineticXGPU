# KineticXGPU  
### GPU-Accelerated Boltzmann Collision Operator with Neural Surrogate

KineticXGPU is a PyTorch-based implementation of a Boltzmann collision operator designed for high-performance kinetic simulations. The project focuses on:

- GPU acceleration of collision term evaluation  
- Benchmarking CPU vs GPU performance  
- Studying relaxation toward Maxwell–Boltzmann equilibrium
  
This repository is part of ongoing work on accelerating kinetic simulations relevant to non-equilibrium dynamics in the early universe.

---

## Motivation
The collision term is typically the computational bottleneck in Boltzmann equation solvers. The Boltzmann equation has the structure

$$
\frac{\mathrm{d}f}{\mathrm{d}t} = C[f],
$$

where $C[f]$ is an integral operator acting on the distribution function $f$. For self-scattering collisions, after discretizing the momentum grid as $f_i \equiv f(p_i)$, the self-scattering collision operator can be written schematically as

$$
C_{\mathrm{self}}[f_i]
\simeq
\frac{(\Delta p)^2}{2 g_\chi}
\sum_n \sum_m
F(p_n, p_m)
\left[
f_n f_m - f_i \widetilde{f}_j
\right].
$$

Here $\Delta p$ is the momentum-grid spacing, $g_\chi$ is the number of internal degrees of freedom, and $\widetilde{f}_j$ denotes the distribution evaluated at the momentum fixed by momentum conservation. The function $F(p_n,p_m)$, depending on the physical setting and the approximations used, is a one- or two-dimensional integral. The double sum above can be optimized with GPU tensor operations, which is precisely the aim of the code.

This project explores:

- Efficient tensorized implementations in PyTorch  
- GPU acceleration strategies  
- Operator approximation using neural networks  
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
```

Explore notebooks:

```
notebooks/01_benchmark.ipynb
notebooks/02_thermalization.ipynb
```

---

## Benchmarking Methodology

Performance is measured as:

- Wall-clock time per collision operator evaluation  
- Median over multiple runs  
- Explicit CUDA synchronization when benchmarking GPU  
- Fixed grid resolution and batch size  

Example benchmark table (illustrative — to be updated):

| Grid Size | CPU (ms) | GPU (ms) | Speedup |
|------------|----------|----------|---------|
| 1024       | TBD      | TBD      | TBD     |
| 4096       | TBD      | TBD      | TBD     |

Hardware details (to be updated):

- GPU: TBD  
- CUDA version: TBD  
- Precision: float32  

---

## Thermalization Test

A non-equilibrium initial distribution is evolved under the collision operator to test relaxation toward a Maxwell–Boltzmann distribution.

Diagnostics include:

- Convergence of distribution shape  
- Moment evolution (e.g. ⟨p²⟩, ⟨p⁴⟩)  
- Relative deviation from analytic Maxwell–Boltzmann form  

Output plots are saved under:

```
results/thermalization/
```

---

## Neural Surrogate (Work in Progress)

Planned extensions:

- Train neural network surrogate for \( C[f] \)  
- Compare surrogate vs exact operator runtime  
- Error diagnostics on distribution moments  
- Investigate operator-learning approaches (MLP / DeepONet / FNO)  

---
