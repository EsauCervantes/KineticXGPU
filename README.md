# KineticXGPU  
### GPU-Accelerated Boltzmann Collision Operator with Neural Surrogate

KineticXGPU is a PyTorch-based implementation of a Boltzmann collision operator designed for high-performance kinetic simulations. The project focuses on:

- GPU acceleration of collision term evaluation  
- Benchmarking CPU vs GPU performance  
- Studying relaxation toward Maxwell–Boltzmann equilibrium
  
This repository is part of ongoing work on accelerating kinetic simulations relevant to non-equilibrium dynamics in the early universe.

---

## Motivation

The collision term is typically the computational bottleneck in Boltzmann equation solvers. The Boltzmann equation has the structure $\mathrm{d}f/\mathrm{d}t = C[f]$, where C[f] is an integral operator acting over f. For self-scattering collisions and after discretization, the operator takes the form

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
