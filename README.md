# KineticXGPU  
### GPU-Accelerated Boltzmann Collision Operator with Neural Surrogate (WIP)

KineticXGPU is a PyTorch-based implementation of a Boltzmann collision operator designed for high-performance kinetic simulations. The project focuses on:

- GPU acceleration of collision term evaluation  
- Benchmarking CPU vs GPU performance  
- Studying relaxation toward Maxwell–Boltzmann equilibrium  
- Early-stage neural surrogate modeling of the collision operator  

This repository is part of ongoing work on accelerating kinetic simulations relevant to non-equilibrium dynamics and freeze-in studies.

---

## Motivation

The collision term is typically the computational bottleneck in Boltzmann equation solvers.

This project explores:

- Efficient tensorized implementations in PyTorch  
- GPU acceleration strategies (CUDA backend)  
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

## Repository Structure

```
KineticXGPU/
│
├── src/                # Core collision operator and utilities
│   ├── collision_operator.py
│   ├── diagnostics.py
│   └── benchmark.py
│
├── notebooks/          # Experiments and exploratory analysis
│   ├── 01_benchmark.ipynb
│   └── 02_thermalization.ipynb
│
├── files/              # Configuration files / constants / test inputs
│
├── results/            # Generated plots and benchmark outputs
│   ├── benchmarks/
│   └── thermalization/
│
└── README.md
```

---

## Installation

```bash
git clone https://github.com/yourname/KineticXGPU.git
cd KineticXGPU

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.9  
- PyTorch (CUDA-enabled for GPU acceleration)  
- NumPy  
- Matplotlib  

---

## Quickstart

Run CPU benchmark:

```bash
python src/benchmark.py --device cpu
```

Run GPU benchmark:

```bash
python src/benchmark.py --device cuda
```

Run thermalization diagnostics:

```bash
python src/diagnostics.py
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

## Roadmap

- [ ] Complete GPU benchmarking across grid sizes  
- [ ] Add reproducible benchmark logging  
- [ ] Implement neural surrogate training pipeline  
- [ ] Add conservation diagnostics (number / energy)  
- [ ] Mixed precision experiments  
- [ ] Add CI tests  

---

## Example CV Description

> Built KineticXGPU, a PyTorch-based GPU-accelerated Boltzmann collision operator framework; achieved X× speedup over CPU baseline and implemented diagnostics for Maxwell–Boltzmann thermalization. Currently developing neural surrogate operator for scalable kinetic simulations.

---

## License

MIT License
