# KineticXGPU


KineticXGPU is a PyTorch-based solver for the cosmological Boltzmann equation, df/dt = C, at the phase-space level. The code follows an isotropic dark-sector distribution during freeze-in production and elastic 2 -> 2 self-scattering.

If dark matter is produced with a nonthermal momentum distribution, how does self-scattering relax it toward a Maxwell-Boltzmann, Bose-Einstein or Fermi-Dirac shape? Integrated number-density and temperature equations cannot answer this because the shape of the distribution has already been assumed, reducing the problem to a set of ODEs. Here, by contrast, the full distribution f(q,t) is evolved on a momentum grid.

The main numerical bottleneck is the self-scattering collision operator. After discretization, the operator becomes a large bilinear sum over momentum bins. The dense evaluation scales as O(Ng Ngrid^3), where Ng is the angular quadrature order and Ngrid is the number of momentum bins. This is slow on a CPU, but embarrassingly parallel and well suited for a GPU; i.e., a similar tensor structure that appears in machine-learning neural networks. KineticXGPU uses PyTorch to exploit this tensor/GPU infrastructure for a physics collision integral.

The repository includes the hybrid freeze-in/self-scattering solver used in the paper, a companion integrated Boltzmann-equation solver for comparison, scripts to reproduce the saved figures, and benchmark utilities for CPU/GPU and BEST comparisons.

## Installation

Create an environment with Python 3.10 or newer, then install the Python
dependencies and the local package:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For CUDA runs, install a PyTorch build compatible with your CUDA driver. See the
official PyTorch installation selector if the default `pip install torch` does
not provide CUDA support on your system.

## Library Usage

The repository can also be used as a small Python library for evaluating the
contact self-collision operator:

```python
import torch

from kineticxgpu import ContactSelfCollisionOperator, make_log_grid

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
```

The command-line wrapper reads an input `.npz` file containing arrays named
`f` and `q`, then writes the collision term to another `.npz` file:

```bash
kineticxgpu-collision \
  --input input_distribution.npz \
  --output collision_output.npz \
  --mass 1.0 \
  --coupling 1e-3 \
  --statistics classical \
  --Ng 8 \
  --batch-size 8
```

Run the API smoke test with:

```bash
python3 tests/test_api_smoke.py
```

## Quickstart

Run a small fBE example:

```bash
python3 scripts/run_solver.py fbe --config configs/quickstart.json
```

Run the corresponding cBE comparison:

```bash
python3 scripts/run_solver.py cbe --config configs/quickstart.json
```

The default config uses `"device": "auto"`, so CUDA is used when available and
CPU otherwise. fBE outputs are written under `results/runs/fBE/`, and the cBE
reference is written under `results/runs/cBE/`.

The evolution endpoints are set in the JSON through `physics.T_initial` in GeV
and `physics.x_final`, where `x=m_chi/T`. The default `T_initial` is 150 GeV;
lower it if you want to start later, i.e. at a larger `x_initial`.

## Paper Benchmark

The heavier paper settings are in:

```text
configs/paper_benchmark.json
```

Run the fBE scan with:

```bash
python3 scripts/run_solver.py fbe --config configs/paper_benchmark.json
```

Run the cBE reference solution with:

```bash
python3 scripts/run_solver.py cbe --config configs/paper_benchmark.json
```

The paper benchmark uses CUDA and float64 by default. Adjust `device`, `dtype`,
`grid.N`, `collision.Ng`, and `collision.batch_size` in the JSON file for your
hardware.

## Plotting

Plot saved quickstart runs:

```bash
python3 scripts/plot_figures.py \
  --runs-dir results/runs/fBE \
  --cbe-dir results/runs/cBE \
  --out-dir results/plots \
  --run-prefix quickstart
```

Use `--run-prefix paper` after running the paper benchmark:

```bash
python3 scripts/plot_figures.py \
  --runs-dir results/runs/fBE \
  --cbe-dir results/runs/cBE \
  --out-dir results/plots \
  --run-prefix paper
```

By default, the script writes PDF files named `evolution_lambda_<value>`,
`final_distributions`, `velocity_moments`, `temperature_moment`, `rates`, and
`abundance_Y_comparison`.

## Collision Operator Benchmark

Benchmark isolated self-collision calls:

```bash
python3 scripts/benchmark.py --device cpu --dtype float64 --N-list 32 48 64
```

For CUDA:

```bash
python3 scripts/benchmark.py --device cuda --dtype float32 --memory-aware-batch-size
```

CSV outputs are written to `results/CPU-GPU_benchmarks/`.

## BEST Comparison

BEST is not included in this repository. To run a new independent
implementation comparison, clone BEST separately and pass its local path. For a
small comparison run:

```bash
python3 scripts/benchmark_best_comparison.py \
  --best-dir /path/to/BEST \
  --device cuda \
  --dtype float64 \
  --N-list 32 48 64 96
```

The retained paper CSV used `Ng=18`, memory-aware CUDA batching, and CPU rows
for KineticXGPU:

```bash
python3 scripts/benchmark_best_comparison.py \
  --best-dir /path/to/BEST \
  --device cuda \
  --dtype float64 \
  --include-this-work-cpu \
  --memory-aware-batch-size \
  --N-list 32 48 64 96 \
  --Ng 18 \
  --warmup 20 \
  --repeats 20
```

The archived CSV also includes FP32 CUDA rows. The current command-line
interface runs one dtype at a time, so those rows require a separate run with
`--dtype float32` or the retained CSV below.

Use `--skip-best` to benchmark only the KineticXGPU operator.

The retained CSV
`results/CPU-GPU_benchmarks/best_comparison_boson_raw_N32_48_64_96_Nmu18_warm20_rep20_with_fp32gpu.csv`
contains the BEST-comparison data used for the paper.

The retained comparison was produced on:

| Component | Specification |
|:--|:--|
| System | Lenovo ThinkPad P15 Gen 1 |
| CPU | Intel Core i7-10750H @ 2.60 GHz, 6 cores / 12 threads |
| GPU | NVIDIA Quadro T2000 Mobile / Max-Q, 4 GB memory |
| System memory | 32 GB RAM |
| CUDA version | 12.2 |

## Repository Layout

```text
src/
  collision.py        self-collision operators, source terms, diagnostics
  solver.py           RK4, adaptive Heun, and hybrid fBE solver
  cBE_solver.py       coupled Boltzmann-equation comparison solver
  cosmology.py        cosmological background functions
  grid_log.py         logarithmic momentum-grid utilities
  thermodynamics.py   equilibrium and source-rate helpers

scripts/
  run_solver.py                 run the cBE and fBE solvers
  plot_figures.py               plot saved fBE/cBE runs
  benchmark.py                  isolated operator CPU/GPU benchmarks
  benchmark_best_comparison.py  optional comparison against a local BEST checkout

configs/
  quickstart.json       small CPU/GPU-auto run for testing the installation
  paper_benchmark.json  heavier settings used for the paper scan
```

Generated runs and plots are written under `results/` and ignored by Git. The
only result file kept in the repository is the BEST-comparison CSV used in the
paper.

## Notes

- `results/` contains generated data and plots and is ignored by Git, except
  for the retained BEST-comparison CSV.
- The small `quickstart` config is intended as a smoke test, not as the final
  physics scan.
- The current fBE solver path uses `C_MB` by default. Set
  `collision.statistics` to `boson` or `fermion` to use `C_quantum`.

## Citation

This code accompanies the paper **KineticXGPU: A Tensorized Collision Operator
for Dark-Sector Self-Scattering**:

```text
https://arxiv.org/abs/2607.00755
```

If you use KineticXGPU, please cite:

```bibtex
@article{Cervantes:2026fbk,
    author = "Cervantes, Esau",
    title = "{KineticXGPU: A Tensorized Collision Operator for Dark-Sector Self-Scattering}",
    eprint = "2607.00755",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "7",
    year = "2026"
}
```

## License

KineticXGPU is released under the [MIT License](LICENSE).
