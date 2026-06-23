# KineticXGPU

GPU-accelerated Boltzmann solvers for isotropic dark-sector phase-space
distributions with elastic 2 -> 2 self-collisions.

The core implementation is written in PyTorch. It evaluates the self-collision
operator on CPU or CUDA devices and includes the hybrid freeze-in/self-scattering
solver used for the benchmark runs in the paper.

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

## Installation

Create an environment with Python 3.10 or newer, then install the Python
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA runs, install a PyTorch build compatible with your CUDA driver. See the
official PyTorch installation selector if the default `pip install torch` does
not provide CUDA support on your system.

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
| CPU | Intel Core i7-10750H @ 2.60 GHz, 6 cores / 12 threads |
| GPU | NVIDIA Quadro T2000 Mobile / Max-Q, 4 GB memory |
| CUDA version | 12.2 |

## Notes

- `results/` contains generated data and plots and is ignored by Git, except
  for the retained BEST-comparison CSV.
- The small `quickstart` config is intended as a smoke test, not as the final
  physics scan.
- The current fBE solver path uses `C_MB` by default. Set
  `collision.statistics` to `boson` or `fermion` to use `C_quantum`.
