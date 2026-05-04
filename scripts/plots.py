# plots.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Plot style
# ============================================================

USE_TEX = True  # Set True only if a full LaTeX installation is available.

plt.rcParams.update(
    {
        "text.usetex": USE_TEX,
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "savefig.bbox": "tight",
    }
)


# ============================================================
# Paths
# ============================================================

THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parent
PROJECT_ROOT = SRC_DIR.parent

RESULTS_DIR = PROJECT_ROOT / "results"
BENCHMARK_DIR = RESULTS_DIR / "CPU-GPU_benchmarks"
PLOTS_DIR = RESULTS_DIR / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Labels
# ============================================================

OPERATOR_LABELS = {
    "semi_linearized": "projected dense",
    "semi_linearized_projected": "projected dense",
    "C_self_torch_logq": "projected dense",
    "conservative_scatter": "conservative deposition",
    "C_self_torch_logq_conservative_scatter": "conservative deposition",
}

SHAPE_LABELS = {
    "MB_T_eq_m": r"Maxwell--Boltzmann",
    "two_bump": r"two-bump nonthermal",
    "hot_tail": r"hot-tail distortion",
}

DEVICE_LABELS = {
    "cpu": "CPU",
    "cuda": "GPU",
}


def operator_label(name):
    return OPERATOR_LABELS.get(str(name), str(name).replace("_", " "))


def shape_label(name):
    return SHAPE_LABELS.get(str(name), str(name).replace("_", " "))


def device_label(name):
    return DEVICE_LABELS.get(str(name), str(name).upper())


def curve_label(device, op):
    return rf"{device_label(device)}, {operator_label(op)}"


def safe_filename(text):
    return (
        str(text)
        .replace(" ", "_")
        .replace("/", "-")
        .replace("\\", "-")
        .replace("$", "")
        .replace("{", "")
        .replace("}", "")
        .replace("^", "")
    )


# ============================================================
# Loading
# ============================================================

def load_all_benchmarks():
    csv_files = sorted(BENCHMARK_DIR.glob("collision_benchmark_*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No benchmark CSV files found in:\n{BENCHMARK_DIR}\n"
            "Run benchmark.py first."
        )

    frames = []

    for path in csv_files:
        df = pd.read_csv(path)
        df["source_file"] = path.name
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)

    print("Loaded benchmark files:")
    for path in csv_files:
        print(f"  {path.name}")

    print("\nColumns:")
    print(df_all.columns.tolist())

    return df_all


def savefig(name):
    out = PLOTS_DIR / name
    plt.tight_layout()
    plt.savefig(out, dpi=250)
    print(f"Saved: {out}")
    plt.show()


# ============================================================
# Optional deduplication
# ============================================================

def deduplicate_benchmarks(df):
    """
    If the same benchmark settings appear multiple times because the script
    was rerun, keep the most recent row according to source-file order.

    This avoids overplotting identical CPU/GPU curves.
    """
    key_cols = [
        "device",
        "dtype",
        "N",
        "shape",
        "operator",
        "Ng",
        "batch_size",
        "lam",
        "m",
        "a",
        "apply_conservation_projection",
    ]

    existing = [col for col in key_cols if col in df.columns]

    if not existing:
        return df

    return df.drop_duplicates(subset=existing, keep="last").copy()


# ============================================================
# Plotting functions
# ============================================================

def plot_runtime_vs_N(df):
    for shape in sorted(df["shape"].unique()):
        sub = df[df["shape"] == shape]

        plt.figure(figsize=(8.2, 5.2))

        for (device, op), g in sub.groupby(["device", "operator"]):
            g = g.sort_values("N")

            plt.loglog(
                g["N"],
                g["time_median_s"],
                marker="o",
                linewidth=1.8,
                markersize=5,
                label=curve_label(device, op),
            )

        plt.xlabel(r"$N_{\rm grid}$")
        plt.ylabel(r"median runtime per collision call [s]")
        plt.title(rf"Collision-operator runtime: {shape_label(shape)}")
        plt.legend(frameon=True)

        savefig(f"runtime_vs_N_{safe_filename(shape)}.png")


def plot_speedup_vs_N(df):
    for shape in sorted(df["shape"].unique()):
        sub = df[df["shape"] == shape]

        if not {"cpu", "cuda"}.issubset(set(sub["device"].unique())):
            print(f"Skipping speedup plot for {shape}: need both CPU and GPU data.")
            continue

        plt.figure(figsize=(8.2, 5.2))

        for op in sorted(sub["operator"].unique()):
            cpu = sub[(sub["device"] == "cpu") & (sub["operator"] == op)]
            gpu = sub[(sub["device"] == "cuda") & (sub["operator"] == op)]

            merge_cols = [
                "N",
                "shape",
                "operator",
                "dtype",
                "Ng",
                "batch_size",
                "lam",
                "m",
                "a",
                "apply_conservation_projection",
            ]

            merge_cols = [col for col in merge_cols if col in cpu.columns and col in gpu.columns]

            merged = pd.merge(
                cpu,
                gpu,
                on=merge_cols,
                suffixes=("_cpu", "_gpu"),
            )

            if len(merged) == 0:
                print(
                    f"No matching CPU/GPU rows for shape={shape}, operator={op}. "
                    "Check N-list, dtype, Ng, batch_size, lam, m, a."
                )
                continue

            merged = merged.sort_values("N")
            speedup = merged["time_median_s_cpu"] / merged["time_median_s_gpu"]

            plt.semilogx(
                merged["N"],
                speedup,
                marker="o",
                linewidth=1.8,
                markersize=5,
                label=operator_label(op),
            )

        plt.axhline(1.0, linestyle="--", linewidth=1.2)
        plt.xlabel(r"$N_{\rm grid}$")
        plt.ylabel(r"speedup, $t_{\rm CPU}/t_{\rm GPU}$")
        plt.title(rf"GPU speedup: {shape_label(shape)}")
        plt.legend(frameon=True)

        savefig(f"speedup_vs_N_{safe_filename(shape)}.png")


def plot_energy_conservation(df):
    if "rel_energy" not in df.columns:
        print("Skipping energy conservation plot: rel_energy column not found.")
        return

    for shape in sorted(df["shape"].unique()):
        sub = df[df["shape"] == shape]

        plt.figure(figsize=(8.2, 5.2))

        for (device, op), g in sub.groupby(["device", "operator"]):
            g = g.sort_values("N")

            y = g["rel_energy"].abs()

            plt.loglog(
                g["N"],
                y,
                marker="o",
                linewidth=1.8,
                markersize=5,
                label=curve_label(device, op),
            )

        plt.xlabel(r"$N_{\rm grid}$")
        plt.ylabel(r"$|\Delta E|_{\rm rel}$")
        plt.title(rf"Energy conservation diagnostic: {shape_label(shape)}")
        plt.legend(frameon=True)

        savefig(f"energy_conservation_{safe_filename(shape)}.png")


def plot_number_conservation(df):
    if "rel_number" not in df.columns:
        print("Skipping number conservation plot: rel_number column not found.")
        return

    for shape in sorted(df["shape"].unique()):
        sub = df[df["shape"] == shape]

        plt.figure(figsize=(8.2, 5.2))

        for (device, op), g in sub.groupby(["device", "operator"]):
            g = g.sort_values("N")

            y = g["rel_number"].abs()

            plt.loglog(
                g["N"],
                y,
                marker="o",
                linewidth=1.8,
                markersize=5,
                label=curve_label(device, op),
            )

        plt.xlabel(r"$N_{\rm grid}$")
        plt.ylabel(r"$|\Delta N|_{\rm rel}$")
        plt.title(rf"Number conservation diagnostic: {shape_label(shape)}")
        plt.legend(frameon=True)

        savefig(f"number_conservation_{safe_filename(shape)}.png")


def plot_outside_weight_fraction(df):
    if "outside_weight_fraction" not in df.columns:
        print("Skipping outside-weight plot: outside_weight_fraction column not found.")
        return

    for shape in sorted(df["shape"].unique()):
        sub = df[df["shape"] == shape]

        plt.figure(figsize=(8.2, 5.2))

        for (device, op), g in sub.groupby(["device", "operator"]):
            g = g.sort_values("N")

            y = g["outside_weight_fraction"].abs()

            plt.loglog(
                g["N"],
                y,
                marker="o",
                linewidth=1.8,
                markersize=5,
                label=curve_label(device, op),
            )

        plt.xlabel(r"$N_{\rm grid}$")
        plt.ylabel(r"outside-grid kernel-weight fraction")
        plt.title(rf"Grid-leakage diagnostic: {shape_label(shape)}")
        plt.legend(frameon=True)

        savefig(f"outside_weight_fraction_{safe_filename(shape)}.png")


def plot_valid_fraction(df):
    if "valid_fraction_given_phys" not in df.columns:
        print("Skipping valid-fraction plot: valid_fraction_given_phys column not found.")
        return

    for shape in sorted(df["shape"].unique()):
        sub = df[df["shape"] == shape]

        plt.figure(figsize=(8.2, 5.2))

        for (device, op), g in sub.groupby(["device", "operator"]):
            g = g.sort_values("N")

            plt.semilogx(
                g["N"],
                g["valid_fraction_given_phys"],
                marker="o",
                linewidth=1.8,
                markersize=5,
                label=curve_label(device, op),
            )

        plt.xlabel(r"$N_{\rm grid}$")
        plt.ylabel(r"valid fraction of physical configurations")
        plt.title(rf"Valid phase-space fraction: {shape_label(shape)}")
        plt.legend(frameon=True)

        savefig(f"valid_fraction_{safe_filename(shape)}.png")


# ============================================================
# Main
# ============================================================

def main():
    print(f"Benchmark directory: {BENCHMARK_DIR}")
    print(f"Plots directory:     {PLOTS_DIR}")

    df = load_all_benchmarks()
    df = deduplicate_benchmarks(df)

    print("\nPreview:")
    print(df.head())

    plot_runtime_vs_N(df)
    plot_speedup_vs_N(df)
    plot_energy_conservation(df)
    plot_number_conservation(df)
    plot_outside_weight_fraction(df)
    plot_valid_fraction(df)


if __name__ == "__main__":
    main()
