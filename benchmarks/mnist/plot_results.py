#!/usr/bin/env python3
"""Plot benchmark timing results: per-run trajectories and mean/stddev comparison.

Reads raw.csv and summary.csv from a benchmark run directory and writes a
combined figure to a plots/ directory next to results/.

Usage:
    python plot_results.py <run_dir> --title "MNIST MLP training time"
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "grid.color": GRID,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def plot_benchmark(run_dir: str, title: str) -> str:
    run_dir = os.path.normpath(run_dir)
    df_raw = pd.read_csv(os.path.join(run_dir, "raw.csv"))
    df_summary = pd.read_csv(os.path.join(run_dir, "summary.csv"))
    df_summary["run"] = df_summary["scope"].str.extract(r"run_(\d+)").astype(int)
    df_summary = df_summary.sort_values("run").reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, color=INK_PRIMARY, fontsize=14, x=0.02, ha="left")

    for _, group in df_raw.groupby("run"):
        group = group.sort_values("epoch")
        ax1.plot(group["epoch"], group["time_ms"] / 1000, color=BLUE, linewidth=1.5,
                  alpha=0.6, solid_capstyle="round")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Time (s)")
    ax1.set_title("Per-epoch time by run", color=INK_PRIMARY, fontsize=13, loc="left")
    ax1.grid(axis="y", linewidth=0.8)
    ax1.set_axisbelow(True)

    n_runs = df_raw["run"].nunique()
    overall_mean = df_summary["mean_ms"].mean() / 1000
    ax2.bar(df_summary["run"], df_summary["mean_ms"] / 1000, yerr=df_summary["stddev_ms"] / 1000,
            color=BLUE, width=0.6, capsize=4, error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1.2})
    ax2.axhline(overall_mean, color=INK_MUTED, linewidth=1, linestyle="--")
    ax2.annotate(f"overall mean: {overall_mean:.1f}s", xy=(n_runs - 1, overall_mean), xytext=(0, 6),
                 textcoords="offset points", ha="right", color=INK_SECONDARY, fontsize=9)
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel("Run")
    ax2.set_title("Mean time per run (± stddev)", color=INK_PRIMARY, fontsize=13, loc="left")
    ax2.set_xticks(df_summary["run"])
    ax2.grid(axis="y", linewidth=0.8)
    ax2.set_axisbelow(True)

    fig.tight_layout(rect=(0, 0, 1, 0.94))

    results_dir = os.path.dirname(run_dir)
    benchmark_dir = os.path.dirname(results_dir)
    plots_dir = os.path.join(benchmark_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    out_path = os.path.join(plots_dir, f"{os.path.basename(run_dir)}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Path to the benchmark run directory (contains raw.csv and summary.csv)")
    parser.add_argument("--title", required=True, help="Title describing what was measured")
    args = parser.parse_args()

    out_path = plot_benchmark(args.run_dir, args.title)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
