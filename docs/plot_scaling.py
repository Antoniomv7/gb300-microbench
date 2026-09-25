#!/usr/bin/env python3
"""Draw measured UMMA scaling from the archived CSV without altering its analysis."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "docs/umma_device_scaling.svg")
    args = parser.parse_args()
    with (root / "results/umma_device_scaling.csv").open(newline="") as source:
        rows = {(r["method"], r["scale"]): r for r in csv.DictReader(source)}

    methods = ("umma_1sm", "umma_2sm")
    labels = ("One-SM work units", "Two-SM work units")
    colors = ("#28628b", "#b45b32")
    means, samples, efficiencies, efficiency_samples = [], [], [], []
    for method in methods:
        isolated = rows[method, "isolated"]
        device = rows[method, "device_scale"]
        means.append(float(device["mean_tflops"]))
        samples.append([float(device[f"campaign_{i}_tflops"]) for i in range(1, 4)])
        efficiencies.append(100 * float(device["scaling_efficiency_raw"]))
        efficiency_samples.append([
            100 * float(device[f"campaign_{i}_tflops"])
            / (int(device["work_units"]) * float(isolated[f"campaign_{i}_tflops"]))
            for i in range(1, 4)
        ])

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#a1aab3", "axes.labelcolor": "#253343",
        "text.color": "#253343", "xtick.color": "#253343", "ytick.color": "#253343",
        "svg.fonttype": "none", "svg.hashsalt": "gb300-measured-scaling",
    })
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    panels = (
        (means, samples, "Throughput across 148 SMs", "TFLOP/s", 2500, ",.1f"),
        (efficiencies, efficiency_samples, "Scaling relative to isolated units", "% of isolated throughput per SM", 110, ".1f"),
    )
    for ax, (values, campaigns, title, ylabel, ymax, fmt) in zip(axes, panels):
        error = [
            [max(0, v - min(s)) for v, s in zip(values, campaigns)],
            [max(0, max(s) - v) for v, s in zip(values, campaigns)],
        ]
        ax.bar(range(2), values, width=0.56, color=colors, yerr=error,
               capsize=4, error_kw={"elinewidth": 1, "capthick": 1}, zorder=3)
        for x, value in enumerate(values):
            suffix = "%" if ax is axes[1] else ""
            ypos = value - ymax * 0.09 if ax is axes[1] else value + ymax * 0.028
            color = "white" if ax is axes[1] else "#253343"
            ax.text(x, ypos, f"{value:{fmt}}{suffix}",
                    ha="center", va="bottom", color=color, fontweight="bold")
        ax.set(xticks=range(2), xticklabels=labels, ylim=(0, ymax), ylabel=ylabel)
        ax.set_title(title, fontsize=11, pad=12, fontweight="bold")
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(axis="y", color="#e3e7eb", linewidth=0.8, zorder=0)
        ax.tick_params(axis="x", length=0, pad=8)
    axes[1].axhline(100, color="#737f8a", linestyle="--", linewidth=0.9, zorder=2)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.22, wspace=0.39)
    fig.text(0.5, 0.055, "CUDA-event measurements · means and ranges of three campaigns",
             ha="center", fontsize=9, color="#596775")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Date": None} if args.output.suffix == ".svg" else None
    fig.savefig(args.output, metadata=metadata, facecolor="white", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
