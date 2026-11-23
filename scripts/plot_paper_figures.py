"""Reproduce publication-quality summary figures from results/paper_summary_metrics.csv."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    os.makedirs("figures", exist_ok=True)
    df = pd.read_csv("results/paper_summary_metrics.csv")
    main_df = df[df["table"] == "Table9_Main"].copy()

    pal = {
        "C": "#8d99ae", "R": "#4a5568", "F": "#e09f3e",
        "F+S": "#b5651d", "H": "#2d7d6a", "H-NU": "#6bc5a8"
    }
    methods = main_df["method"].tolist()
    colors = [pal[m] for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), dpi=300)
    for ax, col, title in [
        (axes[0], "burden_pp", "Disease Burden (pp)"),
        (axes[1], "copper_sp_sec", "Copper Load (sprays/sector)"),
        (axes[2], "cost_J", "Composite Cost J"),
    ]:
        vals = main_df[col].to_numpy()
        ax.bar(methods, vals, color=colors, edgecolor="#1a202c", lw=0.7)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", ls=":", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = "figures/summary_main_metrics.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {out}")


if __name__ == "__main__":
    main()
