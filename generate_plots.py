"""
Run this separately after the main experiment finishes.
    python3 generate_plots.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.lines import Line2D

TABLES_DIR  = Path("results/tables")
FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

#loading data 

summary_df = pd.read_csv(TABLES_DIR / "summary_results.csv")
print(f"  Rows     : {len(summary_df)}")
print(f"  Datasets : {summary_df['dataset_name'].unique()}")
print(f"  Variations: {summary_df['variation_type'].unique()}")
print(f"  Models   : {summary_df['model_name'].unique()}")

#style
MODEL_COLORS = {
    "logistic_regression": "#2196F3",
    "random_forest": "#4CAF50",
    "mlp": "#FF5722",
}
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "mlp": "MLP",
}
DATASET_LABELS = {
    "adult_income": "Adult Income",
    "breast_cancer": "Breast Cancer",
}
sns.set_theme(style="whitegrid", font_scale=1.05)


#stability of baseline
def plot_rq1(df):
    base    = df[df["variation_type"] == "baseline"].copy()
    metrics = [
        ("mean_jaccard",  "std_jaccard",  "Jaccard Similarity"),
        ("mean_spearman", "std_spearman", "Spearman Correlation"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("LIME Stability on Baseline Dataset",
                 fontsize=14, fontweight="bold")

    for ax, (mean_col, std_col, ylabel) in zip(axes, metrics):
        x_pos, xticks, xlabels = 0, [], []
        for ds_name, ds_label in DATASET_LABELS.items():
            grp_positions = []
            for model in MODEL_COLORS:
                row = base[(base["dataset_name"] == ds_name) &
                           (base["model_name"]   == model)]
                if row.empty:
                    x_pos += 1
                    continue
                ax.bar(x_pos, row[mean_col].values[0],
                       yerr=row[std_col].values[0],
                       color=MODEL_COLORS[model], capsize=5,
                       alpha=0.85, width=0.7)
                grp_positions.append(x_pos)
                x_pos += 1
            if grp_positions:
                xticks.append(np.mean(grp_positions))
                xlabels.append(ds_label)
            x_pos += 0.8
        ax.set_ylabel(ylabel)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, ls="--", color="grey", lw=0.8, alpha=0.5)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for c in MODEL_COLORS.values()]
    axes[1].legend(handles, MODEL_LABELS.values(),
                   loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rq1_baseline_stability.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: rq1_baseline_stability.png")


#effect of variations 
def plot_rq2(df, variation_type, xlabel, filename):
    base = df[df["variation_type"] == "baseline"].copy()
    base["variation_value"] = 0.0
    var  = df[df["variation_type"] == variation_type].copy()
    sub  = pd.concat([base, var], ignore_index=True).sort_values("variation_value")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Effect of {variation_type.title()} on LIME Stability",
                 fontsize=14, fontweight="bold")

    for ax, (ds_name, ds_label) in zip(axes, DATASET_LABELS.items()):
        ds_sub = sub[sub["dataset_name"] == ds_name]
        ax.set_title(ds_label)
        for model in MODEL_COLORS:
            rows = ds_sub[ds_sub["model_name"] == model]
            if rows.empty:
                continue
            ax.plot(rows["variation_value"], rows["mean_jaccard"],
                    color=MODEL_COLORS[model], ls="-", marker="o",
                    markersize=5, lw=1.8)
            ax.plot(rows["variation_value"], rows["mean_spearman"],
                    color=MODEL_COLORS[model], ls="--", marker="s",
                    markersize=5, lw=1.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Stability Score")
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, ls=":", color="grey", lw=0.7)

    leg = (
        [Line2D([0],[0], color=MODEL_COLORS[m], lw=2,
                label=MODEL_LABELS[m]) for m in MODEL_COLORS]
        + [Line2D([0],[0], color="grey", lw=2, ls="-",  label="Jaccard"),
           Line2D([0],[0], color="grey", lw=2, ls="--", label="Spearman")]
    )
    axes[0].legend(handles=leg, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


#comparison of model
def plot_rq3(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LIME Stability Across ML Models",
                 fontsize=14, fontweight="bold")

    for ax, (ds_name, ds_label) in zip(axes, DATASET_LABELS.items()):
        grp = (df[df["dataset_name"] == ds_name]
               .groupby("model_name")[["mean_jaccard", "mean_spearman"]]
               .mean().reset_index())

        x     = np.arange(len(grp))
        width = 0.35
        bars1 = ax.bar(x - width/2, grp["mean_jaccard"], width,
                       color=[MODEL_COLORS[m] for m in grp["model_name"]],
                       alpha=0.85, label="Jaccard")
        bars2 = ax.bar(x + width/2, grp["mean_spearman"], width,
                       color=[MODEL_COLORS[m] for m in grp["model_name"]],
                       alpha=0.45, label="Spearman", hatch="//")

        ax.set_title(ds_label)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_LABELS[m] for m in grp["model_name"]], fontsize=9)
        ax.set_ylabel("Average Stability Score")
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, ls="--", color="grey", lw=0.7)

        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    axes[0].legend(["Jaccard", "Spearman"], fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "rq3_model_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: rq3_model_comparison.png")


#heatmap
def plot_heatmap(df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Heatmap — Mean Jaccard Stability (all conditions)",
                 fontsize=14, fontweight="bold")

    for ax, (ds_name, ds_label) in zip(axes, DATASET_LABELS.items()):
        pivot = (df[df["dataset_name"] == ds_name]
                 .pivot_table(index="variation_type",
                              columns="model_name",
                              values="mean_jaccard",
                              aggfunc="mean"))
        pivot.columns = [MODEL_LABELS[c] for c in pivot.columns]
        sns.heatmap(pivot, ax=ax, annot=True, fmt=".2f",
                    cmap="YlOrRd_r", vmin=0, vmax=1,
                    linewidths=0.5,
                    cbar_kws={"label": "Jaccard Similarity"})
        ax.set_title(ds_label)
        ax.set_xlabel("")
        ax.set_ylabel("Variation Type")
        ax.tick_params(axis="x", rotation=25)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "heatmap_all_conditions.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: heatmap_all_conditions.png")


#generating all plots
print("\nGenerating plots")
plot_rq1(summary_df)
plot_rq2(summary_df, "noise",     "Noise Level",             "rq2a_noise.png")
plot_rq2(summary_df, "imbalance", "Majority Class Fraction", "rq2b_imbalance.png")
plot_rq3(summary_df)
plot_heatmap(summary_df)

print("\n✓ All 5 plots saved to results/figures/")
