from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config


COLORS = {
    "Conv": "#0072B2",
    "LSTM": "#009E73",
    "Attention contribution": "#D55E00",
    "Context": "#CC79A7",
    "Neural": "#4D4D4D",
    "Full": "#0072B2",
    "Zero Conv signal": "#E69F00",
    "Zero LSTM signal": "#D55E00",
    "Uniform Attention": "#009E73",
}


def setup_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
        "axes.unicode_minus": False,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.8,
        "savefig.dpi": 300,
    })
    sns.set_style("ticks")


def save(fig, name: str):
    base = config.FIGURE_DIR / name
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure1_baseline():
    df = pd.read_csv(config.TABLE_DIR / "experiment1_baseline_merged.csv")
    long = df.melt(id_vars="fold", value_vars=["accuracy", "recall", "f1"],
                   var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(5.4, 3.0), constrained_layout=True)
    palette = {"accuracy": "#0072B2", "recall": "#009E73", "f1": "#D55E00"}
    for metric, sub in long.groupby("metric"):
        x = np.arange(len(sub))
        ax.plot(x, sub["value"], marker="o", linewidth=1.4, markersize=5,
                color=palette[metric], label=metric.capitalize())
    ax.set_xticks(range(3), ["Fold 1", "Fold 2", "Fold 3"])
    ax.set_ylabel("Score (%)")
    ax.set_ylim(65, 80)
    ax.set_title("Legacy baseline reproduced on the original three test folds")
    ax.legend(frameon=False, ncol=3)
    sns.despine(ax=ax)
    save(fig, "figure1_baseline_reproduction")


def _paired_recording_plot(ax, df, x_col, y_col, order, title, ylabel):
    jitter = np.linspace(-0.09, 0.09, max(1, df["recording"].nunique()))
    for j, (recording, sub) in enumerate(df.groupby("recording")):
        sub = sub.set_index(x_col).reindex(order)
        x = np.arange(len(order)) + jitter[j]
        ax.plot(x, sub[y_col], color="#A0A0A0", alpha=0.35, linewidth=0.7, zorder=1)
        ax.scatter(x, sub[y_col], c=[COLORS.get(v, "#4D4D4D") for v in order],
                   s=15, alpha=0.75, edgecolor="white", linewidth=0.3, zorder=2)
    means = df.groupby(x_col)[y_col].mean().reindex(order)
    ax.scatter(np.arange(len(order)), means, marker="D", s=40,
               c=[COLORS.get(v, "#4D4D4D") for v in order],
               edgecolor="black", linewidth=0.5, zorder=3)
    ax.axhline(0, color="#777777", linestyle="--", linewidth=0.7)
    ax.set_xticks(range(len(order)), [x.replace(" contribution", "\ncontribution") for x in order])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    sns.despine(ax=ax)


def figure2_rsa_cka():
    df = pd.read_csv(config.TABLE_DIR / "experiment2_rsa_cka_by_recording.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    for col, run in enumerate(["run1", "run2"]):
        sub = df[df["run"] == run]
        _paired_recording_plot(
            axes[0, col], sub, "layer", "z_euclidean_rsa", config.LAYER_ORDER,
            f"{run.upper()}: neural–model RSA", "Spearman RSA (r)"
        )
        _paired_recording_plot(
            axes[1, col], sub, "layer", "linear_cka", config.LAYER_ORDER,
            f"{run.upper()}: neural–model linear CKA", "Linear CKA"
        )
    save(fig, "figure2_neural_model_rsa_cka")

    # Presentation-friendly landscape exports keep labels readable on 16:9 slides.
    for metric, ylabel, suffix in [
        ("z_euclidean_rsa", "Spearman RSA (r)", "rsa"),
        ("linear_cka", "Linear CKA", "cka"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
        for ax, run in zip(axes, ["run1", "run2"]):
            sub = df[df["run"] == run]
            _paired_recording_plot(
                ax, sub, "layer", metric, config.LAYER_ORDER,
                f"{run.upper()}", ylabel
            )
        save(fig, f"figure2_{suffix}_landscape")


def figure3_decision():
    df = pd.read_csv(config.TABLE_DIR / "experiment3_decision_rsa_by_recording.csv")
    order = ["Neural"] + config.LAYER_ORDER
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    for ax, run in zip(axes, ["run1", "run2"]):
        sub = df[df["run"] == run]
        _paired_recording_plot(
            ax, sub, "representation", "decision_rsa", order,
            f"{run.upper()}: alignment with behavioral decision geometry",
            "Decision RSA (r)"
        )
        ax.tick_params(axis="x", rotation=18)
    save(fig, "figure3_decision_geometry")


def figure4_overlap():
    df = pd.read_csv(config.TABLE_DIR / "experiment4_overlap_by_recording.csv")
    layers = ["Conv", "LSTM", "Attention contribution", "Attention weights"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True, sharey=True)
    palette = {"Shared 2.5 s overlap": "#0072B2", "Same-label non-overlap control": "#BDBDBD"}
    for ax, run in zip(axes, ["run1", "run2"]):
        sub = df[df["run"] == run].copy()
        sns.boxplot(data=sub, x="layer", y="similarity", hue="pair_type", order=layers,
                    palette=palette, showfliers=False, width=0.65, linewidth=0.7, ax=ax)
        sns.stripplot(data=sub, x="layer", y="similarity", hue="pair_type", order=layers,
                      dodge=True, palette=palette, alpha=0.35, size=2.2, ax=ax, legend=False)
        ax.set_title(f"{run.upper()}: identical shared segment")
        ax.set_xlabel("")
        ax.set_ylabel("Cosine similarity")
        ax.tick_params(axis="x", rotation=18)
        if ax is axes[0]:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:2], labels[:2], frameon=False, loc="lower left")
        else:
            ax.get_legend().remove()
        sns.despine(ax=ax)
    save(fig, "figure4_overlap_context_dependence")


def figure5_intervention():
    acc = pd.read_csv(config.TABLE_DIR / "experiment5_intervention_merged.csv")
    align = pd.read_csv(config.TABLE_DIR / "experiment5_intervention_alignment_by_recording.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), constrained_layout=True)
    for condition in config.INTERVENTION_ORDER:
        sub = acc[acc["condition"] == condition]
        axes[0].plot(range(3), sub["accuracy"], marker="o", linewidth=1.2,
                     color=COLORS[condition], label=condition)
    axes[0].set_xticks(range(3), ["Fold 1", "Fold 2", "Fold 3"])
    axes[0].set_ylabel("Legacy merged accuracy (%)")
    axes[0].set_title("Prediction under frozen-model intervention")
    axes[0].legend(frameon=False, fontsize=6)
    sns.despine(ax=axes[0])

    sub = (
        align[align["condition"].isin(["Full", "Uniform Attention"])]
        .groupby(["recording", "condition"], as_index=False)["context_neural_rsa"].mean()
    )
    _paired_recording_plot(
        axes[1], sub, "condition", "context_neural_rsa", ["Full", "Uniform Attention"],
        "Context–neural correspondence after intervention", "Context–neural RSA (r)"
    )
    axes[1].text(
        0.02, 0.98,
        "Zero Conv / Zero LSTM: context collapses\n(across-window variance ≈ 0), so RSA is undefined",
        transform=axes[1].transAxes, va="top", ha="left", fontsize=7,
        color="#555555",
    )
    save(fig, "figure5_functional_interventions")


def make_all_figures():
    config.ensure_dirs()
    setup_style()
    figure1_baseline()
    figure2_rsa_cka()
    figure3_decision()
    figure4_overlap()
    figure5_intervention()
    print(f"Figures written to {config.FIGURE_DIR}")


if __name__ == "__main__":
    make_all_figures()
