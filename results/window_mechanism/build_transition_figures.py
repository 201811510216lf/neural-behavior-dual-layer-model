"""Build publication figures from the paired stage-wise prediction audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source_data"
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
QA = ROOT / "qa"
SKILL_SCRIPTS = Path.home() / ".codex" / "skills" / "scipilot-figure-skill" / "scripts"
sys.path.insert(0, str(SKILL_SCRIPTS))

from check_figure import check_figure  # noqa: E402
from export_figure import export_figure  # noqa: E402
from visual_qa import audit_layout, render_preview  # noqa: E402


COMPONENTS: Sequence[Tuple[str, int, str]] = (
    ("run1", 0, "Run1\nL1"),
    ("run1", 1, "Run1\nL2/L4/L5 gate"),
    ("run1", 2, "Run1\nL3"),
    ("run2", 0, "Run2\nL2"),
    ("run2", 1, "Run2\nL4"),
    ("run2", 2, "Run2\nL5"),
)

MASK_ORDER = [
    "mask_max_L1",
    "mask_max_L2",
    "mask_max_L3",
    "mask_max_L4",
    "mask_max_L5",
    "attention_zero_L2",
    "attention_zero_L3",
    "attention_zero_L5",
    "attention_shuffle_L2",
    "attention_shuffle_L3",
    "attention_shuffle_L5",
]
MASK_LABELS = {
    "mask_max_L1": "Max mask L1",
    "mask_max_L2": "Max mask L2",
    "mask_max_L3": "Max mask L3",
    "mask_max_L4": "Max mask L4",
    "mask_max_L5": "Max mask L5",
    "attention_zero_L2": "Attn zero L2",
    "attention_zero_L3": "Attn zero L3",
    "attention_zero_L5": "Attn zero L5",
    "attention_shuffle_L2": "Attn shuffle L2",
    "attention_shuffle_L3": "Attn shuffle L3",
    "attention_shuffle_L5": "Attn shuffle L5",
}
STRUCTURE_ORDER = ["structure_no_conv", "structure_no_lstm", "structure_no_attention"]
STRUCTURE_LABELS = {
    "structure_no_conv": "NoConv",
    "structure_no_lstm": "NoLSTM",
    "structure_no_attention": "NoAttention",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    sns.set_style("ticks")


def add_panel_labels(axes: Iterable[plt.Axes], labels: str) -> None:
    for axis, label in zip(axes, labels):
        axis.annotate(
            label,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(-17, 7),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


def export_and_audit(fig: plt.Figure, name: str, size: Tuple[float, float]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    preview = QA / f"{name}_preview.png"
    render_preview(fig, str(preview), dpi=150)
    issues = audit_layout(fig)
    (QA / f"{name}_layout_audit.json").write_text(
        json.dumps([{"level": level, "message": message} for level, message in issues], indent=2),
        encoding="utf-8",
    )
    export_figure(
        fig,
        str(FIGURES / name),
        formats=["svg", "pdf", "png", "tiff"],
        size_inches=size,
        dpi=600,
        grayscale_preview=True,
        tight=False,
    )
    plt.close(fig)


def component_matrix(summary: pd.DataFrame, ids: Sequence[str], metric: str) -> pd.DataFrame:
    rows: List[Dict] = []
    part = summary[(summary["scope"] == "true_class") & summary["comparison_id"].isin(ids)]
    for comparison_id in ids:
        for stage, class_id, component_label in COMPONENTS:
            selected = part[
                (part["comparison_id"] == comparison_id)
                & (part["stage"] == stage)
                & (part["class_id"] == class_id)
            ]
            rows.append(
                {
                    "comparison_id": comparison_id,
                    "component": component_label,
                    metric: float(selected.iloc[0][metric]) if not selected.empty else np.nan,
                }
            )
    return pd.DataFrame(rows).pivot(index="comparison_id", columns="component", values=metric).reindex(
        index=ids, columns=[item[2] for item in COMPONENTS]
    )


def draw_rate_heatmaps(
    summary: pd.DataFrame,
    ids: Sequence[str],
    labels: Dict[str, str],
    name: str,
    size: Tuple[float, float],
    vmax: float,
) -> None:
    damage = component_matrix(summary, ids, "damage_rate_percent")
    recovery = component_matrix(summary, ids, "recovery_rate_percent")
    damage.index = [labels[index] for index in damage.index]
    recovery.index = [labels[index] for index in recovery.index]

    fig, axes = plt.subplots(1, 2, figsize=size, constrained_layout=True)
    for axis, matrix, title, cmap in (
        (axes[0], damage, "Correct to wrong (damage)", "mako_r"),
        (axes[1], recovery, "Wrong to correct (recovery)", "crest"),
    ):
        sns.heatmap(
            matrix,
            ax=axis,
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            annot=True,
            fmt=".1f",
            annot_kws={"fontsize": 5.5},
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Rate (%)", "shrink": 0.72, "pad": 0.02},
        )
        axis.set_title(title)
        axis.set_xlabel("Hierarchical stage component")
        axis.set_ylabel("")
        axis.set_xticklabels(axis.get_xticklabels(), rotation=0, ha="center")
        axis.tick_params(axis="y", rotation=0)
    add_panel_labels(axes, "ab")
    export_and_audit(fig, name, size)


def draw_key_transition_counts(summary: pd.DataFrame) -> None:
    key_ids = ["mask_max_L2", "mask_max_L5"]
    labels = {"mask_max_L2": "Maximum mask L2", "mask_max_L5": "Maximum mask L5"}
    data = summary[
        (summary["scope"] == "true_class") & summary["comparison_id"].isin(key_ids)
    ].copy()
    data["component"] = data.apply(
        lambda row: next(
            label for stage, class_id, label in COMPONENTS
            if stage == row["stage"] and class_id == int(row["class_id"])
        ),
        axis=1,
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True, sharey=True)
    y = np.arange(len(COMPONENTS))
    for axis, comparison_id in zip(axes, key_ids):
        part = data[data["comparison_id"] == comparison_id].set_index("component").reindex(
            [item[2] for item in COMPONENTS]
        )
        damage = -part["CW"].to_numpy(dtype=float)
        recovery = part["WC"].to_numpy(dtype=float)
        axis.barh(y, damage, height=0.58, color="#D55E00", label="CW: damage")
        axis.barh(y, recovery, height=0.58, color="#0072B2", label="WC: recovery")
        axis.axvline(0, color="#222222", linewidth=0.8)
        axis.set_yticks(y, [item[2].replace("\n", " ") for item in COMPONENTS])
        axis.invert_yaxis()
        axis.set_title(labels[comparison_id])
        axis.set_xlabel("Changed windows (CW shown left; WC right)")
        limit = max(5.0, float(np.nanmax(np.abs(np.r_[damage, recovery]))) * 1.2)
        axis.set_xlim(-limit, limit)
        for yi, value in enumerate(damage):
            if value:
                axis.text(value - 0.7, yi, f"{int(abs(value))}", va="center", ha="right", fontsize=6)
        for yi, value in enumerate(recovery):
            if value:
                axis.text(value + 0.7, yi, f"{int(value)}", va="center", ha="left", fontsize=6)
    axes[1].legend(loc="lower right")
    add_panel_labels(axes, "ab")
    export_and_audit(fig, "fig_i_key_CW_WC_counts_stagewise", (7.2, 2.7))


def matrix_from_long(data: pd.DataFrame, stage: str, value: str) -> np.ndarray:
    part = data[data["stage"] == stage]
    return (
        part.pivot(index="true_class", columns="predicted_class", values=value)
        .sort_index()
        .sort_index(axis=1)
        .to_numpy(dtype=float)
    )


def draw_key_confusions(confusions: pd.DataFrame, comparison_id: str, display: str, name: str) -> None:
    data = confusions[confusions["comparison_id"] == comparison_id]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.45), constrained_layout=True)
    stage_labels = {
        "run1": ["L1", "L2/L4/L5\ngate", "L3"],
        "run2": ["L2", "L4", "L5"],
    }
    delta_limit = max(
        1.0,
        float(np.nanmax(np.abs(data["delta_row_pp"].to_numpy(dtype=float)))),
    )
    for row_index, stage in enumerate(("run1", "run2")):
        baseline = matrix_from_long(data, stage, "baseline_row_percent")
        intervention = matrix_from_long(data, stage, "intervention_row_percent")
        delta = matrix_from_long(data, stage, "delta_row_pp")
        for column_index, (matrix, title) in enumerate(
            ((baseline, "Baseline"), (intervention, display), (delta, "Delta"))
        ):
            axis = axes[row_index, column_index]
            if column_index < 2:
                sns.heatmap(
                    matrix,
                    ax=axis,
                    cmap="Blues",
                    vmin=0,
                    vmax=100,
                    annot=True,
                    fmt=".1f",
                    annot_kws={"fontsize": 6},
                    square=True,
                    linewidths=0.5,
                    linecolor="white",
                    cbar=column_index == 1,
                    cbar_kws={"label": "Row-normalized (%)", "shrink": 0.72, "pad": 0.02},
                )
            else:
                sns.heatmap(
                    matrix,
                    ax=axis,
                    cmap="RdBu_r",
                    center=0,
                    vmin=-delta_limit,
                    vmax=delta_limit,
                    annot=True,
                    fmt="+.1f",
                    annot_kws={"fontsize": 6},
                    square=True,
                    linewidths=0.5,
                    linecolor="white",
                    cbar=True,
                    cbar_kws={"label": "Delta (pp)", "shrink": 0.72, "pad": 0.02},
                )
            axis.set_title(f"{stage.upper()} | {title}")
            axis.set_xlabel("Predicted class")
            axis.set_ylabel("True class" if column_index == 0 else "")
            axis.set_xticklabels(stage_labels[stage], rotation=0)
            axis.set_yticklabels(stage_labels[stage], rotation=0)
    add_panel_labels(axes.flat, "abcdef")
    export_and_audit(fig, name, (7.2, 4.45))


def run_file_checks(names: Sequence[str]) -> None:
    records = []
    for name in names:
        for suffix in (".svg", ".pdf", ".png", ".tiff"):
            path = FIGURES / f"{name}{suffix}"
            issues, info = check_figure(str(path), min_dpi=300)
            records.append(
                {
                    "file": path.name,
                    "issues": [{"level": level, "message": message} for level, message in issues],
                    "info": info,
                }
            )
    (QA / "transition_figure_file_checks.json").write_text(
        json.dumps(records, indent=2, default=str), encoding="utf-8"
    )


def main() -> None:
    apply_style()
    summary_path = TABLES / "table_c_transition_summary_pooled.csv"
    confusion_path = SOURCE / "paired_confusion_matrices_stagewise.csv"
    if not summary_path.exists() or not confusion_path.exists():
        raise FileNotFoundError("Run run_paired_transition_analysis.py first")
    summary = pd.read_csv(summary_path)
    confusions = pd.read_csv(confusion_path)

    draw_rate_heatmaps(
        summary,
        MASK_ORDER,
        MASK_LABELS,
        "fig_g_mask_damage_recovery_stagewise",
        (7.2, 5.0),
        vmax=25.0,
    )
    draw_rate_heatmaps(
        summary,
        STRUCTURE_ORDER,
        STRUCTURE_LABELS,
        "fig_h_structure_damage_recovery_stagewise",
        (7.2, 2.4),
        vmax=70.0,
    )
    draw_key_transition_counts(summary)
    draw_key_confusions(
        confusions,
        "mask_max_L2",
        "Maximum mask L2",
        "fig_j_L2_max_confusion_stagewise",
    )
    draw_key_confusions(
        confusions,
        "mask_max_L5",
        "Maximum mask L5",
        "fig_k_L5_max_confusion_stagewise",
    )
    names = [
        "fig_g_mask_damage_recovery_stagewise",
        "fig_h_structure_damage_recovery_stagewise",
        "fig_i_key_CW_WC_counts_stagewise",
        "fig_j_L2_max_confusion_stagewise",
        "fig_k_L5_max_confusion_stagewise",
    ]
    run_file_checks(names)
    print("Transition figure suite complete.", flush=True)


if __name__ == "__main__":
    main()
