"""Build publication figures and summary tables from the experiment bundle."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
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


OKABE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#222222",
    "grey": "#8A8A8A",
}
ARCH_COLORS = {
    "Full": "#0F4D92",
    "NoConv": "#42949E",
    "NoLSTM": "#9A4D8E",
    "NoAttention": "#D55E00",
}
MECHANISM_COLORS = {
    "Maximum": OKABE["vermillion"],
    "Attention shuffle": OKABE["blue"],
    "Attention zero": OKABE["sky"],
    "LSTM 50 ms": "#6F6F6F",
    "LSTM 100 ms": "#929292",
    "LSTM 200 ms": "#B5B5B5",
}
TARGETS = (2, 3, 5)
TARGET_LABELS = {2: "L2 / Stage 2", 3: "L3 / Hesitating", 5: "L5 / Walking"}
OUTCOME_KEYS = {2: "stage2_macro_recall", 3: "c2", 5: "c4"}


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
            "lines.linewidth": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    sns.set_context("paper", rc={"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8})
    sns.set_style("ticks")


def add_panel_labels(axes: Sequence[plt.Axes], labels: str) -> None:
    for ax, label in zip(axes, labels):
        ax.annotate(
            label,
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(-15, 7),
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


def symmetric_limit(values: Iterable[float], minimum: float = 1.0) -> float:
    arr = np.asarray(list(values), dtype=float)
    value = float(np.nanmax(np.abs(arr))) if arr.size else minimum
    return max(minimum, np.ceil(value * 2.0) / 2.0)


def fig_a_window_behavior() -> None:
    data = pd.read_csv(SOURCE / "fig_a_window_behavior_delta_recall.csv")
    behavior_order = [
        "Hole Exploration", "Wrong Hole Exploration", "Hesitating", "Changing Direction", "Walking"
    ]
    window_order = ["L1", "L2", "L3", "L4", "L5"]
    means = (
        data.groupby(["window", "behavior"], as_index=False)["delta_perturb_minus_baseline"].mean()
        .pivot(index="window", columns="behavior", values="delta_perturb_minus_baseline")
        .reindex(index=window_order, columns=behavior_order)
    )
    limit = symmetric_limit(means.to_numpy().ravel(), 2.0)

    fig = plt.figure(figsize=(7.2, 3.15), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 1.0], wspace=0.18)
    ax_hm = fig.add_subplot(gs[0, 0])
    ax_dot = fig.add_subplot(gs[0, 1])
    hm = sns.heatmap(
        means,
        ax=ax_hm,
        cmap="RdBu_r",
        center=0,
        vmin=-limit,
        vmax=limit,
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": 6},
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Mean delta recall (pp)", "shrink": 0.75, "pad": 0.02},
        square=True,
    )
    ax_hm.set_xlabel("Decoded behavior")
    ax_hm.set_ylabel("Perturbed significant window")
    ax_hm.set_xticklabels(["Hole", "Wrong\nhole", "Hesitate", "Change\ndir.", "Walk"], rotation=0, ha="center")
    hm.collections[0].colorbar.ax.tick_params(labelsize=6)

    key_specs = [
        ("L2", "Wrong Hole Exploration", "L2 -> Wrong hole"),
        ("L2", "Changing Direction", "L2 -> Direction"),
        ("L2", "Walking", "L2 -> Walking"),
        ("L3", "Hesitating", "L3 -> Hesitating"),
        ("L5", "Walking", "L5 -> Walking"),
    ]
    for yi, (window, behavior, label) in enumerate(key_specs):
        vals = data[(data.window == window) & (data.behavior == behavior)]["delta_perturb_minus_baseline"].to_numpy()
        jitter = np.linspace(-0.08, 0.08, len(vals))
        ax_dot.scatter(vals, yi + jitter, s=17, facecolor="white", edgecolor=OKABE["black"], linewidth=0.7, zorder=3)
        ax_dot.scatter([vals.mean()], [yi], s=32, marker="D", color=OKABE["vermillion"], edgecolor="white", linewidth=0.5, zorder=4)
    ax_dot.axvline(0, color="#777777", linestyle="--", linewidth=0.8)
    ax_dot.set_yticks(range(len(key_specs)))
    ax_dot.set_yticklabels([item[2] for item in key_specs])
    ax_dot.invert_yaxis()
    ax_dot.set_xlabel("Delta recall (pp)")
    ax_dot.set_title("Key cells: all three folds")
    ax_dot.grid(axis="x", color="#E6E6E6", linewidth=0.5)
    add_panel_labels([ax_hm, ax_dot], "ab")
    export_and_audit(fig, "fig_a_window_behavior", (7.2, 3.15))


def draw_delta_heatmap(ax: plt.Axes, data: pd.DataFrame, target: int, limit: float, show_y: bool) -> None:
    metric = OUTCOME_KEYS[target]
    part = data[data.target == target]
    baseline = part[part.condition == "baseline"].set_index(["architecture_label", "fold"])[metric]
    rows = []
    for _, row in part[part.condition != "baseline"].iterrows():
        rows.append(
            {
                "architecture": row.architecture_label,
                "condition": row.condition,
                "delta": float(row[metric] - baseline.loc[(row.architecture_label, row.fold)]),
            }
        )
    delta = pd.DataFrame(rows)
    matrix = (
        delta.groupby(["architecture", "condition"])["delta"].mean().unstack()
        .reindex(index=["Full", "NoConv", "NoLSTM", "NoAttention"])
        .reindex(columns=["maximum", "attention_shuffle", "attention_zero", "random_window"])
    )
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-limit,
        vmax=limit,
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": 6},
        linewidths=0.5,
        linecolor="white",
        cbar=False,
    )
    ax.set_xticklabels(["Max", "Att.\nshuf.", "Att.\nzero", "Random"], rotation=0, ha="center")
    if show_y:
        ax.set_yticklabels(["Full", "−Conv", "−LSTM", "−Attn"], rotation=0, fontsize=5.5)
    if not show_y:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(TARGET_LABELS[target])


def fig_b_architecture_interaction() -> None:
    data = pd.read_csv(SOURCE / "fig_b_architecture_perturbation_fold.csv")
    detail = pd.read_csv(TABLES / "table_a_fold_level_interactions.csv")
    all_deltas = []
    for target in TARGETS:
        metric = OUTCOME_KEYS[target]
        part = data[data.target == target]
        base = part[part.condition == "baseline"].set_index(["architecture", "fold"])[metric]
        for _, row in part[part.condition != "baseline"].iterrows():
            all_deltas.append(float(row[metric] - base.loc[(row.architecture, row.fold)]))
    limit = symmetric_limit(all_deltas, 3.0)

    fig = plt.figure(figsize=(7.2, 5.65), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.82], hspace=0.22, wspace=0.14)
    heat_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for index, (ax, target) in enumerate(zip(heat_axes, TARGETS)):
        draw_delta_heatmap(ax, data, target, limit, show_y=index == 0)
    heat_axes[0].set_ylabel("Architecture")
    sm = mpl.cm.ScalarMappable(norm=mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), cmap="RdBu_r")
    cbar = fig.colorbar(sm, ax=heat_axes, orientation="vertical", fraction=0.025, pad=0.02, aspect=22)
    cbar.set_label("Mean perturbation delta (pp; perturbation - baseline)")
    cbar.ax.tick_params(labelsize=6)

    forest_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    key = {
        2: ("maximum", "no_lstm"),
        3: ("maximum", "no_attention"),
        5: ("attention_shuffle", "no_attention"),
    }
    module_labels = {"no_lstm": "NoLSTM", "no_attention": "NoAttention"}
    for ax, target in zip(forest_axes, TARGETS):
        condition, module = key[target]
        full = detail[(detail.target == target) & (detail.condition == condition) & (detail.architecture == "full")]
        ablated = detail[(detail.target == target) & (detail.condition == condition) & (detail.architecture == module)]
        merged = full[["fold", "delta_perturb_minus_baseline"]].merge(
            ablated[["fold", "delta_perturb_minus_baseline"]], on="fold", suffixes=("_full", "_ablated")
        )
        interaction = merged.delta_perturb_minus_baseline_full - merged.delta_perturb_minus_baseline_ablated
        y = np.arange(len(merged))
        for yi, (_, row) in zip(y, merged.iterrows()):
            ax.plot(
                [row.delta_perturb_minus_baseline_ablated, row.delta_perturb_minus_baseline_full],
                [yi, yi], color="#B5B5B5", linewidth=0.8, zorder=1,
            )
            ax.scatter(row.delta_perturb_minus_baseline_full, yi, s=22, color=ARCH_COLORS["Full"], marker="o", zorder=3)
            ax.scatter(row.delta_perturb_minus_baseline_ablated, yi, s=22, color=ARCH_COLORS[module_labels[module]], marker="s", zorder=3)
        ax.axvline(0, color="#777777", linestyle="--", linewidth=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(merged.fold)
        ax.set_xlabel("Delta outcome (pp)")
        ax.set_title(f"{condition.replace('_', ' ').title()}\nmean interaction = {interaction.mean():.1f} pp")
        ax.grid(axis="x", color="#E6E6E6", linewidth=0.5)
    handles = [mpl.lines.Line2D([], [], color=ARCH_COLORS["Full"], marker="o", linestyle="", label="Full")]
    handles += [
        mpl.lines.Line2D([], [], color=ARCH_COLORS[label], marker="s", linestyle="", label=label)
        for label in ("NoLSTM", "NoAttention")
    ]
    forest_axes[-1].legend(handles=handles, loc="upper left")
    add_panel_labels(heat_axes + forest_axes, "abcdef")
    export_and_audit(fig, "fig_b_architecture_interaction", (7.2, 5.65))


def fig_c_mechanism_contrast() -> None:
    data = pd.read_csv(SOURCE / "fig_c_mechanism_contrast_fold.csv")
    mechanisms = list(MECHANISM_COLORS)
    limit = symmetric_limit(data.delta_perturb_minus_baseline, 2.0)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.85), constrained_layout=True, sharey=True)
    for ax, target in zip(axes, TARGETS):
        part = data[data.target == target]
        for xi, mechanism in enumerate(mechanisms):
            vals = part[part.mechanism == mechanism].delta_perturb_minus_baseline.to_numpy()
            jitter = np.linspace(-0.10, 0.10, len(vals))
            ax.scatter(
                xi + jitter,
                vals,
                s=16,
                facecolor="white",
                edgecolor=MECHANISM_COLORS[mechanism],
                linewidth=0.8,
                zorder=3,
            )
            ax.scatter(xi, vals.mean(), s=31, marker="D", color=MECHANISM_COLORS[mechanism], edgecolor="white", linewidth=0.5, zorder=4)
        ax.axhline(0, color="#777777", linestyle="--", linewidth=0.7)
        ax.set_xticks(range(len(mechanisms)))
        ax.set_xticklabels(["Max", "Att.\nshuf.", "Att.\nzero", "L50", "L100", "L200"], rotation=0, ha="center", fontsize=5.5)
        ax.set_ylim(-limit, limit)
        ax.set_title(TARGET_LABELS[target])
        ax.grid(axis="y", color="#E8E8E8", linewidth=0.5)
    axes[0].set_ylabel("Delta outcome (pp)")
    add_panel_labels(axes, "abc")
    export_and_audit(fig, "fig_c_mechanism_contrast", (7.2, 2.85))


def fig_d_random_null() -> None:
    contiguous = pd.read_csv(SOURCE / "fig_d_random_window_null.csv")
    shape_path = SOURCE / "fig_d_shape_matched_null.csv"
    data = pd.read_csv(shape_path) if shape_path.exists() else contiguous
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.10), constrained_layout=False)
    colors = [OKABE["blue"], OKABE["purple"], OKABE["orange"]]
    for ax, target, color in zip(axes, TARGETS, colors):
        part = data[data.target == target]
        values = part.random_delta.to_numpy()
        if np.nanstd(values) < 1e-10:
            ax.axvline(float(np.nanmean(values)), color=color, linewidth=4, alpha=0.45)
            ax.text(0.5, 0.55, "Degenerate null\n(sparse mapped coverage)", transform=ax.transAxes, ha="center", va="center")
        else:
            sns.histplot(values, bins=35, stat="density", color=color, alpha=0.38, edgecolor="white", linewidth=0.3, ax=ax)
            sns.kdeplot(values, color=color, linewidth=1.2, ax=ax)
            if shape_path.exists():
                reference = contiguous[contiguous.target == target].random_delta.to_numpy()
                if np.nanstd(reference) > 1e-10:
                    sns.kdeplot(reference, color="#777777", linestyle="--", linewidth=0.9, ax=ax)
        observed = part[["fold", "observed_maximum_delta", "empirical_one_sided_p"]].drop_duplicates()
        for i, row in observed.reset_index(drop=True).iterrows():
            ax.axvline(row.observed_maximum_delta, color=OKABE["vermillion"], linewidth=0.7, alpha=0.6)
            ax.scatter(row.observed_maximum_delta, ax.get_ylim()[1] * (0.90 - i * 0.08), s=18, marker="v", color=OKABE["vermillion"], zorder=4)
        ax.axvline(observed.observed_maximum_delta.mean(), color=OKABE["vermillion"], linewidth=1.6)
        p_text = ", ".join(f"{p:.3f}" for p in observed.empirical_one_sided_p)
        ax.set_title(TARGET_LABELS[target])
        ax.set_xlabel("Random-window delta (pp)")
        ax.set_ylabel("Density" if target == 2 else "")
        prefix = "Shape-matched p" if shape_path.exists() else "Foldwise empirical p"
        ax.text(0.03, 0.97, f"{prefix}: {p_text}", transform=ax.transAxes, ha="left", va="top", fontsize=5.5)
    if shape_path.exists():
        handles = [
            mpl.lines.Line2D([], [], color=colors[0], linewidth=1.2, label="Exact-mask circular shift"),
            mpl.lines.Line2D([], [], color="#777777", linewidth=0.9, linestyle="--", label="Matched-count contiguous"),
            mpl.lines.Line2D([], [], color=OKABE["vermillion"], linewidth=1.5, label="Observed maximum"),
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=3,
            fontsize=5.5,
        )
    add_panel_labels(axes, "abc")
    fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=1.0)
    export_and_audit(fig, "fig_d_random_window_null", (7.2, 3.10))


def fig_e_l3_drop_keep() -> None:
    data = pd.read_csv(SOURCE / "fig_e_l3_drop_keep_fold.csv")
    order = ["baseline", "maximum", "keep_only_mean"]
    labels = ["Baseline", "Drop L3\n(maximum)", "Keep L3 only\n(mean elsewhere)"]
    fig, ax = plt.subplots(figsize=(3.5, 2.7), constrained_layout=True)
    colors = [ARCH_COLORS["Full"], OKABE["vermillion"], OKABE["purple"]]
    for _, fold_data in data.groupby("fold"):
        values = [float(fold_data[fold_data.condition == condition].c2.iloc[0]) for condition in order]
        ax.plot(range(3), values, color="#BDBDBD", linewidth=0.7, zorder=1)
        ax.scatter(range(3), values, s=20, facecolor="white", edgecolor=colors, linewidth=0.8, zorder=3)
    means = [float(data[data.condition == condition].c2.mean()) for condition in order]
    ax.scatter(range(3), means, s=38, marker="D", color=colors, edgecolor="white", linewidth=0.5, zorder=4)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Hesitating recall (%)")
    ax.set_ylim(0, 100)
    ax.set_title("L3 necessity/sufficiency diagnostic")
    ax.text(0.02, 0.03, "Caution: very sparse L3 mapped-window coverage", transform=ax.transAxes, fontsize=5.7, color="#555555")
    ax.grid(axis="y", color="#E8E8E8", linewidth=0.5)
    export_and_audit(fig, "fig_e_l3_drop_keep", (3.5, 2.7))


def build_table_b() -> pd.DataFrame:
    sig = pd.read_csv(SOURCE / "significant_timepoints_used.csv")
    coverage = pd.read_csv(SOURCE / "mapping_coverage.csv")
    max_data = pd.read_csv(SOURCE / "fig_a_window_behavior_delta_recall.csv")
    interactions = pd.read_csv(TABLES / "table_a_interaction_summary.csv")
    rows = []
    for target in TARGETS:
        label = f"L{target}"
        timepoints = sig[sig.label == target]
        cov = coverage[coverage.target == target]
        if target == 2:
            behavior_sensitivity = max_data[(max_data.window == label) & (max_data.behavior.isin(["Wrong Hole Exploration", "Changing Direction", "Walking"]))].groupby("fold").delta_perturb_minus_baseline.mean()
            interpretation = "Candidate shared Stage-2 temporal information"
        elif target == 3:
            behavior_sensitivity = max_data[(max_data.window == label) & (max_data.behavior == "Hesitating")].set_index("fold").delta_perturb_minus_baseline
            interpretation = "Inconclusive: weak fitted-model effect but severely limited mapped coverage"
        else:
            behavior_sensitivity = max_data[(max_data.window == label) & (max_data.behavior == "Walking")].set_index("fold").delta_perturb_minus_baseline
            interpretation = "Candidate Walking-dominant content and temporal-allocation dependence"
        target_interaction = interactions[interactions.target == target]
        strongest = target_interaction.iloc[target_interaction.interaction_mean_pp.abs().argmax()]
        rows.append(
            {
                "window": label,
                "n_uncorrected_significant_timepoints": int(len(timepoints)),
                "minimum_uncorrected_p": float(timepoints.p_value.min()),
                "mean_behavior_delta_pp": float(behavior_sensitivity.mean()),
                "behavior_delta_sd_pp": float(behavior_sensitivity.std(ddof=1)),
                "mapped_centers_run1": int(cov[cov.run == "run1"].center_occurrences.iloc[0]),
                "mapped_centers_run2": int(cov[cov.run == "run2"].center_occurrences.iloc[0]),
                "strongest_descriptive_interaction": f"{strongest.condition}; {strongest.comparison}",
                "interaction_mean_pp": float(strongest.interaction_mean_pp),
                "interaction_exact_p": float(strongest.exact_sign_flip_p),
                "interaction_bh_fdr_q": float(strongest.bh_fdr_q),
                "interpretation_status": interpretation,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "table_b_window_conclusions.csv", index=False, encoding="utf-8-sig")
    return table


def write_results_report(table_b: pd.DataFrame) -> None:
    architecture = pd.read_csv(SOURCE / "fig_b_architecture_perturbation_fold.csv")
    shape_table = TABLES / "shape_matched_random_window_empirical_p.csv"
    empirical = pd.read_csv(shape_table if shape_table.exists() else TABLES / "random_window_empirical_p.csv")
    lines = [
        "# Results summary",
        "",
        "All deltas are perturbation minus baseline in percentage points. Negative values indicate worse performance.",
        "",
        "## Window-level findings",
        "",
    ]
    for _, row in table_b.iterrows():
        lines.append(
            f"- **{row.window}:** mean target-outcome delta {row.mean_behavior_delta_pp:.2f} +/- "
            f"{row.behavior_delta_sd_pp:.2f} pp across three folds; mapped centers "
            f"run1/run2 = {int(row.mapped_centers_run1)}/{int(row.mapped_centers_run2)}. "
            f"{row.interpretation_status}."
        )
    lines += [
        "",
        "## Random-window specificity",
        "",
    ]
    for target in TARGETS:
        part = empirical[empirical.target == target]
        lines.append(
            f"- L{target}: foldwise one-sided empirical p = "
            + ", ".join(f"{value:.4f}" for value in part.empirical_one_sided_p)
            + "."
        )
    lines += [
        "",
        "## Mandatory limitations",
        "",
        "- n=3 folds makes interaction tests low-powered; exact two-sided sign-flip p cannot be smaller than 0.25.",
        "- Legacy overlapping-window, window-level splits limit train/test independence.",
        "- The legacy checkpoint protocol selects the best epoch on the held-out fold and therefore gives optimistic absolute performance.",
        "- L3 coverage is exceptionally sparse, so a near-zero perturbation effect is not evidence that L3 contains no behavioral information.",
        "- Pointwise group p values are uncorrected in the supplied table and are not equivalent to model-decisive evidence.",
    ]
    (ROOT / "RESULTS_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_file_checks() -> None:
    records = []
    for path in sorted(FIGURES.iterdir()):
        if path.suffix.lower() not in {".svg", ".pdf", ".png", ".tiff"} or path.name.endswith("_grayscale.png"):
            continue
        try:
            issues, info = check_figure(str(path), min_dpi=300)
            records.append(
                {
                    "file": path.name,
                    "issues": [{"level": level, "message": message} for level, message in issues],
                    "info": info,
                }
            )
        except Exception as exc:
            records.append({"file": path.name, "error": repr(exc)})
    (QA / "figure_file_checks.json").write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")


def write_data_profiles() -> None:
    jobs = [
        (
            SOURCE / "fig_a_window_behavior_delta_recall.csv",
            ["window", "behavior", "fold"],
            QA / "data_profile_fig_a.md",
        ),
        (
            SOURCE / "fig_b_architecture_perturbation_fold.csv",
            ["architecture", "condition", "target", "fold"],
            QA / "data_profile_fig_b.md",
        ),
        (
            SOURCE / "fig_d_random_window_null.csv",
            ["target", "fold"],
            QA / "data_profile_fig_d.md",
        ),
    ]
    profiler = SKILL_SCRIPTS / "profile_data.py"
    for source, groups, destination in jobs:
        command = [sys.executable, str(profiler), str(source)]
        for group in groups:
            command.extend(["--group", group])
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
        text = completed.stdout.replace(str(source), f"source_data/{source.name}")
        if completed.stderr:
            text += "\n\n## stderr\n\n" + completed.stderr
        destination.write_text(text, encoding="utf-8")


def main() -> None:
    apply_style()
    required = [
        SOURCE / "fig_a_window_behavior_delta_recall.csv",
        SOURCE / "fig_b_architecture_perturbation_fold.csv",
        SOURCE / "fig_c_mechanism_contrast_fold.csv",
        SOURCE / "fig_d_random_window_null.csv",
        SOURCE / "fig_e_l3_drop_keep_fold.csv",
        TABLES / "table_a_interaction_summary.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Run run_experiments.py first. Missing:\n" + "\n".join(missing))
    table_b = build_table_b()
    fig_a_window_behavior()
    fig_b_architecture_interaction()
    fig_c_mechanism_contrast()
    fig_d_random_null()
    fig_e_l3_drop_keep()
    write_results_report(table_b)
    run_file_checks()
    write_data_profiles()
    print("Figure suite complete.")


if __name__ == "__main__":
    main()
