"""Inference-only paired prediction audit for the legacy hierarchical model.

This script does not train or modify any model.  It reuses the frozen three-fold
Run1/Run2 checkpoints and writes sample-aligned baseline/intervention predictions.
Because Run1 and Run2 are built from different window sets, all transition counts,
McNemar tests, and confusion matrices are reported per stage.  The historical
five-class table is reproduced only through its legacy metric-merging convention.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import run_experiments as core


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source_data"
TABLES = ROOT / "tables"
QA = ROOT / "qa"

STAGE_LABELS: Mapping[str, Mapping[int, str]] = {
    "run1": {
        0: "L1 Hole exploration",
        1: "Gate: L2/L4/L5",
        2: "L3 Hesitating",
    },
    "run2": {
        0: "L2 Wrong hole",
        1: "L4 Changing direction",
        2: "L5 Walking",
    },
}

MASK_SPECS = [
    *[
        {
            "comparison_id": f"mask_max_L{target}",
            "comparison_family": "input_mask",
            "target": target,
            "condition": "maximum",
            "label": f"Maximum mask L{target}",
        }
        for target in range(1, 6)
    ],
    *[
        {
            "comparison_id": f"attention_zero_L{target}",
            "comparison_family": "attention_intervention",
            "target": target,
            "condition": "attention_zero",
            "label": f"Attention zero L{target}",
        }
        for target in (2, 3, 5)
    ],
    *[
        {
            "comparison_id": f"attention_shuffle_L{target}",
            "comparison_family": "attention_intervention",
            "target": target,
            "condition": "attention_shuffle",
            "label": f"Attention shuffle L{target}",
        }
        for target in (2, 3, 5)
    ],
]

STRUCTURE_SPECS = [
    {
        "comparison_id": f"structure_{architecture}",
        "comparison_family": "structural_ablation",
        "architecture": architecture,
        "label": f"Full vs {core.ARCH_LABELS[architecture]}",
    }
    for architecture in ("no_conv", "no_lstm", "no_attention")
]


def bh_adjust(values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(values), dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_fold_partition(stage: str, n_samples: int) -> Dict:
    fold_indices = {fold: core.split_indices(stage, fold)[1] for fold in core.FOLDS}
    flat = [index for fold in core.FOLDS for index in fold_indices[fold]]
    counts = pd.Series(flat).value_counts()
    return {
        "stage": stage,
        "dataset_windows": int(n_samples),
        "test_rows_across_folds": int(len(flat)),
        "unique_test_indices": int(len(set(flat))),
        "duplicate_test_indices": int((counts > 1).sum()),
        "missing_test_indices": int(len(set(range(n_samples)).difference(flat))),
        "is_exact_partition": bool(len(flat) == n_samples and len(set(flat)) == n_samples),
    }


def prediction_frame(
    *,
    stage: str,
    fold: str,
    indices: List[int],
    baseline: Dict[str, np.ndarray],
    intervention: Dict[str, np.ndarray],
    comparison_id: str,
    comparison_label: str,
    comparison_family: str,
    condition: str,
    baseline_architecture: str,
    intervention_architecture: str,
    target: Optional[int],
    affected: Optional[np.ndarray],
) -> pd.DataFrame:
    y_true = baseline["y_true"].astype(int)
    other_true = intervention["y_true"].astype(int)
    if not np.array_equal(y_true, other_true):
        raise AssertionError(f"Unpaired labels for {comparison_id} {stage} {fold}")
    base_pred = baseline["probability"].argmax(axis=1).astype(int)
    int_pred = intervention["probability"].argmax(axis=1).astype(int)
    row = np.arange(len(y_true))
    base_correct = base_pred == y_true
    int_correct = int_pred == y_true
    transition = np.select(
        [base_correct & int_correct, base_correct & ~int_correct, ~base_correct & int_correct],
        ["CC", "CW", "WC"],
        default="WW",
    )
    labels = STAGE_LABELS[stage]
    if affected is None:
        affected_values = pd.array([pd.NA] * len(y_true), dtype="boolean")
    else:
        affected_values = pd.array(np.asarray(affected, dtype=bool), dtype="boolean")
    return pd.DataFrame(
        {
            "stage": stage,
            "fold": fold,
            "dataset_index": np.asarray(indices, dtype=int),
            "sample_id": [f"{stage}:{index:04d}" for index in indices],
            "comparison_id": comparison_id,
            "comparison_label": comparison_label,
            "comparison_family": comparison_family,
            "target_label": target,
            "intervention_condition": condition,
            "baseline_architecture": baseline_architecture,
            "intervention_architecture": intervention_architecture,
            "y_true": y_true,
            "y_true_label": [labels[int(value)] for value in y_true],
            "y_pred_baseline": base_pred,
            "y_pred_baseline_label": [labels[int(value)] for value in base_pred],
            "y_pred_intervention": int_pred,
            "y_pred_intervention_label": [labels[int(value)] for value in int_pred],
            "baseline_correct": base_correct,
            "intervention_correct": int_correct,
            "transition": transition,
            "affected_by_mask": affected_values,
            "baseline_true_probability": baseline["probability"][row, y_true],
            "intervention_true_probability": intervention["probability"][row, y_true],
            "delta_true_probability": (
                intervention["probability"][row, y_true] - baseline["probability"][row, y_true]
            ),
        }
    )


def transition_row(part: pd.DataFrame, class_id: int, scope: str) -> Dict:
    counts = part["transition"].value_counts().reindex(["CC", "CW", "WC", "WW"], fill_value=0)
    cc, cw, wc, ww = (int(counts[key]) for key in ("CC", "CW", "WC", "WW"))
    n = int(len(part))
    base_correct_n = cc + cw
    base_wrong_n = wc + ww
    discordant = cw + wc
    p_value = float(binomtest(cw, n=discordant, p=0.5).pvalue) if discordant else 1.0
    return {
        "scope": scope,
        "class_id": int(class_id),
        "class_label": "All stage samples" if class_id < 0 else str(part["y_true_label"].iloc[0]),
        "n": n,
        "CC": cc,
        "CW": cw,
        "WC": wc,
        "WW": ww,
        "baseline_correct_n": base_correct_n,
        "baseline_wrong_n": base_wrong_n,
        "damage_rate_percent": 100.0 * cw / base_correct_n if base_correct_n else np.nan,
        "recovery_rate_percent": 100.0 * wc / base_wrong_n if base_wrong_n else np.nan,
        "net_correct_change": wc - cw,
        "baseline_accuracy_percent": 100.0 * base_correct_n / n if n else np.nan,
        "intervention_accuracy_percent": 100.0 * (cc + wc) / n if n else np.nan,
        "accuracy_delta_pp": 100.0 * (wc - cw) / n if n else np.nan,
        "discordant_n": discordant,
        "mcnemar_exact_p": p_value,
    }


def summarize_transitions(predictions: pd.DataFrame, by_fold: bool) -> pd.DataFrame:
    group_cols = [
        "comparison_id",
        "comparison_label",
        "comparison_family",
        "target_label",
        "intervention_condition",
        "baseline_architecture",
        "intervention_architecture",
        "stage",
    ]
    if by_fold:
        group_cols.append("fold")
    rows: List[Dict] = []
    for keys, group in predictions.groupby(group_cols, dropna=False, sort=False):
        base = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        rows.append({**base, **transition_row(group, -1, "overall")})
        for class_id, class_part in group.groupby("y_true", sort=True):
            rows.append({**base, **transition_row(class_part, int(class_id), "true_class")})
    out = pd.DataFrame(rows)
    out["mcnemar_bh_q"] = bh_adjust(out["mcnemar_exact_p"].fillna(1.0))
    return out


def build_confusion_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    keys = ["comparison_id", "comparison_label", "comparison_family", "stage"]
    for group_key, group in predictions.groupby(keys, sort=False):
        base = dict(zip(keys, group_key))
        n_classes = len(STAGE_LABELS[str(base["stage"])])
        for true_class in range(n_classes):
            selected = group["y_true"] == true_class
            denominator = int(selected.sum())
            for predicted_class in range(n_classes):
                baseline_count = int((selected & (group["y_pred_baseline"] == predicted_class)).sum())
                intervention_count = int((selected & (group["y_pred_intervention"] == predicted_class)).sum())
                rows.append(
                    {
                        **base,
                        "true_class": true_class,
                        "true_label": STAGE_LABELS[str(base["stage"])][true_class],
                        "predicted_class": predicted_class,
                        "predicted_label": STAGE_LABELS[str(base["stage"])][predicted_class],
                        "row_n": denominator,
                        "baseline_count": baseline_count,
                        "intervention_count": intervention_count,
                        "baseline_row_percent": 100.0 * baseline_count / denominator if denominator else np.nan,
                        "intervention_row_percent": 100.0 * intervention_count / denominator if denominator else np.nan,
                        "delta_row_pp": (
                            100.0 * (intervention_count - baseline_count) / denominator
                            if denominator
                            else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def build_destination_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    keys = ["comparison_id", "comparison_label", "comparison_family", "stage"]
    for group_key, group in predictions.groupby(keys, sort=False):
        base = dict(zip(keys, group_key))
        labels = STAGE_LABELS[str(base["stage"])]
        for true_class, class_part in group.groupby("y_true", sort=True):
            damaged = class_part[class_part["transition"] == "CW"]
            recovered = class_part[class_part["transition"] == "WC"]
            for other_class, count in damaged["y_pred_intervention"].value_counts().sort_index().items():
                rows.append(
                    {
                        **base,
                        "true_class": int(true_class),
                        "true_label": labels[int(true_class)],
                        "transition": "CW_damage",
                        "other_class": int(other_class),
                        "other_label": labels[int(other_class)],
                        "count": int(count),
                        "within_transition_percent": 100.0 * int(count) / max(len(damaged), 1),
                    }
                )
            for other_class, count in recovered["y_pred_baseline"].value_counts().sort_index().items():
                rows.append(
                    {
                        **base,
                        "true_class": int(true_class),
                        "true_label": labels[int(true_class)],
                        "transition": "WC_recovery",
                        "other_class": int(other_class),
                        "other_label": labels[int(other_class)],
                        "count": int(count),
                        "within_transition_percent": 100.0 * int(count) / max(len(recovered), 1),
                    }
                )
    return pd.DataFrame(rows)


def metric_pack(arrays: Dict[str, np.ndarray], n_classes: int) -> Dict:
    return core.pack_from_arrays(
        arrays["y_true"], arrays["probability"], arrays["logits"], n_classes
    )


def compare_saved_metrics(recomputed: List[Dict]) -> pd.DataFrame:
    saved_path = SOURCE / "fig_b_architecture_perturbation_fold.csv"
    if not saved_path.exists():
        return pd.DataFrame(recomputed)
    saved = pd.read_csv(saved_path)
    rows: List[Dict] = []
    metric_keys = ["accuracy", "recall", "f1", "c0", "c1", "c2", "c3", "c4"]
    for row in recomputed:
        match = saved[
            (saved["architecture"] == row["architecture"])
            & (saved["fold"] == row["fold"])
            & (saved["condition"] == row["condition"])
            & (saved["target"] == row["target"])
        ]
        if match.empty:
            continue
        saved_row = match.iloc[0]
        for metric in metric_keys:
            rows.append(
                {
                    "architecture": row["architecture"],
                    "fold": row["fold"],
                    "condition": row["condition"],
                    "target": row["target"],
                    "metric": metric,
                    "recomputed": float(row[metric]),
                    "saved": float(saved_row[metric]),
                    "absolute_difference": abs(float(row[metric]) - float(saved_row[metric])),
                }
            )
    return pd.DataFrame(rows)


def checkpoint_manifest() -> pd.DataFrame:
    rows: List[Dict] = []
    for architecture in core.ARCH_FLAGS:
        for stage in ("run1", "run2"):
            for fold in core.FOLDS:
                path = core.checkpoint_path(architecture, stage, fold)
                rows.append(
                    {
                        "architecture": architecture,
                        "stage": stage,
                        "fold": fold,
                        "path_relative_to_video": str(path.relative_to(ROOT.parents[1])),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    for directory in (SOURCE, TABLES, QA):
        directory.mkdir(parents=True, exist_ok=True)

    sig = core.read_significant_timepoints(core.XLSX_PATH)
    offsets = core.significant_offsets(sig)
    datasets = {
        "run1": core.LegacySegmentDataset(core.Config(), "run1", offsets, radius_bins=100),
        "run2": core.LegacySegmentDataset(core.Config3(), "run2", offsets, radius_bins=100),
    }
    configs = {"run1": core.Config(), "run2": core.Config3()}

    partition = pd.DataFrame(
        [check_fold_partition(stage, len(datasets[stage])) for stage in ("run1", "run2")]
    )
    if not partition["is_exact_partition"].all():
        raise AssertionError("Saved test folds are not an exact out-of-fold partition")

    prediction_parts: List[pd.DataFrame] = []
    recomputed: List[Dict] = []
    for fold_index, fold in enumerate(core.FOLDS):
        stage_baselines: Dict[str, Dict[str, np.ndarray]] = {}
        stage_interventions: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
        test_indices: Dict[str, List[int]] = {}

        for stage in ("run1", "run2"):
            dataset = datasets[stage]
            config = configs[stage]
            indices = core.split_indices(stage, fold)[1]
            test_indices[stage] = indices
            full_model = core.load_model("full", stage, fold, dataset, config)
            _, baseline = core.predict_stage(full_model, dataset, indices, config)
            stage_baselines[stage] = baseline

            for spec in MASK_SPECS:
                _, intervention = core.predict_stage(
                    full_model,
                    dataset,
                    indices,
                    config,
                    target=int(spec["target"]),
                    condition=str(spec["condition"]),
                    shuffle_repeats=20,
                    seed=20260913 + fold_index * 1000 + int(spec["target"]) * 100,
                )
                stage_interventions[(stage, str(spec["comparison_id"]))] = intervention
                affected = dataset.segment_masks[
                    np.asarray(indices, dtype=int), int(spec["target"]) - 1, :
                ].any(axis=1)
                prediction_parts.append(
                    prediction_frame(
                        stage=stage,
                        fold=fold,
                        indices=indices,
                        baseline=baseline,
                        intervention=intervention,
                        comparison_id=str(spec["comparison_id"]),
                        comparison_label=str(spec["label"]),
                        comparison_family=str(spec["comparison_family"]),
                        condition=str(spec["condition"]),
                        baseline_architecture="full",
                        intervention_architecture="full",
                        target=int(spec["target"]),
                        affected=affected,
                    )
                )

            for spec in STRUCTURE_SPECS:
                architecture = str(spec["architecture"])
                intervention_model = core.load_model(architecture, stage, fold, dataset, config)
                _, intervention = core.predict_stage(intervention_model, dataset, indices, config)
                stage_interventions[(stage, str(spec["comparison_id"]))] = intervention
                prediction_parts.append(
                    prediction_frame(
                        stage=stage,
                        fold=fold,
                        indices=indices,
                        baseline=baseline,
                        intervention=intervention,
                        comparison_id=str(spec["comparison_id"]),
                        comparison_label=str(spec["label"]),
                        comparison_family=str(spec["comparison_family"]),
                        condition="baseline",
                        baseline_architecture="full",
                        intervention_architecture=architecture,
                        target=None,
                        affected=None,
                    )
                )

        # Rebuild only combinations that already exist in fig_b and compare exactly.
        for architecture in core.ARCH_FLAGS:
            comparison_id = None if architecture == "full" else f"structure_{architecture}"
            packs = {}
            for stage in ("run1", "run2"):
                arrays = (
                    stage_baselines[stage]
                    if architecture == "full"
                    else stage_interventions[(stage, str(comparison_id))]
                )
                packs[stage] = metric_pack(arrays, configs[stage].num_classes)
            merged = core.merge_stage_packs(packs["run1"], packs["run2"])
            for target in core.TARGETS:
                recomputed.append(
                    {
                        "architecture": architecture,
                        "fold": fold,
                        "condition": "baseline",
                        "target": target,
                        **core.flatten_metric_row(merged),
                    }
                )

        for spec in MASK_SPECS:
            if int(spec["target"]) not in core.TARGETS:
                continue
            packs = {
                stage: metric_pack(
                    stage_interventions[(stage, str(spec["comparison_id"]))],
                    configs[stage].num_classes,
                )
                for stage in ("run1", "run2")
            }
            merged = core.merge_stage_packs(packs["run1"], packs["run2"])
            recomputed.append(
                {
                    "architecture": "full",
                    "fold": fold,
                    "condition": str(spec["condition"]),
                    "target": int(spec["target"]),
                    **core.flatten_metric_row(merged),
                }
            )

        print(f"[paired inference] {fold} complete", flush=True)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions.to_csv(
        SOURCE / "paired_predictions_stagewise.csv", index=False, encoding="utf-8-sig"
    )
    mask_rows = predictions[predictions["affected_by_mask"].notna()].copy()
    coverage = (
        mask_rows.groupby(
            ["comparison_id", "comparison_label", "comparison_family", "target_label", "stage"],
            as_index=False,
        )
        .agg(
            evaluated_windows=("sample_id", "size"),
            affected_windows=("affected_by_mask", "sum"),
        )
    )
    coverage["affected_windows"] = coverage["affected_windows"].astype(int)
    coverage["affected_percent"] = (
        100.0 * coverage["affected_windows"] / coverage["evaluated_windows"]
    )
    coverage.to_csv(
        TABLES / "table_g_mask_coverage_stagewise.csv", index=False, encoding="utf-8-sig"
    )
    summarize_transitions(predictions, by_fold=False).to_csv(
        TABLES / "table_c_transition_summary_pooled.csv", index=False, encoding="utf-8-sig"
    )
    summarize_transitions(predictions, by_fold=True).to_csv(
        TABLES / "table_c_transition_summary_by_fold.csv", index=False, encoding="utf-8-sig"
    )
    build_confusion_table(predictions).to_csv(
        SOURCE / "paired_confusion_matrices_stagewise.csv", index=False, encoding="utf-8-sig"
    )
    build_destination_table(predictions).to_csv(
        TABLES / "table_d_transition_destinations.csv", index=False, encoding="utf-8-sig"
    )
    comparison = compare_saved_metrics(recomputed)
    comparison.to_csv(
        QA / "recomputed_vs_saved_fig_b.csv", index=False, encoding="utf-8-sig"
    )
    checkpoint_manifest().to_csv(
        QA / "checkpoint_manifest_sha256.csv", index=False, encoding="utf-8-sig"
    )
    partition.to_csv(QA / "fold_partition_audit.csv", index=False, encoding="utf-8-sig")

    max_difference = float(comparison["absolute_difference"].max()) if not comparison.empty else np.nan
    qa_payload = {
        "analysis_type": "inference-only paired stage-wise transition audit",
        "python_environment_required": r"C:\Users\LDaC\.conda\envs\pytorch-GPU",
        "n_prediction_rows": int(len(predictions)),
        "n_unique_stage_samples": int(predictions["sample_id"].nunique()),
        "comparisons": int(predictions["comparison_id"].nunique()),
        "max_absolute_reproduction_difference": max_difference,
        "stage_partition_audit": partition.to_dict(orient="records"),
        "statistical_warning": (
            "McNemar exact p values are exploratory because legacy 5-s windows overlap by 50%; "
            "window rows are not independent biological replicates."
        ),
        "hierarchy_warning": (
            "Run1 and Run2 use different constructed window sets. No synthetic five-class "
            "sample-level confusion matrix is produced."
        ),
    }
    (QA / "paired_transition_run_metadata.json").write_text(
        json.dumps(qa_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(qa_payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
