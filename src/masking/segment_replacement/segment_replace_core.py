"""Frozen legacy 5 s model: replace significant-point +/-1 s segments.

This module preserves the original model-facing protocol:

* 500 bins per input window (5 s at 10 ms/bin)
* 250-bin step (50% overlap)
* stage-label-constant window selection
* saved three-fold test indices and frozen run1/run2 checkpoints
* the original run1/run2 metric-merging convention

For every workbook-listed 10-ms p<0.05 point, the point is mapped from the
true behavior onset back to the recording timeline.  Inside each existing
legacy 5-s window, [center-1 s, center+1 s) is replaced and clipped at the
window boundaries.  Replacement is calculated independently for every neural
channel from that channel's original full 5-s window: maximum, mean, or minimum.
No model is retrained.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset, Subset


ANALYSIS_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ANALYSIS_ROOT.parents[1]
REPO_ROOT = SOURCE_ROOT.parent
MODEL_SOURCE_DIR = SOURCE_ROOT / "dual_layer_model"
sys.path.insert(0, str(MODEL_SOURCE_DIR))

from config import Config, Config3  # noqa: E402
from model import ConvFeatureAttentionLSTM  # noqa: E402


FINAL_CLASS_NAMES = {
    0: "Hole Exploration",
    1: "Wrong Hole Exploration",
    2: "Hesitating",
    3: "Changing Direction",
    4: "Walking",
}
MODE_DIRS = {
    "max": ANALYSIS_ROOT / "01_maximum_replacement",
    "mean": ANALYSIS_ROOT / "02_mean_replacement",
    "min": ANALYSIS_ROOT / "03_minimum_replacement",
}
MODE_TITLES = {
    "max": "5-s window per-neuron maximum",
    "mean": "5-s window per-neuron mean",
    "min": "5-s window per-neuron minimum",
}


def natural_key(path: Path):
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", path.name)]


def mean_sd(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def mean_sd_text(values: Iterable[float]) -> str:
    mean, sd = mean_sd(values)
    return f"{mean:.2f} ± {sd:.2f}"


def read_significant_timepoints(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name="Summary_P_long")
    required = {"window_ms", "label", "label_name", "time_s", "p_value"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Summary_P_long is missing columns: {sorted(missing)}")

    sig = df[(df["window_ms"] == 10) & (df["p_value"] < 0.05)].copy()
    sig = sig[sig["label"].isin([1, 2, 3, 4, 5])]
    sig["time_s"] = sig["time_s"].astype(float).round(2)
    sig = sig[sig["time_s"].between(-1.0, 1.0, inclusive="both")]
    sig["offset_bin"] = np.rint(sig["time_s"] / 0.01).astype(int)
    sig = sig.sort_values(["label", "time_s"]).reset_index(drop=True)
    if sig.empty:
        raise ValueError("No 10-ms p<0.05 timepoints were found within -1 to +1 s.")
    return sig


def significant_offsets(sig: pd.DataFrame) -> Dict[int, List[int]]:
    return {
        label: sorted(sig.loc[sig["label"] == label, "offset_bin"].astype(int).unique().tolist())
        for label in range(1, 6)
    }


def discover_neural_columns(files: List[Path]) -> List[str]:
    columns = set()
    for path in files:
        header = pd.read_csv(path, nrows=0).columns
        columns.update(col for col in header if re.fullmatch(r"N\d+", str(col)))
    if not columns:
        raise ValueError("No N1, N2, ... neural columns were found.")
    return sorted(columns, key=lambda col: int(col[1:]))


def expand_centers_to_segments(center_flags: np.ndarray, radius_bins: int) -> Tuple[np.ndarray, Dict[int, int]]:
    """Expand each center to [center-radius, center+radius), clipped per window."""
    if center_flags.ndim != 3:
        raise ValueError("center_flags must have shape window x label x time")
    n_windows, n_labels, window_size = center_flags.shape
    segments = np.zeros_like(center_flags, dtype=bool)
    clipped_by_label = {label: 0 for label in range(1, n_labels + 1)}
    for window_index in range(n_windows):
        for label_index in range(n_labels):
            for center in np.flatnonzero(center_flags[window_index, label_index]):
                start = max(0, int(center) - radius_bins)
                stop = min(window_size, int(center) + radius_bins)
                if start > int(center) - radius_bins or stop < int(center) + radius_bins:
                    clipped_by_label[label_index + 1] += 1
                segments[window_index, label_index, start:stop] = True
    return segments, clipped_by_label


class LegacySegmentDataset(Dataset):
    """Exact legacy windows plus mapped significant centers and +/-1 s masks."""

    def __init__(self, config, stage: str, offsets_by_label: Dict[int, List[int]], radius_bins: int = 100):
        self.config = config
        self.stage = stage
        self.radius_bins = int(radius_bins)
        files = sorted(Path(config.data_dir).glob("*_aligned.csv.gz"), key=natural_key)
        if not files:
            raise FileNotFoundError(f"No *_aligned.csv.gz files found in {config.data_dir}")
        neural_columns = discover_neural_columns(files)

        all_features: List[np.ndarray] = []
        all_final_labels: List[np.ndarray] = []
        all_center_flags: List[np.ndarray] = []
        all_recording_ids: List[np.ndarray] = []
        all_time_bins: List[np.ndarray] = []

        requested = {label: 0 for label in range(1, 6)}
        mapped_kept = {label: 0 for label in range(1, 6)}
        removed_label0 = {label: 0 for label in range(1, 6)}
        outside_recording = {label: 0 for label in range(1, 6)}
        onset_counts = {label: 0 for label in range(1, 6)}

        for recording_id, path in enumerate(files):
            df = pd.read_csv(path)
            for column in neural_columns:
                if column not in df.columns:
                    df[column] = 0.0
            df = df.sort_values("time_bin").reset_index(drop=True)

            raw_labels = df["label"].astype(int).to_numpy()
            time_bins = df["time_bin"].astype(int).to_numpy()
            time_bin_to_row = {int(value): index for index, value in enumerate(time_bins)}
            valid = np.isin(raw_labels, list(config.valid_raw_labels))
            flags = np.zeros((len(df), 5), dtype=bool)

            for raw_label in range(1, 6):
                offsets = offsets_by_label.get(raw_label, [])
                is_label = raw_labels == raw_label
                previous_same = np.r_[False, is_label[:-1]]
                onsets = np.flatnonzero(is_label & ~previous_same)
                onset_counts[raw_label] += int(len(onsets))
                for onset_row in onsets:
                    onset_bin = int(time_bins[onset_row])
                    for offset in offsets:
                        requested[raw_label] += 1
                        target_bin = onset_bin + int(offset)
                        target_row = time_bin_to_row.get(target_bin)
                        if target_row is None:
                            outside_recording[raw_label] += 1
                        elif not valid[target_row]:
                            removed_label0[raw_label] += 1
                        else:
                            flags[target_row, raw_label - 1] = True
                            mapped_kept[raw_label] += 1

            if not np.any(valid):
                continue
            final_labels = np.vectorize(config.raw_label_to_final_class.get)(raw_labels[valid]).astype(np.int64)
            features = df.loc[valid, neural_columns].to_numpy(dtype=np.float32)
            all_features.append(features)
            all_final_labels.append(final_labels)
            all_center_flags.append(flags[valid])
            all_recording_ids.append(np.full(int(valid.sum()), recording_id, dtype=np.int16))
            all_time_bins.append(time_bins[valid].astype(np.int32))

        features = np.vstack(all_features)
        final_labels = np.concatenate(all_final_labels)
        center_flags = np.vstack(all_center_flags)
        recording_ids = np.concatenate(all_recording_ids)
        kept_time_bins = np.concatenate(all_time_bins)

        if stage == "run1":
            stage_labels = np.where(np.isin(final_labels, list(config.merged_stage_classes)), 1, final_labels)
        elif stage == "run2":
            keep_stage = np.isin(final_labels, list(config.merged_stage_classes))
            features = features[keep_stage]
            final_labels = final_labels[keep_stage]
            center_flags = center_flags[keep_stage]
            recording_ids = recording_ids[keep_stage]
            kept_time_bins = kept_time_bins[keep_stage]
            stage_labels = np.vectorize(config.run2_label_map.get)(final_labels).astype(np.int64)
        else:
            raise ValueError("stage must be 'run1' or 'run2'")

        windows: List[np.ndarray] = []
        labels: List[int] = []
        window_centers: List[np.ndarray] = []
        cross_recording_windows = 0
        discontinuous_windows = 0

        for start in range(0, len(stage_labels) - config.window_size + 1, config.step_size):
            stop = start + config.window_size
            segment_stage = stage_labels[start:stop]
            if not np.all(segment_stage == segment_stage[0]):
                continue
            windows.append(features[start:stop, :].T.astype(np.float32))
            labels.append(int(segment_stage[0]))
            window_centers.append(center_flags[start:stop, :].T)

            rec = recording_ids[start:stop]
            bins = kept_time_bins[start:stop]
            if np.any(rec != rec[0]):
                cross_recording_windows += 1
            same_recording = rec[1:] == rec[:-1]
            if np.any(same_recording & (bins[1:] - bins[:-1] != 1)):
                discontinuous_windows += 1

        self.x = np.stack(windows).astype(np.float32)
        self.y = np.asarray(labels, dtype=np.int64)
        self.center_flags = np.stack(window_centers).astype(bool)
        self.segment_masks, clipped_by_label = expand_centers_to_segments(self.center_flags, self.radius_bins)
        self.neural_columns = neural_columns
        self.mapping_stats = {
            "stage": stage,
            "n_windows": int(len(self.y)),
            "input_size": int(self.x.shape[1]),
            "window_size": int(self.x.shape[2]),
            "step_size": int(config.step_size),
            "segment_radius_bins": self.radius_bins,
            "segment_nominal_bins": int(2 * self.radius_bins),
            "cross_recording_windows": int(cross_recording_windows),
            "time_discontinuous_windows": int(discontinuous_windows),
            "onset_counts": onset_counts,
            "requested_significant_centers": requested,
            "mapped_to_model_rows_before_windowing": mapped_kept,
            "removed_label0_or_invalid": removed_label0,
            "outside_recording": outside_recording,
            "center_occurrences_across_all_windows": {
                label: int(self.center_flags[:, label - 1, :].sum()) for label in range(1, 6)
            },
            "segment_bins_across_all_windows": {
                label: int(self.segment_masks[:, label - 1, :].sum()) for label in range(1, 6)
            },
            "boundary_clipped_center_occurrences": clipped_by_label,
        }

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.x[index]),
            torch.tensor(int(self.y[index]), dtype=torch.long),
            torch.from_numpy(self.segment_masks[index]),
            int(index),
        )


def make_model(config, input_size: int):
    return ConvFeatureAttentionLSTM(
        input_size=input_size,
        hidden_size=config.hidden_size,
        attention_hidden_size=config.attention_hidden_size,
        num_classes=config.num_classes,
        conv_channels=config.conv_channels,
        kernel_size=config.kernel_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(config.device)


def load_state(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_fold_models(model_dir: Path, fold_tag: str, input_size: int):
    cfg1, cfg2 = Config(), Config3()
    paths = [
        model_dir / f"best_test_model_run1_{fold_tag}.pth",
        model_dir / f"best_test_model_run2_{fold_tag}.pth",
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing model file(s):\n" + "\n".join(missing))
    run1, run2 = make_model(cfg1, input_size), make_model(cfg2, input_size)
    run1.load_state_dict(load_state(paths[0], cfg1.device))
    run2.load_state_dict(load_state(paths[1], cfg2.device))
    run1.eval()
    run2.eval()
    return run1, run2


def load_test_indices(model_dir: Path, run_tag: str, fold_tag: str) -> List[int]:
    path = model_dir / f"fold_split_{run_tag}_{fold_tag}.json"
    with open(path, "r", encoding="utf-8") as handle:
        split = json.load(handle)
    return list(map(int, split["test_indices"]))


def load_saved_accuracy(model_dir: Path, run_tag: str, fold_tag: str) -> Optional[float]:
    path = model_dir / f"predict_results_{run_tag}_{fold_tag}__on_fold_valid.jsonl"
    if not path.exists():
        return None
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return float(json.loads(rows[-1])["test"]["accuracy"]) if rows else None


def compute_stage_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, num_classes: int):
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = 0.0
    try:
        aupr = average_precision_score(np.eye(num_classes)[y_true], y_prob, average="macro")
    except Exception:
        aupr = 0.0
    per_class = {}
    for class_id in range(num_classes):
        selected = y_true == class_id
        total = int(selected.sum())
        correct = int((y_pred[selected] == class_id).sum()) if total else 0
        per_class[f"class_{class_id}"] = {
            "acc": float(correct / total * 100.0) if total else 0.0,
            "correct": correct,
            "total": total,
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred) * 100.0),
        "per_class": per_class,
        "metrics": {
            "recall": float(recall * 100.0),
            "f1": float(f1 * 100.0),
            "mcc": float(mcc * 100.0),
            "auc": float(auc * 100.0),
            "aupr": float(aupr * 100.0),
        },
    }


def replace_segment(x: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    """Fill masked time bins using each channel's statistic from the original 5-s window."""
    if mode == "max":
        fill = x.amax(dim=2, keepdim=True)
    elif mode == "mean":
        fill = x.mean(dim=2, keepdim=True)
    elif mode == "min":
        fill = x.amin(dim=2, keepdim=True)
    else:
        raise ValueError(f"Unsupported replacement mode: {mode}")
    return torch.where(mask.unsqueeze(1), fill.expand_as(x), x)


@torch.no_grad()
def evaluate_stage(
    model,
    dataset,
    indices: List[int],
    config,
    target_raw_label: Optional[int],
    replacement_mode: Optional[str],
):
    loader = DataLoader(Subset(dataset, indices), batch_size=config.batch_size, shuffle=False)
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[np.ndarray] = []
    affected_windows = 0
    replaced_time_bins = 0
    for x, y, segment_masks, _ in loader:
        if target_raw_label is not None:
            mask = segment_masks[:, target_raw_label - 1, :].bool()
            affected_windows += int(mask.any(dim=1).sum().item())
            replaced_time_bins += int(mask.sum().item())
            x = replace_segment(x, mask, str(replacement_mode))
        logits = model(x.to(config.device))
        probability = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probability, dim=1)
        y_true.extend(y.numpy().astype(int).tolist())
        y_pred.extend(prediction.detach().cpu().numpy().astype(int).tolist())
        y_prob.append(probability.detach().cpu().numpy())
    true_array = np.asarray(y_true, dtype=int)
    pred_array = np.asarray(y_pred, dtype=int)
    probability_array = np.concatenate(y_prob, axis=0)
    pack = compute_stage_metrics(true_array, pred_array, probability_array, config.num_classes)
    pack["affected_windows"] = affected_windows
    pack["replaced_time_bins"] = replaced_time_bins
    pack["replaced_neural_values"] = int(replaced_time_bins * dataset.x.shape[1])
    return pack


def merge_training_style(run1_pack: Dict, run2_pack: Dict):
    def class_acc(pack, class_id):
        return float(pack["per_class"].get(f"class_{class_id}", {}).get("acc", 0.0))

    a0, a1, a2 = (class_acc(run1_pack, index) for index in range(3))
    b0, b1, b2 = (class_acc(run2_pack, index) for index in range(3))
    metrics1, metrics2 = run1_pack["metrics"], run2_pack["metrics"]
    return {
        "accuracy": (float(run1_pack["accuracy"]) + float(run2_pack["accuracy"])) / 2.0,
        "recall": (float(metrics1["recall"]) + float(metrics2["recall"])) / 2.0,
        "f1": (float(metrics1["f1"]) + float(metrics2["f1"])) / 2.0,
        "mcc": (float(metrics1["mcc"]) + float(metrics2["mcc"])) / 2.0,
        "auc": (float(metrics1["auc"]) + float(metrics2["auc"])) / 2.0,
        "aupr": (float(metrics1["aupr"]) + float(metrics2["aupr"])) / 2.0,
        "c0": a0,
        "c1": (a1 + b0) / 2.0,
        "c2": a2,
        "c3": (a1 + b1) / 2.0,
        "c4": (a1 + b2) / 2.0,
    }


def aggregate_rows(fold_rows: List[Dict]) -> List[Dict]:
    output = []
    for condition in ["baseline"] + [f"ablate_l{index}" for index in range(1, 6)]:
        rows = [row for row in fold_rows if row["condition"] == condition]
        target = None if condition == "baseline" else int(condition[-1])
        output.append(
            {
                "condition": condition,
                "display_name": "Our Proposed" if target is None else f"Ablate L{target}",
                "target_raw_label": target,
                "folds": rows,
                **{
                    f"fold_{key}": [float(row[key]) for row in rows]
                    for key in ["accuracy", "recall", "f1", "c0", "c1", "c2", "c3", "c4"]
                },
            }
        )
    return output


def build_summary_tables(output_dir: Path, aggregate: List[Dict]):
    overall = pd.DataFrame(
        [
            {
                "Method": row["display_name"],
                "Acc": mean_sd_text(row["fold_accuracy"]),
                "Recall": mean_sd_text(row["fold_recall"]),
                "F1": mean_sd_text(row["fold_f1"]),
            }
            for row in aggregate
        ]
    )
    classes = pd.DataFrame(
        [
            {
                "Method": row["display_name"],
                "Hole Exploration": mean_sd_text(row["fold_c0"]),
                "Wrong Hole Exploration": mean_sd_text(row["fold_c1"]),
                "Hesitating": mean_sd_text(row["fold_c2"]),
                "Changing Direction": mean_sd_text(row["fold_c3"]),
                "Walking": mean_sd_text(row["fold_c4"]),
            }
            for row in aggregate
        ]
    )
    overall.to_csv(output_dir / "table4_overall_mean_sd.csv", index=False, encoding="utf-8-sig")
    classes.to_csv(output_dir / "table5_class_accuracy_mean_sd.csv", index=False, encoding="utf-8-sig")

    baseline = aggregate[0]
    deltas = []
    for row in aggregate[1:]:
        values = {
            key: np.asarray(baseline[f"fold_{key}"]) - np.asarray(row[f"fold_{key}"])
            for key in ["accuracy", "recall", "f1", "c0", "c1", "c2", "c3", "c4"]
        }
        deltas.append(
            {
                "condition": row["display_name"],
                "target_raw_label": int(row["target_raw_label"]),
                **{f"drop_{key}_mean": float(array.mean()) for key, array in values.items()},
                **{f"drop_{key}_sd": float(array.std(ddof=1)) for key, array in values.items()},
            }
        )
    pd.DataFrame(deltas).to_csv(output_dir / "ablation_delta_vs_baseline.csv", index=False, encoding="utf-8-sig")
    return overall, classes


def write_mode_outputs(
    mode: str,
    output_dir: Path,
    sig: pd.DataFrame,
    ds1: LegacySegmentDataset,
    ds2: LegacySegmentDataset,
    fold_rows: List[Dict],
    aggregate: List[Dict],
    args,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(output_dir / "condition_fold_metrics.csv", index=False, encoding="utf-8-sig")
    sig.to_csv(output_dir / "significant_10ms_timepoints_used.csv", index=False, encoding="utf-8-sig")
    with open(output_dir / "legacy_5s_segment_replace_summary.json", "w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, ensure_ascii=False, indent=2)
    with open(output_dir / "dataset_mapping_qc.json", "w", encoding="utf-8") as handle:
        json.dump({"run1": ds1.mapping_stats, "run2": ds2.mapping_stats}, handle, ensure_ascii=False, indent=2)
    with open(output_dir / "analysis_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "python": sys.executable,
                "replacement_mode": mode,
                "replacement_definition": MODE_TITLES[mode],
                "statistic_source": "each neural channel's original full 5-s window",
                "segment_definition": "[significant_center-1s, significant_center+1s), clipped to the existing 5-s window",
                "segment_radius_bins": 100,
                "nominal_segment_bins": 200,
                "model_dir": str(args.model_dir),
                "xlsx_path": str(args.xlsx_path),
                "window_bins": 500,
                "window_seconds": 5.0,
                "step_bins": 250,
                "overlap_fraction": 0.5,
                "constant_stage_label_required": True,
                "significance_rule": "window_ms == 10 and p_value < 0.05 and -1 <= time_s <= 1",
                "retraining": False,
                "folds": args.folds,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    overall, classes = build_summary_tables(output_dir, aggregate)
    baseline = aggregate[0]
    lines = [
        f"# {MODE_TITLES[mode]} replacement results",
        "",
        "Protocol: frozen 5-s windows, 50% overlap, constant stage labels, saved three-fold indices, and original run1/run2 model merging.",
        "Each p<0.05 10-ms center was expanded to a clipped +/-1-s segment and replaced per neural channel.",
        "",
        "## Three-fold summary",
        "",
    ]
    for row in aggregate:
        lines.append(
            f"- {row['display_name']}: Acc {mean_sd_text(row['fold_accuracy'])}; "
            f"Recall {mean_sd_text(row['fold_recall'])}; F1 {mean_sd_text(row['fold_f1'])}."
        )
    lines.extend(["", "## Change versus baseline", ""])
    for row in aggregate[1:]:
        drop_acc = np.asarray(baseline["fold_accuracy"]) - np.asarray(row["fold_accuracy"])
        drop_recall = np.asarray(baseline["fold_recall"]) - np.asarray(row["fold_recall"])
        drop_f1 = np.asarray(baseline["fold_f1"]) - np.asarray(row["fold_f1"])
        lines.append(
            f"- {row['display_name']}: Acc drop {drop_acc.mean():.2f} pp; "
            f"Recall drop {drop_recall.mean():.2f} pp; F1 drop {drop_f1.mean():.2f} pp."
        )
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return overall, classes


def run_modes(args, modes: Sequence[str]):
    invalid = [mode for mode in modes if mode not in MODE_DIRS]
    if invalid:
        raise ValueError(f"Unsupported replacement modes: {invalid}")
    sig = read_significant_timepoints(args.xlsx_path)
    offsets = significant_offsets(sig)
    ds1 = LegacySegmentDataset(Config(), "run1", offsets, radius_bins=100)
    ds2 = LegacySegmentDataset(Config3(), "run2", offsets, radius_bins=100)
    if ds1.x.shape[2] != 500 or ds2.x.shape[2] != 500:
        raise AssertionError("Legacy model input is no longer 500 bins.")
    if Config.step_size != 250 or Config3.step_size != 250:
        raise AssertionError("Legacy step is no longer 250 bins (50% overlap).")

    rows_by_mode = {mode: [] for mode in modes}
    for fold_tag in args.folds:
        model1, model2 = load_fold_models(args.model_dir, fold_tag, ds1.x.shape[1])
        indices1 = load_test_indices(args.model_dir, "run1", fold_tag)
        indices2 = load_test_indices(args.model_dir, "run2", fold_tag)
        baseline1 = evaluate_stage(model1, ds1, indices1, Config(), None, None)
        baseline2 = evaluate_stage(model2, ds2, indices2, Config3(), None, None)
        saved1 = load_saved_accuracy(args.model_dir, "run1", fold_tag)
        saved2 = load_saved_accuracy(args.model_dir, "run2", fold_tag)
        if saved1 is not None and not np.isclose(baseline1["accuracy"], saved1, atol=1e-8):
            raise AssertionError(f"run1 {fold_tag} baseline mismatch: {baseline1['accuracy']} vs {saved1}")
        if saved2 is not None and not np.isclose(baseline2["accuracy"], saved2, atol=1e-8):
            raise AssertionError(f"run2 {fold_tag} baseline mismatch: {baseline2['accuracy']} vs {saved2}")
        baseline_merged = merge_training_style(baseline1, baseline2)

        for mode in modes:
            rows_by_mode[mode].append(
                {
                    "condition": "baseline",
                    "display_name": "Our Proposed",
                    "target_raw_label": None,
                    "fold": fold_tag,
                    **baseline_merged,
                    "run1_accuracy": float(baseline1["accuracy"]),
                    "run2_accuracy": float(baseline2["accuracy"]),
                    "run1_affected_windows": 0,
                    "run2_affected_windows": 0,
                    "run1_replaced_time_bins": 0,
                    "run2_replaced_time_bins": 0,
                }
            )
            for target in range(1, 6):
                pack1 = evaluate_stage(model1, ds1, indices1, Config(), target, mode)
                pack2 = evaluate_stage(model2, ds2, indices2, Config3(), target, mode)
                merged = merge_training_style(pack1, pack2)
                rows_by_mode[mode].append(
                    {
                        "condition": f"ablate_l{target}",
                        "display_name": f"Ablate L{target}",
                        "target_raw_label": target,
                        "fold": fold_tag,
                        **merged,
                        "run1_accuracy": float(pack1["accuracy"]),
                        "run2_accuracy": float(pack2["accuracy"]),
                        "run1_affected_windows": int(pack1["affected_windows"]),
                        "run2_affected_windows": int(pack2["affected_windows"]),
                        "run1_replaced_time_bins": int(pack1["replaced_time_bins"]),
                        "run2_replaced_time_bins": int(pack2["replaced_time_bins"]),
                        "run1_replaced_neural_values": int(pack1["replaced_neural_values"]),
                        "run2_replaced_neural_values": int(pack2["replaced_neural_values"]),
                    }
                )
        del model1, model2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for mode in modes:
        output_dir = MODE_DIRS[mode] / "outputs"
        aggregate = aggregate_rows(rows_by_mode[mode])
        overall, _ = write_mode_outputs(mode, output_dir, sig, ds1, ds2, rows_by_mode[mode], aggregate, args)
        print(f"\n[{mode}] {MODE_TITLES[mode]}")
        print(overall.to_string(index=False))
        print(f"Saved outputs to: {output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument(
        "--xlsx-path",
        type=Path,
        default=REPO_ROOT / "data" / "significant_timepoints.xlsx",
    )
    parser.add_argument("--folds", nargs="+", default=["fold01", "fold02", "fold03"])
    return parser


def main(selected_modes: Optional[Sequence[str]] = None):
    parser = build_parser()
    args = parser.parse_args()
    modes = list(selected_modes) if selected_modes is not None else ["max", "mean", "min"]
    run_modes(args, modes)


if __name__ == "__main__":
    main()
