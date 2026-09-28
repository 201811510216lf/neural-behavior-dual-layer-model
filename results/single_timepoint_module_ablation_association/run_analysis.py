"""Frozen-model single-timepoint x module-ablation association analysis.

Hard constraints preserved:
  * no training and no resplitting;
  * legacy 5 s windows (500 x 10 ms) with 250-bin step;
  * stage-label-constant selection and saved Run1/Run2 three-fold test indices;
  * point-wise masks map behavior onsets back to the recording timeline;
  * [center-1 s, center+1 s) replacement, clipped inside each legacy window;
  * each neural channel is filled with its own maximum from the original 5 s window.

The script creates stage-wise prediction/metric source data, cross-ablation
association tables, QA/provenance files, publication figures and a Chinese
results summary.  It never calls an optimizer or a training loop.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import sklearn
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset, Subset


SCRIPT_PATH = Path(__file__).resolve()
OUT_ROOT = SCRIPT_PATH.parent
VIDEO_ROOT = OUT_ROOT.parents[1]
PROJECT_ROOT = VIDEO_ROOT / "raster" / "brain_model_regroup_1246_vs_35_fixed"
LEGACY_ROOT = PROJECT_ROOT / "outputs_legacy"
RESULT_ROOT = VIDEO_ROOT / "raster" / "result"
SEGMENT_ROOT = PROJECT_ROOT / "legacy_5s_50pct_significant_segment_replace"
XLSX_PATH = (
    VIDEO_ROOT
    / "outputs"
    / "regroup_1246_vs_35_label_timepoint_ttest_bigtable"
    / "regroup_1246_vs_35_label_timepoint_ttest_bigtable.xlsx"
)

SOURCE_DIR = OUT_ROOT / "source_data"
TABLE_DIR = OUT_ROOT / "tables"
FIGURE_DIR = OUT_ROOT / "figures"
QA_DIR = OUT_ROOT / "qa"
for directory in (SOURCE_DIR, TABLE_DIR, FIGURE_DIR, QA_DIR):
    directory.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SEGMENT_ROOT))
sys.path.insert(0, str(RESULT_ROOT))

from config import Config, Config3  # noqa: E402
from model import ConvFeatureAttentionLSTM  # noqa: E402
from segment_replace_core import (  # noqa: E402
    LegacySegmentDataset,
    load_state,
    read_significant_timepoints,
    significant_offsets,
)


FOLDS = ("fold01", "fold02", "fold03")
STAGES = ("run1", "run2")
ARCHITECTURES = ("full", "no_conv", "no_lstm", "no_attention")
MODULES = ("no_conv", "no_lstm", "no_attention")
ARCH_LABELS = {
    "full": "Full",
    "no_conv": "NoConv",
    "no_lstm": "NoLSTM",
    "no_attention": "NoAttention",
}
ARCH_FLAGS = {
    "full": {},
    "no_conv": {"ablate_conv": True},
    "no_lstm": {"ablate_lstm": True},
    "no_attention": {"ablate_attention": True},
}
BEHAVIOR_NAMES = {
    1: "Hole Exploration",
    2: "Wrong Hole Exploration",
    3: "Hesitating",
    4: "Changing Direction",
    5: "Walking",
}
STAGE_CLASS_NAMES = {
    "run1": {0: "L1", 1: "L2/L4/L5 gate", 2: "L3"},
    "run2": {0: "L2", 1: "L4", 2: "L5"},
}
CANONICAL_BEHAVIOR_STAGE_CLASS = {
    1: ("run1", 0),
    2: ("run2", 0),
    3: ("run1", 2),
    4: ("run2", 1),
    5: ("run2", 2),
}
EXPECTED_POINTS = {
    1: [-0.69, -0.15],
    2: [-0.57, -0.39, 0.06, 0.46, 0.82],
    3: [-0.08, 0.42, 0.65, 0.95],
    4: [-0.56, -0.55],
    5: [0.00, 0.04, 0.05, 0.06, 0.07, 0.08, 0.12, 0.13, 0.33, 0.35, 0.70],
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def natural_key(path: Path) -> List[object]:
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", path.name)]


def time_token(value: float) -> str:
    sign = "p" if value >= 0 else "m"
    return sign + f"{abs(value):.2f}".replace(".", "p")


def display_time(value: float) -> str:
    return f"{value:+.2f}s"


def sign_symbol(value: float, tol: float = 0.0) -> str:
    if value > tol:
        return "+"
    if value < -tol:
        return "-"
    return "0"


def safe_corr(x: Sequence[float], y: Sequence[float], method: str) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    valid = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[valid], ya[valid]
    if len(xa) < 2 or np.ptp(xa) == 0 or np.ptp(ya) == 0:
        return float("nan")
    if method == "pearson":
        return float(pearsonr(xa, ya).statistic)
    if method == "spearman":
        return float(spearmanr(xa, ya).statistic)
    if method == "cosine":
        denom = float(np.linalg.norm(xa) * np.linalg.norm(ya))
        return float(np.dot(xa, ya) / denom) if denom > 0 else float("nan")
    raise ValueError(method)


def binary_phi(x: Sequence[bool], y: Sequence[bool]) -> float:
    xa = np.asarray(x, dtype=int)
    ya = np.asarray(y, dtype=int)
    if len(np.unique(xa)) < 2 or len(np.unique(ya)) < 2:
        return float("nan")
    return float(matthews_corrcoef(xa, ya))


def jaccard(x: Sequence[bool], y: Sequence[bool]) -> float:
    xa = np.asarray(x, dtype=bool)
    ya = np.asarray(y, dtype=bool)
    union = int(np.logical_or(xa, ya).sum())
    return float(np.logical_and(xa, ya).sum() / union) if union else float("nan")


def exact_condition_table(sig: pd.DataFrame) -> pd.DataFrame:
    points = sig[["label", "label_name", "time_s", "t_stat", "p_value"]].drop_duplicates().copy()
    points = points.sort_values(["label", "time_s"]).reset_index(drop=True)
    observed = {
        int(label): group["time_s"].astype(float).round(2).tolist()
        for label, group in points.groupby("label", sort=True)
    }
    if len(points) != 24 or observed != EXPECTED_POINTS:
        qa = {
            "status": "ERROR",
            "reason": "The significant-point gate did not yield the required 24 points.",
            "expected": EXPECTED_POINTS,
            "observed": observed,
            "n_observed": int(len(points)),
        }
        (QA_DIR / "QA_ERROR_SIGNIFICANT_POINTS.json").write_text(
            json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        raise RuntimeError("QA ERROR: expected exactly the specified 24 significant timepoints")
    points.insert(0, "condition_index", np.arange(len(points), dtype=int))
    points.insert(
        1,
        "condition_id",
        [f"tp{i + 1:02d}_L{int(row.label)}_{time_token(float(row.time_s))}" for i, row in points.iterrows()],
    )
    points["offset_bin"] = np.rint(points["time_s"].astype(float) / 0.01).astype(int)
    points["display_label"] = [
        f"L{int(row.label)} {display_time(float(row.time_s))}" for _, row in points.iterrows()
    ]
    points.to_csv(SOURCE_DIR / "significant_timepoints_24.csv", index=False, encoding="utf-8-sig")
    return points


def discover_neural_columns(files: Sequence[Path]) -> List[str]:
    columns: set[str] = set()
    for path in files:
        header = pd.read_csv(path, nrows=0).columns
        columns.update(str(col) for col in header if re.fullmatch(r"N\d+", str(col)))
    if not columns:
        raise ValueError("No N1, N2, ... neural columns were found")
    return sorted(columns, key=lambda col: int(col[1:]))


@dataclass
class WindowMeta:
    recording_id: int
    recording_file: str
    recording_key: str
    recording_mixed: bool
    recording_ids_json: str
    start_time_bin: int
    end_time_bin: int
    discontinuous: bool


class PointwiseLegacyDataset(Dataset):
    """Exact legacy dataset with a separate mapped +/-1 s mask per timepoint."""

    def __init__(self, config, stage: str, conditions: pd.DataFrame, radius_bins: int = 100):
        self.config = config
        self.stage = stage
        self.radius_bins = int(radius_bins)
        self.conditions = conditions.reset_index(drop=True).copy()
        files = sorted(Path(config.data_dir).glob("*_aligned.csv.gz"), key=natural_key)
        if not files:
            raise FileNotFoundError(f"No *_aligned.csv.gz files found in {config.data_dir}")
        neural_columns = discover_neural_columns(files)

        all_features: List[np.ndarray] = []
        all_final_labels: List[np.ndarray] = []
        all_flags: List[np.ndarray] = []
        all_recording_ids: List[np.ndarray] = []
        all_time_bins: List[np.ndarray] = []
        file_names = [path.name for path in files]
        mapping_rows: List[Dict[str, object]] = []

        for recording_id, path in enumerate(files):
            df = pd.read_csv(path).sort_values("time_bin").reset_index(drop=True)
            for column in neural_columns:
                if column not in df.columns:
                    df[column] = 0.0
            raw_labels = df["label"].astype(int).to_numpy()
            time_bins = df["time_bin"].astype(int).to_numpy()
            time_bin_to_row = {int(value): index for index, value in enumerate(time_bins)}
            valid = np.isin(raw_labels, list(config.valid_raw_labels))
            flags = np.zeros((len(df), len(self.conditions)), dtype=bool)

            for cond_i, cond in self.conditions.iterrows():
                raw_label = int(cond["label"])
                offset = int(cond["offset_bin"])
                is_label = raw_labels == raw_label
                previous_same = np.r_[False, is_label[:-1]]
                onsets = np.flatnonzero(is_label & ~previous_same)
                requested = mapped = removed = outside = 0
                for onset_row in onsets:
                    requested += 1
                    target_bin = int(time_bins[onset_row]) + offset
                    target_row = time_bin_to_row.get(target_bin)
                    if target_row is None:
                        outside += 1
                    elif not valid[target_row]:
                        removed += 1
                    else:
                        flags[target_row, cond_i] = True
                        mapped += 1
                mapping_rows.append(
                    {
                        "stage": stage,
                        "recording_id": recording_id,
                        "recording_file": path.name,
                        "condition_id": cond["condition_id"],
                        "label": raw_label,
                        "time_s": float(cond["time_s"]),
                        "onset_count": int(len(onsets)),
                        "requested_centers": requested,
                        "mapped_valid_centers": mapped,
                        "removed_label0_or_invalid": removed,
                        "outside_recording": outside,
                    }
                )

            if not np.any(valid):
                continue
            final_labels = np.vectorize(config.raw_label_to_final_class.get)(raw_labels[valid]).astype(np.int64)
            features = df.loc[valid, neural_columns].to_numpy(dtype=np.float32)
            all_features.append(features)
            all_final_labels.append(final_labels)
            all_flags.append(flags[valid])
            all_recording_ids.append(np.full(int(valid.sum()), recording_id, dtype=np.int16))
            all_time_bins.append(time_bins[valid].astype(np.int32))

        features = np.vstack(all_features)
        final_labels = np.concatenate(all_final_labels)
        point_flags = np.vstack(all_flags)
        recording_ids = np.concatenate(all_recording_ids)
        kept_time_bins = np.concatenate(all_time_bins)

        if stage == "run1":
            stage_labels = np.where(np.isin(final_labels, list(config.merged_stage_classes)), 1, final_labels)
        elif stage == "run2":
            keep_stage = np.isin(final_labels, list(config.merged_stage_classes))
            features = features[keep_stage]
            final_labels = final_labels[keep_stage]
            point_flags = point_flags[keep_stage]
            recording_ids = recording_ids[keep_stage]
            kept_time_bins = kept_time_bins[keep_stage]
            stage_labels = np.vectorize(config.run2_label_map.get)(final_labels).astype(np.int64)
        else:
            raise ValueError(stage)

        windows: List[np.ndarray] = []
        labels: List[int] = []
        centers: List[np.ndarray] = []
        metadata: List[WindowMeta] = []
        for start in range(0, len(stage_labels) - config.window_size + 1, config.step_size):
            stop = start + config.window_size
            segment_stage = stage_labels[start:stop]
            if not np.all(segment_stage == segment_stage[0]):
                continue
            windows.append(features[start:stop, :].T.astype(np.float32))
            labels.append(int(segment_stage[0]))
            centers.append(point_flags[start:stop, :].T)
            rec = recording_ids[start:stop]
            bins = kept_time_bins[start:stop]
            unique_rec = np.unique(rec).astype(int).tolist()
            same_recording = rec[1:] == rec[:-1]
            discontinuous = bool(np.any(same_recording & (bins[1:] - bins[:-1] != 1)))
            mixed = len(unique_rec) != 1
            if mixed:
                recording_file = "MIXED"
                recording_key = "MIXED:" + "+".join(map(str, unique_rec))
                rec_id = -1
            else:
                rec_id = int(unique_rec[0])
                recording_file = file_names[rec_id]
                recording_key = recording_file
            metadata.append(
                WindowMeta(
                    recording_id=rec_id,
                    recording_file=recording_file,
                    recording_key=recording_key,
                    recording_mixed=mixed,
                    recording_ids_json=json.dumps(unique_rec),
                    start_time_bin=int(bins[0]),
                    end_time_bin=int(bins[-1]),
                    discontinuous=discontinuous,
                )
            )

        self.x = np.stack(windows).astype(np.float32)
        self.y = np.asarray(labels, dtype=np.int64)
        self.center_flags = np.stack(centers).astype(bool)
        self.segment_masks = np.zeros_like(self.center_flags, dtype=bool)
        clipped = np.zeros(len(self.conditions), dtype=int)
        for window_i in range(len(self.y)):
            for cond_i in range(len(self.conditions)):
                for center in np.flatnonzero(self.center_flags[window_i, cond_i]):
                    start = max(0, int(center) - self.radius_bins)
                    stop = min(config.window_size, int(center) + self.radius_bins)
                    if start > int(center) - self.radius_bins or stop < int(center) + self.radius_bins:
                        clipped[cond_i] += 1
                    self.segment_masks[window_i, cond_i, start:stop] = True
        self.neural_columns = neural_columns
        self.metadata = pd.DataFrame([vars(row) for row in metadata])
        self.metadata.insert(0, "dataset_index", np.arange(len(self.metadata), dtype=int))
        self.mapping_detail = pd.DataFrame(mapping_rows)
        coverage = []
        for cond_i, cond in self.conditions.iterrows():
            mask = self.segment_masks[:, cond_i, :]
            coverage.append(
                {
                    "stage": stage,
                    "condition_id": cond["condition_id"],
                    "label": int(cond["label"]),
                    "time_s": float(cond["time_s"]),
                    "center_occurrences_all_windows": int(self.center_flags[:, cond_i, :].sum()),
                    "affected_windows_all": int(mask.any(axis=1).sum()),
                    "replaced_bins_all": int(mask.sum()),
                    "boundary_clipped_centers": int(clipped[cond_i]),
                }
            )
        self.coverage = pd.DataFrame(coverage)

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.x[index]),
            torch.tensor(int(self.y[index]), dtype=torch.long),
            torch.from_numpy(self.segment_masks[index]),
            int(index),
        )


def make_model(config, input_size: int, architecture: str) -> ConvFeatureAttentionLSTM:
    return ConvFeatureAttentionLSTM(
        input_size=input_size,
        hidden_size=config.hidden_size,
        attention_hidden_size=config.attention_hidden_size,
        num_classes=config.num_classes,
        conv_channels=config.conv_channels,
        kernel_size=config.kernel_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        **ARCH_FLAGS[architecture],
    ).to(config.device)


def checkpoint_path(architecture: str, stage: str, fold: str) -> Path:
    if architecture == "full":
        return LEGACY_ROOT / f"best_test_model_{stage}_{fold}.pth"
    return RESULT_ROOT / "checkpoints" / architecture / stage / f"{fold}.pth"


def split_path(stage: str, fold: str) -> Path:
    return LEGACY_ROOT / f"fold_split_{stage}_{fold}.json"


def test_indices(stage: str, fold: str) -> List[int]:
    payload = json.loads(split_path(stage, fold).read_text(encoding="utf-8"))
    return list(map(int, payload["test_indices"]))


def load_model(architecture: str, stage: str, fold: str, dataset, config):
    path = checkpoint_path(architecture, stage, fold)
    if not path.exists():
        raise FileNotFoundError(f"Required frozen checkpoint unavailable: {path}")
    model = make_model(config, dataset.x.shape[1], architecture)
    model.load_state_dict(load_state(path, config.device))
    model.eval()
    return model


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


@torch.no_grad()
def infer_clean(model, dataset, indices: Sequence[int], config) -> Dict[str, np.ndarray]:
    loader = DataLoader(Subset(dataset, list(indices)), batch_size=config.batch_size, shuffle=False)
    ys: List[np.ndarray] = []
    logits_all: List[np.ndarray] = []
    for x, y, _, _ in loader:
        logits = model(x.float().to(config.device)).detach().cpu().numpy()
        ys.append(y.numpy().astype(int))
        logits_all.append(logits)
    y_true = np.concatenate(ys)
    logits = np.concatenate(logits_all)
    probabilities = softmax_numpy(logits)
    return {
        "y_true": y_true,
        "logits": logits,
        "probabilities": probabilities,
        "pred": probabilities.argmax(axis=1).astype(int),
    }


@torch.no_grad()
def infer_masked_sparse(
    model,
    dataset: PointwiseLegacyDataset,
    indices: Sequence[int],
    config,
    condition_index: int,
    clean: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    global_indices = np.asarray(indices, dtype=int)
    masks = dataset.segment_masks[global_indices, condition_index, :]
    affected_local = np.flatnonzero(masks.any(axis=1))
    logits = np.asarray(clean["logits"]).copy()
    if len(affected_local):
        affected_global = global_indices[affected_local]
        x = torch.from_numpy(dataset.x[affected_global]).float()
        mask = torch.from_numpy(masks[affected_local]).bool()
        changed_logits: List[np.ndarray] = []
        for start in range(0, len(x), config.batch_size):
            xb = x[start : start + config.batch_size].to(config.device)
            mb = mask[start : start + config.batch_size].to(config.device)
            fill = xb.amax(dim=2, keepdim=True)
            changed = torch.where(mb.unsqueeze(1), fill.expand_as(xb), xb)
            changed_logits.append(model(changed).detach().cpu().numpy())
        logits[affected_local] = np.concatenate(changed_logits)
    probabilities = softmax_numpy(logits)
    return {
        "y_true": np.asarray(clean["y_true"]).copy(),
        "logits": logits,
        "probabilities": probabilities,
        "pred": probabilities.argmax(axis=1).astype(int),
        "mask_hit": masks.any(axis=1),
        "replaced_bin_count": masks.sum(axis=1).astype(int),
    }


def probability_details(y_true: np.ndarray, probabilities: np.ndarray, logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rows = np.arange(len(y_true))
    true_probability = probabilities[rows, y_true]
    other = logits.copy()
    other[rows, y_true] = -np.inf
    margin = logits[rows, y_true] - np.max(other, axis=1)
    return true_probability, margin


def metric_bundle(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, object]:
    labels = [0, 1, 2]
    recalls = recall_score(y_true, pred, labels=labels, average=None, zero_division=0) * 100.0
    cm = confusion_matrix(y_true, pred, labels=labels)
    return {
        "n_samples": int(len(y_true)),
        "accuracy_pct": float(accuracy_score(y_true, pred) * 100.0),
        "macro_recall_pct": float(recall_score(y_true, pred, labels=labels, average="macro", zero_division=0) * 100.0),
        "macro_f1_pct": float(f1_score(y_true, pred, labels=labels, average="macro", zero_division=0) * 100.0),
        "mcc": float(matthews_corrcoef(y_true, pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "recall_class_0_pct": float(recalls[0]),
        "recall_class_1_pct": float(recalls[1]),
        "recall_class_2_pct": float(recalls[2]),
        "confusion_matrix_json": json.dumps(cm.astype(int).tolist()),
    }


def rows_for_condition(
    architecture: str,
    condition_id: str,
    condition_type: str,
    target_label: Optional[int],
    time_s: Optional[float],
    stage: str,
    fold: str,
    dataset: PointwiseLegacyDataset,
    indices: Sequence[int],
    arrays: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    idx = np.asarray(indices, dtype=int)
    meta = dataset.metadata.iloc[idx].reset_index(drop=True).copy()
    y_true = np.asarray(arrays["y_true"], dtype=int)
    pred = np.asarray(arrays["pred"], dtype=int)
    prob = np.asarray(arrays["probabilities"], dtype=float)
    logits = np.asarray(arrays["logits"], dtype=float)
    ptrue, margin = probability_details(y_true, prob, logits)
    out = meta
    out.insert(0, "architecture", architecture)
    out.insert(1, "architecture_label", ARCH_LABELS[architecture])
    out.insert(2, "condition_id", condition_id)
    out.insert(3, "condition_type", condition_type)
    out.insert(4, "target_label", target_label)
    out.insert(5, "time_s", time_s)
    out.insert(6, "stage", stage)
    out.insert(7, "fold", fold)
    out["y_true"] = y_true
    out["y_true_stage_label"] = [STAGE_CLASS_NAMES[stage][int(value)] for value in y_true]
    out["pred"] = pred
    out["prob_0"] = prob[:, 0]
    out["prob_1"] = prob[:, 1]
    out["prob_2"] = prob[:, 2]
    out["probability_vector_json"] = [json.dumps(row.tolist()) for row in prob]
    out["logit_0"] = logits[:, 0]
    out["logit_1"] = logits[:, 1]
    out["logit_2"] = logits[:, 2]
    out["logit_vector_json"] = [json.dumps(row.tolist()) for row in logits]
    out["true_class_probability"] = ptrue
    out["logit_margin"] = margin
    out["correct"] = pred == y_true
    if condition_type == "timepoint_mask":
        out["mask_hit"] = np.asarray(arrays["mask_hit"], dtype=bool)
        out["replaced_bin_count"] = np.asarray(arrays["replaced_bin_count"], dtype=int)
    else:
        out["mask_hit"] = False
        out["replaced_bin_count"] = 0
    return out


def run_inference(
    datasets: Mapping[str, PointwiseLegacyDataset],
    configs: Mapping[str, object],
    conditions: pd.DataFrame,
) -> pd.DataFrame:
    chunks: List[pd.DataFrame] = []
    for stage in STAGES:
        dataset = datasets[stage]
        config = configs[stage]
        for fold in FOLDS:
            indices = test_indices(stage, fold)
            for architecture in ARCHITECTURES:
                print(f"[inference] {architecture} {stage} {fold}", flush=True)
                model = load_model(architecture, stage, fold, dataset, config)
                clean = infer_clean(model, dataset, indices, config)
                chunks.append(
                    rows_for_condition(
                        architecture, "clean", "clean", None, None, stage, fold, dataset, indices, clean
                    )
                )
                for cond_i, cond in conditions.iterrows():
                    masked = infer_masked_sparse(model, dataset, indices, config, int(cond_i), clean)
                    chunks.append(
                        rows_for_condition(
                            architecture,
                            str(cond["condition_id"]),
                            "timepoint_mask",
                            int(cond["label"]),
                            float(cond["time_s"]),
                            stage,
                            fold,
                            dataset,
                            indices,
                            masked,
                        )
                    )
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    result = pd.concat(chunks, ignore_index=True)
    result = result.sort_values(["architecture", "condition_id", "stage", "fold", "dataset_index"])
    result.to_csv(SOURCE_DIR / "condition_predictions_stagewise.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")
    return result


def condition_metrics(predictions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    cm_rows: List[Dict[str, object]] = []
    keys = ["architecture", "architecture_label", "condition_id", "condition_type", "target_label", "time_s", "stage"]
    for key, group in predictions.groupby(keys, dropna=False, sort=False):
        base = dict(zip(keys, key))
        for scope, part in [(str(fold), fold_df) for fold, fold_df in group.groupby("fold", sort=True)] + [("oof_pooled", group)]:
            bundle = metric_bundle(part["y_true"].to_numpy(int), part["pred"].to_numpy(int))
            ptrue = part["true_class_probability"].to_numpy(float)
            margin = part["logit_margin"].to_numpy(float)
            row = {**base, "scope": scope, **bundle}
            row["mean_true_class_probability"] = float(np.mean(ptrue))
            row["mean_logit_margin"] = float(np.mean(margin))
            row["affected_windows"] = int(part["mask_hit"].sum())
            row["replaced_bins"] = int(part["replaced_bin_count"].sum())
            rows.append(row)
            cm = confusion_matrix(part["y_true"], part["pred"], labels=[0, 1, 2])
            for true_class in range(3):
                for pred_class in range(3):
                    cm_rows.append(
                        {
                            **base,
                            "scope": scope,
                            "true_class": true_class,
                            "pred_class": pred_class,
                            "count": int(cm[true_class, pred_class]),
                        }
                    )
    metrics = pd.DataFrame(rows)
    cms = pd.DataFrame(cm_rows)
    metrics.to_csv(TABLE_DIR / "condition_metrics_stagewise.csv", index=False, encoding="utf-8-sig")
    cms.to_csv(TABLE_DIR / "confusion_matrices_stagewise.csv", index=False, encoding="utf-8-sig")
    return metrics, cms


def build_pair_definitions(conditions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, cond in conditions.iterrows():
        cid = str(cond["condition_id"])
        common = {"condition_id": cid, "target_label": int(cond["label"]), "time_s": float(cond["time_s"])}
        rows.append(
            {
                "comparison_id": f"data_full__{cid}",
                "comparison_family": "data_ablation_full",
                "module": None,
                "baseline_architecture": "full",
                "baseline_condition": "clean",
                "ablated_architecture": "full",
                "ablated_condition": cid,
                **common,
            }
        )
        for module in MODULES:
            rows.append(
                {
                    "comparison_id": f"data_{module}__{cid}",
                    "comparison_family": "data_ablation_no_module",
                    "module": module,
                    "baseline_architecture": module,
                    "baseline_condition": "clean",
                    "ablated_architecture": module,
                    "ablated_condition": cid,
                    **common,
                }
            )
            rows.append(
                {
                    "comparison_id": f"module_masked__{module}__{cid}",
                    "comparison_family": "module_ablation_masked",
                    "module": module,
                    "baseline_architecture": "full",
                    "baseline_condition": cid,
                    "ablated_architecture": module,
                    "ablated_condition": cid,
                    **common,
                }
            )
    for module in MODULES:
        rows.append(
            {
                "comparison_id": f"module_clean__{module}",
                "comparison_family": "module_ablation_clean",
                "module": module,
                "condition_id": None,
                "target_label": None,
                "time_s": None,
                "baseline_architecture": "full",
                "baseline_condition": "clean",
                "ablated_architecture": module,
                "ablated_condition": "clean",
            }
        )
    definitions = pd.DataFrame(rows)
    definitions.to_csv(SOURCE_DIR / "comparison_definitions.csv", index=False, encoding="utf-8-sig")
    return definitions


def paired_predictions(predictions: pd.DataFrame, definitions: pd.DataFrame) -> pd.DataFrame:
    index_cols = ["stage", "fold", "dataset_index"]
    lookup: Dict[Tuple[str, str], pd.DataFrame] = {}
    keep = index_cols + [
        "recording_id", "recording_file", "recording_key", "recording_mixed", "recording_ids_json",
        "start_time_bin", "end_time_bin", "discontinuous", "y_true", "pred", "probability_vector_json",
        "true_class_probability", "logit_margin", "correct", "mask_hit", "replaced_bin_count",
    ]
    for (architecture, condition_id), group in predictions.groupby(["architecture", "condition_id"], sort=False):
        lookup[(str(architecture), str(condition_id))] = group[keep].copy()

    chunks: List[pd.DataFrame] = []
    for _, definition in definitions.iterrows():
        base = lookup[(str(definition["baseline_architecture"]), str(definition["baseline_condition"]))]
        abl = lookup[(str(definition["ablated_architecture"]), str(definition["ablated_condition"]))]
        merged = base.merge(abl, on=index_cols, suffixes=("_baseline", "_ablated"), validate="one_to_one")
        if not np.array_equal(merged["y_true_baseline"].to_numpy(), merged["y_true_ablated"].to_numpy()):
            raise RuntimeError(f"Paired truth mismatch: {definition['comparison_id']}")
        out = pd.DataFrame(
            {
                "comparison_id": definition["comparison_id"],
                "comparison_family": definition["comparison_family"],
                "module": definition["module"],
                "condition_id": definition["condition_id"],
                "target_label": definition["target_label"],
                "time_s": definition["time_s"],
                "baseline_architecture": definition["baseline_architecture"],
                "ablated_architecture": definition["ablated_architecture"],
                "baseline_condition": definition["baseline_condition"],
                "ablated_condition": definition["ablated_condition"],
                "stage": merged["stage"],
                "fold": merged["fold"],
                "dataset_index": merged["dataset_index"],
                "recording_id": merged["recording_id_baseline"],
                "recording_file": merged["recording_file_baseline"],
                "recording_key": merged["recording_key_baseline"],
                "recording_mixed": merged["recording_mixed_baseline"],
                "recording_ids_json": merged["recording_ids_json_baseline"],
                "start_time_bin": merged["start_time_bin_baseline"],
                "end_time_bin": merged["end_time_bin_baseline"],
                "discontinuous": merged["discontinuous_baseline"],
                "y_true": merged["y_true_baseline"].astype(int),
                "baseline_pred": merged["pred_baseline"].astype(int),
                "ablated_pred": merged["pred_ablated"].astype(int),
                "baseline_probability_vector": merged["probability_vector_json_baseline"],
                "ablated_probability_vector": merged["probability_vector_json_ablated"],
                "baseline_true_class_probability": merged["true_class_probability_baseline"],
                "ablated_true_class_probability": merged["true_class_probability_ablated"],
                "baseline_logit_margin": merged["logit_margin_baseline"],
                "ablated_logit_margin": merged["logit_margin_ablated"],
                "baseline_correct": merged["correct_baseline"].astype(bool),
                "ablated_correct": merged["correct_ablated"].astype(bool),
                "mask_hit": merged["mask_hit_ablated"].astype(bool),
                "replaced_bin_count": merged["replaced_bin_count_ablated"].astype(int),
            }
        )
        out["true_class_probability_delta"] = (
            out["ablated_true_class_probability"] - out["baseline_true_class_probability"]
        )
        out["logit_margin_delta"] = out["ablated_logit_margin"] - out["baseline_logit_margin"]
        out["true_class_probability_delta_sign"] = out["true_class_probability_delta"].map(sign_symbol)
        out["logit_margin_delta_sign"] = out["logit_margin_delta"].map(sign_symbol)
        out["transition"] = np.select(
            [
                out["baseline_correct"] & out["ablated_correct"],
                out["baseline_correct"] & ~out["ablated_correct"],
                ~out["baseline_correct"] & out["ablated_correct"],
            ],
            ["CC", "CW", "WC"],
            default="WW",
        )
        chunks.append(out)
    result = pd.concat(chunks, ignore_index=True)
    result.to_csv(SOURCE_DIR / "paired_predictions_stagewise.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")
    return result


def paired_metric_tables(pairs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: List[Dict[str, object]] = []
    ovr_rows: List[Dict[str, object]] = []
    keys = ["comparison_id", "comparison_family", "module", "condition_id", "target_label", "time_s", "stage"]
    for key, group in pairs.groupby(keys, dropna=False, sort=False):
        base = dict(zip(keys, key))
        scopes = [(str(fold), fold_df) for fold, fold_df in group.groupby("fold", sort=True)] + [("oof_pooled", group)]
        for scope, part in scopes:
            y = part["y_true"].to_numpy(int)
            pred_b = part["baseline_pred"].to_numpy(int)
            pred_a = part["ablated_pred"].to_numpy(int)
            mb = metric_bundle(y, pred_b)
            ma = metric_bundle(y, pred_a)
            row: Dict[str, object] = {**base, "scope": scope, "n_samples": int(len(part))}
            for metric in ("accuracy_pct", "macro_recall_pct", "macro_f1_pct", "mcc"):
                row[f"baseline_{metric}"] = mb[metric]
                row[f"ablated_{metric}"] = ma[metric]
                row[f"delta_{metric}"] = float(ma[metric]) - float(mb[metric])
                row[f"delta_{metric}_sign"] = sign_symbol(float(row[f"delta_{metric}"]))
            for class_id in range(3):
                key_name = f"recall_class_{class_id}_pct"
                row[f"baseline_{key_name}"] = mb[key_name]
                row[f"ablated_{key_name}"] = ma[key_name]
                row[f"delta_{key_name}"] = float(ma[key_name]) - float(mb[key_name])
            row["confusion_baseline_json"] = mb["confusion_matrix_json"]
            row["confusion_ablated_json"] = ma["confusion_matrix_json"]
            row["confusion_delta_json"] = json.dumps(
                (np.asarray(json.loads(ma["confusion_matrix_json"])) - np.asarray(json.loads(mb["confusion_matrix_json"]))).tolist()
            )
            row["mean_true_probability_delta"] = float(part["true_class_probability_delta"].mean())
            row["mean_logit_margin_delta"] = float(part["logit_margin_delta"].mean())
            counts = part["transition"].value_counts()
            for transition in ("CC", "CW", "WC", "WW"):
                row[transition] = int(counts.get(transition, 0))
            row["affected_windows"] = int(part["mask_hit"].sum())
            row["replaced_bins"] = int(part["replaced_bin_count"].sum())
            metric_rows.append(row)

            for class_id in range(3):
                b_pos = pred_b == class_id
                a_pos = pred_a == class_id
                truth = y == class_id
                base_fp = int((~truth & b_pos).sum())
                base_fn = int((truth & ~b_pos).sum())
                abl_fp = int((~truth & a_pos).sum())
                abl_fn = int((truth & ~a_pos).sum())
                ovr_rows.append(
                    {
                        **base,
                        "scope": scope,
                        "stage_class": class_id,
                        "stage_class_label": STAGE_CLASS_NAMES[str(base["stage"])][class_id],
                        "baseline_TP": int((truth & b_pos).sum()),
                        "baseline_FP": base_fp,
                        "baseline_FN": base_fn,
                        "baseline_TN": int((~truth & ~b_pos).sum()),
                        "ablated_TP": int((truth & a_pos).sum()),
                        "ablated_FP": abl_fp,
                        "ablated_FN": abl_fn,
                        "ablated_TN": int((~truth & ~a_pos).sum()),
                        "delta_FP": abl_fp - base_fp,
                        "delta_FN": abl_fn - base_fn,
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    ovr = pd.DataFrame(ovr_rows)
    metrics.to_csv(TABLE_DIR / "paired_metric_deltas_stagewise.csv", index=False, encoding="utf-8-sig")
    ovr.to_csv(TABLE_DIR / "one_vs_rest_delta_fp_fn_stagewise.csv", index=False, encoding="utf-8-sig")
    return metrics, ovr


def legacy_recall_vectors(condition_metrics_df: pd.DataFrame, conditions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows = condition_metrics_df[condition_metrics_df["scope"].isin(FOLDS)].copy()
    merged_rows: List[Dict[str, object]] = []
    for (architecture, condition_id, fold), group in fold_rows.groupby(["architecture", "condition_id", "scope"], sort=False):
        by_stage = {row.stage: row for row in group.itertuples()}
        if set(by_stage) != set(STAGES):
            raise RuntimeError(f"Missing stage metric for {architecture} {condition_id} {fold}")
        r1, r2 = by_stage["run1"], by_stage["run2"]
        recalls = {
            1: float(r1.recall_class_0_pct),
            2: (float(r1.recall_class_1_pct) + float(r2.recall_class_0_pct)) / 2.0,
            3: float(r1.recall_class_2_pct),
            4: (float(r1.recall_class_1_pct) + float(r2.recall_class_1_pct)) / 2.0,
            5: (float(r1.recall_class_1_pct) + float(r2.recall_class_2_pct)) / 2.0,
        }
        for behavior, value in recalls.items():
            merged_rows.append(
                {
                    "architecture": architecture,
                    "condition_id": condition_id,
                    "fold": fold,
                    "behavior": behavior,
                    "behavior_label": f"L{behavior}",
                    "recall_pct": value,
                }
            )
    merged = pd.DataFrame(merged_rows)
    mean_vec = merged.groupby(["architecture", "condition_id", "behavior", "behavior_label"], as_index=False)["recall_pct"].mean()

    effect_rows: List[Dict[str, object]] = []
    full_clean = mean_vec[(mean_vec.architecture == "full") & (mean_vec.condition_id == "clean")].set_index("behavior")
    for _, cond in conditions.iterrows():
        masked = mean_vec[(mean_vec.architecture == "full") & (mean_vec.condition_id == cond.condition_id)].set_index("behavior")
        for behavior in range(1, 6):
            delta = float(masked.loc[behavior, "recall_pct"] - full_clean.loc[behavior, "recall_pct"])
            effect_rows.append(
                {
                    "effect_type": "data_ablation_full",
                    "effect_id": cond.condition_id,
                    "module": None,
                    "target_label": int(cond.label),
                    "time_s": float(cond.time_s),
                    "behavior": behavior,
                    "behavior_label": f"L{behavior}",
                    "delta_recall_pp": delta,
                    "delta_sign": sign_symbol(delta),
                }
            )
    for module in MODULES:
        module_clean = mean_vec[(mean_vec.architecture == module) & (mean_vec.condition_id == "clean")].set_index("behavior")
        for behavior in range(1, 6):
            delta = float(module_clean.loc[behavior, "recall_pct"] - full_clean.loc[behavior, "recall_pct"])
            effect_rows.append(
                {
                    "effect_type": "module_ablation_clean",
                    "effect_id": module,
                    "module": module,
                    "target_label": None,
                    "time_s": None,
                    "behavior": behavior,
                    "behavior_label": f"L{behavior}",
                    "delta_recall_pp": delta,
                    "delta_sign": sign_symbol(delta),
                }
            )
    effects = pd.DataFrame(effect_rows)
    merged.to_csv(SOURCE_DIR / "legacy_recall_by_fold.csv", index=False, encoding="utf-8-sig")
    mean_vec.to_csv(SOURCE_DIR / "legacy_recall_mean_vectors.csv", index=False, encoding="utf-8-sig")
    effects.to_csv(TABLE_DIR / "recall_delta_vectors.csv", index=False, encoding="utf-8-sig")
    return mean_vec, effects


def recall_vector_correlations(effects: pd.DataFrame, conditions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, cond in conditions.iterrows():
        data_vec = effects[(effects.effect_type == "data_ablation_full") & (effects.effect_id == cond.condition_id)].sort_values("behavior")
        x = data_vec["delta_recall_pp"].to_numpy(float)
        for module in MODULES:
            mod_vec = effects[(effects.effect_type == "module_ablation_clean") & (effects.effect_id == module)].sort_values("behavior")
            y = mod_vec["delta_recall_pp"].to_numpy(float)
            rows.append(
                {
                    "condition_id": cond.condition_id,
                    "target_label": int(cond.label),
                    "time_s": float(cond.time_s),
                    "module": module,
                    "pearson": safe_corr(x, y, "pearson"),
                    "spearman": safe_corr(x, y, "spearman"),
                    "cosine_similarity": safe_corr(x, y, "cosine"),
                    "n_behaviors": 5,
                    "data_vector_json": json.dumps(x.tolist()),
                    "module_vector_json": json.dumps(y.tolist()),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(TABLE_DIR / "A_recall_vector_similarity.csv", index=False, encoding="utf-8-sig")
    return result


def recording_bootstrap_spearman(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    repeats: int = 500,
    seed: int = 20260925,
) -> Tuple[float, float, int, int]:
    usable = df[~df["recording_mixed"].astype(bool)].copy().reset_index(drop=True)
    groups = sorted(usable["recording_key"].dropna().unique().tolist())
    if len(groups) < 3:
        return float("nan"), float("nan"), len(groups), int(len(df) - len(usable))
    # Spearman is Pearson correlation of ranks. Rank once, then use a recording
    # cluster bootstrap on those pooled ranks. This preserves the requested
    # recording-level resampling while avoiding hundreds of thousands of slow
    # pandas concatenations and repeated ranking operations.
    rank_x = usable[x_col].rank(method="average").to_numpy(float)
    rank_y = usable[y_col].rank(method="average").to_numpy(float)
    by_group = {
        key: np.flatnonzero(usable["recording_key"].to_numpy() == key)
        for key in groups
    }
    rng = np.random.default_rng(seed)
    values: List[float] = []
    for _ in range(repeats):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        sampled_indices = np.concatenate([by_group[key] for key in sampled])
        value = safe_corr(rank_x[sampled_indices], rank_y[sampled_indices], "pearson")
        if np.isfinite(value):
            values.append(value)
    if len(values) < max(20, repeats // 10):
        return float("nan"), float("nan"), len(groups), int(len(df) - len(usable))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)), len(groups), int(len(df) - len(usable))


def sample_effect_associations(pairs: pd.DataFrame, conditions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pair_lookup = {key: group.copy() for key, group in pairs.groupby("comparison_id", sort=False)}
    rows: List[Dict[str, object]] = []
    joined_chunks: List[pd.DataFrame] = []
    for _, cond in conditions.iterrows():
        data = pair_lookup[f"data_full__{cond.condition_id}"]
        for module in MODULES:
            mod = pair_lookup[f"module_clean__{module}"]
            key_cols = ["stage", "fold", "dataset_index"]
            for stage in STAGES:
                d = data[data.stage == stage]
                m = mod[mod.stage == stage]
                joined = d.merge(
                    m[key_cols + ["true_class_probability_delta", "logit_margin_delta"]],
                    on=key_cols,
                    suffixes=("_data", "_module"),
                    validate="one_to_one",
                )
                joined["condition_id"] = cond.condition_id
                joined["target_label"] = int(cond.label)
                joined["time_s"] = float(cond.time_s)
                joined["module"] = module
                joined_chunks.append(joined)
                for metric, x_col, y_col in (
                    (
                        "true_class_probability",
                        "true_class_probability_delta_data",
                        "true_class_probability_delta_module",
                    ),
                    ("logit_margin", "logit_margin_delta_data", "logit_margin_delta_module"),
                ):
                    point = safe_corr(joined[x_col], joined[y_col], "spearman")
                    ci_low, ci_high, n_recordings, n_mixed = recording_bootstrap_spearman(
                        joined, x_col, y_col, repeats=500, seed=20260925 + int(cond.condition_index) * 100 + list(MODULES).index(module) * 10
                    )
                    rows.append(
                        {
                            "condition_id": cond.condition_id,
                            "target_label": int(cond.label),
                            "time_s": float(cond.time_s),
                            "module": module,
                            "stage": stage,
                            "metric": metric,
                            "spearman": point,
                            "bootstrap_ci_low": ci_low,
                            "bootstrap_ci_high": ci_high,
                            "bootstrap_unit": "recording_file",
                            "bootstrap_repeats": 500,
                            "n_samples": int(len(joined)),
                            "n_recordings": n_recordings,
                            "n_mixed_recording_windows_excluded_from_ci": n_mixed,
                        }
                    )
    result = pd.DataFrame(rows)
    joined_all = pd.concat(joined_chunks, ignore_index=True)
    result.to_csv(TABLE_DIR / "B_sample_effect_spearman.csv", index=False, encoding="utf-8-sig")
    joined_all.to_csv(SOURCE_DIR / "sample_effect_pairs_for_scatter.csv.gz", index=False, encoding="utf-8-sig", compression="gzip")
    return result, joined_all


def transition_set_associations(pairs: pd.DataFrame, conditions: pd.DataFrame) -> pd.DataFrame:
    pair_lookup = {key: group.copy() for key, group in pairs.groupby("comparison_id", sort=False)}
    rows: List[Dict[str, object]] = []
    for _, cond in conditions.iterrows():
        data = pair_lookup[f"data_full__{cond.condition_id}"]
        for module in MODULES:
            mod = pair_lookup[f"module_clean__{module}"]
            for stage in (*STAGES, "all_stages"):
                d = data if stage == "all_stages" else data[data.stage == stage]
                m = mod if stage == "all_stages" else mod[mod.stage == stage]
                keys = ["stage", "fold", "dataset_index"]
                merged = d[keys + ["transition"]].merge(
                    m[keys + ["transition"]], on=keys, suffixes=("_data", "_module"), validate="one_to_one"
                )
                data_cw = merged.transition_data.eq("CW")
                mod_cw = merged.transition_module.eq("CW")
                data_wc = merged.transition_data.eq("WC")
                mod_wc = merged.transition_module.eq("WC")
                denom = int(data_cw.sum())
                rows.append(
                    {
                        "condition_id": cond.condition_id,
                        "target_label": int(cond.label),
                        "time_s": float(cond.time_s),
                        "module": module,
                        "stage": stage,
                        "n_samples": int(len(merged)),
                        "data_CW": int(data_cw.sum()),
                        "module_CW": int(mod_cw.sum()),
                        "CW_intersection": int((data_cw & mod_cw).sum()),
                        "CW_jaccard": jaccard(data_cw, mod_cw),
                        "CW_binary_mcc_phi": binary_phi(data_cw, mod_cw),
                        "p_module_CW_given_data_CW": float((data_cw & mod_cw).sum() / denom) if denom else float("nan"),
                        "data_WC": int(data_wc.sum()),
                        "module_WC": int(mod_wc.sum()),
                        "WC_intersection": int((data_wc & mod_wc).sum()),
                        "WC_jaccard": jaccard(data_wc, mod_wc),
                        "WC_binary_mcc_phi": binary_phi(data_wc, mod_wc),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(TABLE_DIR / "C_damage_recovery_set_association.csv", index=False, encoding="utf-8-sig")
    return result


def confusion_delta_similarity(paired_metrics_df: pd.DataFrame, conditions: pd.DataFrame) -> pd.DataFrame:
    pooled = paired_metrics_df[paired_metrics_df.scope == "oof_pooled"].copy()
    lookup = {row.comparison_id: row for row in pooled.itertuples()}
    rows: List[Dict[str, object]] = []
    offdiag = np.array([False, True, True, True, False, True, True, True, False])
    for _, cond in conditions.iterrows():
        for module in MODULES:
            for stage in STAGES:
                data_row = pooled[(pooled.comparison_id == f"data_full__{cond.condition_id}") & (pooled.stage == stage)].iloc[0]
                mod_row = pooled[(pooled.comparison_id == f"module_clean__{module}") & (pooled.stage == stage)].iloc[0]
                d = np.asarray(json.loads(data_row.confusion_delta_json), dtype=float).ravel()[offdiag]
                m = np.asarray(json.loads(mod_row.confusion_delta_json), dtype=float).ravel()[offdiag]
                rows.append(
                    {
                        "condition_id": cond.condition_id,
                        "target_label": int(cond.label),
                        "time_s": float(cond.time_s),
                        "module": module,
                        "stage": stage,
                        "pearson": safe_corr(d, m, "pearson"),
                        "spearman": safe_corr(d, m, "spearman"),
                        "cosine_similarity": safe_corr(d, m, "cosine"),
                        "data_offdiag_delta_json": json.dumps(d.tolist()),
                        "module_offdiag_delta_json": json.dumps(m.tolist()),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(TABLE_DIR / "E_confusion_delta_offdiagonal_similarity.csv", index=False, encoding="utf-8-sig")
    return result


def canonical_fp_fn(ovr: pd.DataFrame, conditions: pd.DataFrame) -> pd.DataFrame:
    pooled = ovr[ovr.scope == "oof_pooled"].copy()
    rows: List[Dict[str, object]] = []
    ids = [f"data_full__{cid}" for cid in conditions.condition_id] + [f"module_clean__{m}" for m in MODULES]
    for comparison_id in ids:
        part = pooled[pooled.comparison_id == comparison_id]
        if part.empty:
            continue
        first = part.iloc[0]
        effect_type = "data_ablation_full" if comparison_id.startswith("data_full") else "module_ablation_clean"
        effect_id = first.condition_id if effect_type.startswith("data") else first.module
        for behavior, (stage, class_id) in CANONICAL_BEHAVIOR_STAGE_CLASS.items():
            row = part[(part.stage == stage) & (part.stage_class == class_id)].iloc[0]
            rows.append(
                {
                    "comparison_id": comparison_id,
                    "effect_type": effect_type,
                    "effect_id": effect_id,
                    "module": first.module,
                    "condition_id": first.condition_id,
                    "target_label": first.target_label,
                    "time_s": first.time_s,
                    "behavior": behavior,
                    "behavior_label": f"L{behavior}",
                    "source_stage": stage,
                    "source_stage_class": class_id,
                    "delta_FP": int(row.delta_FP),
                    "delta_FN": int(row.delta_FN),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(TABLE_DIR / "D_canonical_behavior_delta_fp_fn.csv", index=False, encoding="utf-8-sig")
    return result


def double_ablation_interactions(
    predictions: pd.DataFrame,
    mean_vectors: pd.DataFrame,
    conditions: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Legacy five-behavior recall interaction.
    recall_rows: List[Dict[str, object]] = []
    vector_lookup = {
        (arch, cid): group.set_index("behavior")["recall_pct"]
        for (arch, cid), group in mean_vectors.groupby(["architecture", "condition_id"], sort=False)
    }
    for _, cond in conditions.iterrows():
        full_effect = vector_lookup[("full", cond.condition_id)] - vector_lookup[("full", "clean")]
        for module in MODULES:
            no_effect = vector_lookup[(module, cond.condition_id)] - vector_lookup[(module, "clean")]
            interaction = full_effect - no_effect
            for behavior in range(1, 6):
                recall_rows.append(
                    {
                        "condition_id": cond.condition_id,
                        "target_label": int(cond.label),
                        "time_s": float(cond.time_s),
                        "module": module,
                        "behavior": behavior,
                        "behavior_label": f"L{behavior}",
                        "data_effect_full_delta_recall_pp": float(full_effect.loc[behavior]),
                        "data_effect_no_module_delta_recall_pp": float(no_effect.loc[behavior]),
                        "interaction_delta_recall_pp": float(interaction.loc[behavior]),
                        "interaction_sign": sign_symbol(float(interaction.loc[behavior])),
                    }
                )
    recall_df = pd.DataFrame(recall_rows)
    recall_df.to_csv(TABLE_DIR / "F_double_ablation_interaction_recall.csv", index=False, encoding="utf-8-sig")

    # Per-stage/per-class recall, true-probability and logit-margin interaction.
    rows: List[Dict[str, object]] = []
    for _, cond in conditions.iterrows():
        for module in MODULES:
            for stage in STAGES:
                subset = predictions[predictions.stage == stage]
                groups = {}
                for arch, cid, name in (
                    ("full", "clean", "full_clean"),
                    ("full", cond.condition_id, "full_mask"),
                    (module, "clean", "module_clean"),
                    (module, cond.condition_id, "module_mask"),
                ):
                    groups[name] = subset[(subset.architecture == arch) & (subset.condition_id == cid)]
                for class_id in range(3):
                    metrics = {}
                    for name, group in groups.items():
                        truth = group.y_true.to_numpy(int)
                        pred = group.pred.to_numpy(int)
                        selected = truth == class_id
                        recall = float((pred[selected] == class_id).mean() * 100.0) if selected.any() else float("nan")
                        metrics[name] = {
                            "recall": recall,
                            "ptrue": float(group.loc[selected, "true_class_probability"].mean()) if selected.any() else float("nan"),
                            "margin": float(group.loc[selected, "logit_margin"].mean()) if selected.any() else float("nan"),
                            "n": int(selected.sum()),
                        }
                    for metric in ("recall", "ptrue", "margin"):
                        full_effect = metrics["full_mask"][metric] - metrics["full_clean"][metric]
                        no_effect = metrics["module_mask"][metric] - metrics["module_clean"][metric]
                        rows.append(
                            {
                                "condition_id": cond.condition_id,
                                "target_label": int(cond.label),
                                "time_s": float(cond.time_s),
                                "module": module,
                                "stage": stage,
                                "stage_class": class_id,
                                "stage_class_label": STAGE_CLASS_NAMES[stage][class_id],
                                "metric": metric,
                                "n_samples_in_class": metrics["full_clean"]["n"],
                                "data_effect_full": full_effect,
                                "data_effect_no_module": no_effect,
                                "interaction": full_effect - no_effect,
                                "interaction_sign": sign_symbol(full_effect - no_effect),
                            }
                        )
    detail = pd.DataFrame(rows)
    detail.to_csv(TABLE_DIR / "F_double_ablation_interaction_stage_class.csv", index=False, encoding="utf-8-sig")
    return recall_df, detail


def ttest_ablation_association(
    conditions: pd.DataFrame,
    effects: pd.DataFrame,
    pairs: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    for _, cond in conditions.iterrows():
        behavior = int(cond.label)
        recall_row = effects[
            (effects.effect_type == "data_ablation_full")
            & (effects.effect_id == cond.condition_id)
            & (effects.behavior == behavior)
        ].iloc[0]
        stage, class_id = CANONICAL_BEHAVIOR_STAGE_CLASS[behavior]
        pair = pairs[(pairs.comparison_id == f"data_full__{cond.condition_id}") & (pairs.stage == stage)]
        target_samples = pair[pair.y_true == class_id]
        rows.append(
            {
                "condition_id": cond.condition_id,
                "label": behavior,
                "label_name": cond.label_name,
                "time_s": float(cond.time_s),
                "t_stat": float(cond.t_stat),
                "t_stat_sign": sign_symbol(float(cond.t_stat)),
                "p_value": float(cond.p_value),
                "target_behavior_delta_recall_pp": float(recall_row.delta_recall_pp),
                "target_behavior_delta_recall_sign": sign_symbol(float(recall_row.delta_recall_pp)),
                "target_sample_mean_delta_ptrue": float(target_samples.true_class_probability_delta.mean()),
                "target_sample_mean_delta_margin": float(target_samples.logit_margin_delta.mean()),
                "n_target_samples": int(len(target_samples)),
                "source_stage": stage,
                "source_stage_class": class_id,
            }
        )
    detail = pd.DataFrame(rows)
    corr_rows = []
    comparisons = (
        ("t_stat", "target_behavior_delta_recall_pp"),
        ("abs_t_stat", "abs_target_behavior_delta_recall_pp"),
        ("t_stat", "target_sample_mean_delta_ptrue"),
        ("abs_t_stat", "abs_target_sample_mean_delta_ptrue"),
        ("t_stat", "target_sample_mean_delta_margin"),
        ("abs_t_stat", "abs_target_sample_mean_delta_margin"),
    )
    detail["abs_t_stat"] = detail.t_stat.abs()
    detail["abs_target_behavior_delta_recall_pp"] = detail.target_behavior_delta_recall_pp.abs()
    detail["abs_target_sample_mean_delta_ptrue"] = detail.target_sample_mean_delta_ptrue.abs()
    detail["abs_target_sample_mean_delta_margin"] = detail.target_sample_mean_delta_margin.abs()
    for x_col, y_col in comparisons:
        corr_rows.append(
            {
                "x": x_col,
                "y": y_col,
                "pearson": safe_corr(detail[x_col], detail[y_col], "pearson"),
                "spearman": safe_corr(detail[x_col], detail[y_col], "spearman"),
                "n_timepoints": int(len(detail)),
                "inference": "exploratory coefficient only; no independent-window p-value",
            }
        )
    correlations = pd.DataFrame(corr_rows)
    detail.to_csv(TABLE_DIR / "G_ttest_vs_data_ablation_detail.csv", index=False, encoding="utf-8-sig")
    correlations.to_csv(TABLE_DIR / "G_ttest_vs_data_ablation_correlations.csv", index=False, encoding="utf-8-sig")
    return detail, correlations


def mask_overlap_tables(datasets: Mapping[str, PointwiseLegacyDataset], conditions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for stage, dataset in datasets.items():
        masks = dataset.segment_masks.reshape(len(dataset), len(conditions), -1)
        for i, a in conditions.iterrows():
            ma = masks[:, i, :]
            for j, b in conditions.iterrows():
                mb = masks[:, j, :]
                intersection = int(np.logical_and(ma, mb).sum())
                union = int(np.logical_or(ma, mb).sum())
                rows.append(
                    {
                        "stage": stage,
                        "condition_a": a.condition_id,
                        "condition_b": b.condition_id,
                        "label_a": int(a.label),
                        "label_b": int(b.label),
                        "time_s_a": float(a.time_s),
                        "time_s_b": float(b.time_s),
                        "intersection_bins": intersection,
                        "union_bins": union,
                        "jaccard_overlap": float(intersection / union) if union else float("nan"),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(QA_DIR / "mask_overlap_matrix_long.csv", index=False, encoding="utf-8-sig")
    for stage in STAGES:
        matrix = result[result.stage == stage].pivot(index="condition_a", columns="condition_b", values="jaccard_overlap")
        matrix.to_csv(QA_DIR / f"mask_overlap_matrix_{stage}.csv", encoding="utf-8-sig")
    return result


def qa_dataset_equivalence(
    datasets: Mapping[str, PointwiseLegacyDataset],
    conditions: pd.DataFrame,
) -> pd.DataFrame:
    offsets = {
        label: sorted(conditions.loc[conditions.label == label, "offset_bin"].astype(int).tolist())
        for label in range(1, 6)
    }
    rows: List[Dict[str, object]] = []
    for stage, config in (("run1", Config()), ("run2", Config3())):
        legacy = LegacySegmentDataset(config, stage, offsets, radius_bins=100)
        current = datasets[stage]
        x_equal = np.array_equal(legacy.x, current.x)
        y_equal = np.array_equal(legacy.y, current.y)
        for label in range(1, 6):
            cond_indices = conditions.index[conditions.label == label].to_numpy(int)
            union = current.segment_masks[:, cond_indices, :].any(axis=1)
            mask_equal = np.array_equal(union, legacy.segment_masks[:, label - 1, :])
            rows.append(
                {
                    "stage": stage,
                    "label": label,
                    "x_equal_to_existing_segment_replacement": x_equal,
                    "y_equal_to_existing_segment_replacement": y_equal,
                    "union_of_point_masks_equals_existing_label_mask": mask_equal,
                    "n_windows": len(current),
                }
            )
        if not (x_equal and y_equal and all(row["union_of_point_masks_equals_existing_label_mask"] for row in rows if row["stage"] == stage)):
            raise RuntimeError(f"Dataset/mask equivalence QA failed for {stage}")
    result = pd.DataFrame(rows)
    result.to_csv(QA_DIR / "dataset_and_mask_equivalence.csv", index=False, encoding="utf-8-sig")
    return result


def qa_fold_partitions(datasets: Mapping[str, PointwiseLegacyDataset]) -> pd.DataFrame:
    rows = []
    for stage, dataset in datasets.items():
        all_test: List[int] = []
        for fold in FOLDS:
            idx = test_indices(stage, fold)
            all_test.extend(idx)
            rows.append(
                {
                    "stage": stage,
                    "fold": fold,
                    "n_test": len(idx),
                    "n_unique_test": len(set(idx)),
                    "within_fold_unique": len(idx) == len(set(idx)),
                    "min_index": min(idx),
                    "max_index": max(idx),
                    "split_file": str(split_path(stage, fold)),
                    "split_sha256": sha256_file(split_path(stage, fold)),
                    "used_by_all_architectures": True,
                }
            )
        partition = len(all_test) == len(set(all_test)) == len(dataset) and set(all_test) == set(range(len(dataset)))
        if not partition:
            raise RuntimeError(f"Saved test folds do not form an exact partition for {stage}")
        rows.append(
            {
                "stage": stage,
                "fold": "ALL",
                "n_test": len(all_test),
                "n_unique_test": len(set(all_test)),
                "within_fold_unique": True,
                "min_index": min(all_test),
                "max_index": max(all_test),
                "split_file": "three saved split files",
                "split_sha256": "",
                "used_by_all_architectures": True,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(QA_DIR / "fold_test_index_alignment.csv", index=False, encoding="utf-8-sig")
    return result


def qa_clean_baseline(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for stage in STAGES:
        for fold in FOLDS:
            saved_path = LEGACY_ROOT / f"pred_{stage}_{fold}__on_fold_valid.csv"
            saved = pd.read_csv(saved_path).sort_values("global_index").reset_index(drop=True)
            current = predictions[
                (predictions.architecture == "full")
                & (predictions.condition_id == "clean")
                & (predictions.stage == stage)
                & (predictions.fold == fold)
            ].sort_values("dataset_index").reset_index(drop=True)
            indices_equal = np.array_equal(saved.global_index.to_numpy(int), current.dataset_index.to_numpy(int))
            true_equal = np.array_equal(saved.true.to_numpy(int), current.y_true.to_numpy(int))
            pred_equal = np.array_equal(saved.pred.to_numpy(int), current.pred.to_numpy(int))
            max_prob_diff = float(
                np.max(
                    np.abs(
                        saved[["prob_class_0", "prob_class_1", "prob_class_2"]].to_numpy(float)
                        - current[["prob_0", "prob_1", "prob_2"]].to_numpy(float)
                    )
                )
            )
            probability_tolerance = 2e-5
            passed = indices_equal and true_equal and pred_equal and max_prob_diff < probability_tolerance
            rows.append(
                {
                    "stage": stage,
                    "fold": fold,
                    "saved_prediction_file": str(saved_path),
                    "indices_equal": indices_equal,
                    "truth_equal": true_equal,
                    "predictions_equal": pred_equal,
                    "max_probability_abs_diff": max_prob_diff,
                    "probability_abs_tolerance": probability_tolerance,
                    "pass": passed,
                }
            )
            if not passed:
                raise RuntimeError(f"Clean baseline reproduction failed for {stage} {fold}")
    result = pd.DataFrame(rows)
    result.to_csv(QA_DIR / "clean_baseline_reproduction.csv", index=False, encoding="utf-8-sig")
    return result


def qa_checkpoint_manifest() -> pd.DataFrame:
    rows = []
    for architecture in ARCHITECTURES:
        for stage in STAGES:
            for fold in FOLDS:
                path = checkpoint_path(architecture, stage, fold)
                rows.append(
                    {
                        "architecture": architecture,
                        "stage": stage,
                        "fold": fold,
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "available": path.exists(),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(QA_DIR / "checkpoint_manifest_sha256.csv", index=False, encoding="utf-8-sig")
    return result


def qa_structural_metrics(mean_vectors: pd.DataFrame, condition_metrics_df: pd.DataFrame) -> pd.DataFrame:
    reference = pd.read_csv(RESULT_ROOT / "tables" / "table_e_current_structural_ablation_coherent.csv")
    folds = condition_metrics_df[(condition_metrics_df.condition_id == "clean") & (condition_metrics_df.scope.isin(FOLDS))]
    rows = []
    for architecture in ARCHITECTURES:
        ref = reference[reference.architecture == architecture].iloc[0]
        fold_values = []
        for fold in FOLDS:
            r1 = folds[(folds.architecture == architecture) & (folds.stage == "run1") & (folds.scope == fold)].iloc[0]
            r2 = folds[(folds.architecture == architecture) & (folds.stage == "run2") & (folds.scope == fold)].iloc[0]
            fold_values.append(
                {
                    "accuracy": (r1.accuracy_pct + r2.accuracy_pct) / 2.0,
                    "recall": (r1.macro_recall_pct + r2.macro_recall_pct) / 2.0,
                    "f1": (r1.macro_f1_pct + r2.macro_f1_pct) / 2.0,
                }
            )
        for metric in ("accuracy", "recall", "f1"):
            observed = float(np.mean([row[metric] for row in fold_values]))
            expected = float(ref[f"{metric}_mean"])
            rows.append(
                {
                    "architecture": architecture,
                    "metric": metric,
                    "observed": observed,
                    "reference": expected,
                    "absolute_difference": abs(observed - expected),
                    "pass": abs(observed - expected) < 1e-8,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(QA_DIR / "structural_clean_metric_reproduction.csv", index=False, encoding="utf-8-sig")
    if not result["pass"].all():
        raise RuntimeError("Structural clean metrics failed to reproduce coherent reference")
    return result


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURE_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def heatmap(ax, data: pd.DataFrame, cmap: str, center: Optional[float], vmin: Optional[float], vmax: Optional[float], cbar_label: str, fmt: str = ".2f"):
    sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        center=center,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": cbar_label, "shrink": 0.72},
        annot=False,
        fmt=fmt,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")


def make_figures(
    conditions: pd.DataFrame,
    effects: pd.DataFrame,
    recall_corr: pd.DataFrame,
    set_assoc: pd.DataFrame,
    canonical_deltas: pd.DataFrame,
    interaction_recall: pd.DataFrame,
    sample_assoc: pd.DataFrame,
    sample_joined: pd.DataFrame,
    overlap: pd.DataFrame,
) -> None:
    order = conditions.condition_id.tolist()
    labels = conditions.set_index("condition_id").loc[order, "display_label"].tolist()

    # Figure 1: 24 x 5 data-ablation delta recall.
    src1 = effects[effects.effect_type == "data_ablation_full"].copy()
    src1.to_csv(SOURCE_DIR / "fig1_data_ablation_delta_recall.csv", index=False, encoding="utf-8-sig")
    mat1 = src1.pivot(index="effect_id", columns="behavior_label", values="delta_recall_pp").reindex(order)
    mat1.index = labels
    lim = max(1.0, float(np.nanmax(np.abs(mat1.to_numpy()))))
    fig, ax = plt.subplots(figsize=(4.2, 7.2))
    heatmap(ax, mat1, "RdBu_r", 0, -lim, lim, "Δ recall (pp)")
    ax.set_title("Single-timepoint maximum replacement")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", labelsize=6)
    save_figure(fig, "fig1_data_ablation_delta_recall_heatmap")

    # Figure 2: 3 x 5 module-ablation delta recall.
    src2 = effects[effects.effect_type == "module_ablation_clean"].copy()
    src2.to_csv(SOURCE_DIR / "fig2_module_ablation_delta_recall.csv", index=False, encoding="utf-8-sig")
    mat2 = src2.pivot(index="effect_id", columns="behavior_label", values="delta_recall_pp").reindex(MODULES)
    mat2.index = [ARCH_LABELS[m] for m in mat2.index]
    lim = max(1.0, float(np.nanmax(np.abs(mat2.to_numpy()))))
    fig, ax = plt.subplots(figsize=(4.2, 2.2))
    heatmap(ax, mat2, "RdBu_r", 0, -lim, lim, "Δ recall (pp)")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    ax.set_title("Frozen structural ablations")
    save_figure(fig, "fig2_module_ablation_delta_recall_heatmap")

    # Figure 3: recall-vector Spearman.
    src3 = recall_corr.copy()
    src3.to_csv(SOURCE_DIR / "fig3_recall_vector_correlation.csv", index=False, encoding="utf-8-sig")
    mat3 = src3.pivot(index="condition_id", columns="module", values="spearman").reindex(order)[list(MODULES)]
    mat3.index = labels
    mat3.columns = [ARCH_LABELS[m] for m in mat3.columns]
    fig, ax = plt.subplots(figsize=(3.7, 7.2))
    heatmap(ax, mat3, "vlag", 0, -1, 1, "Spearman ρ")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_title("Similarity of 5-behavior ΔRecall vectors")
    save_figure(fig, "fig3_timepoint_module_recall_correlation_heatmap")

    # Figure 4: pooled CW damage-set Jaccard.
    src4 = set_assoc[set_assoc.stage == "all_stages"].copy()
    src4.to_csv(SOURCE_DIR / "fig4_cw_damage_jaccard.csv", index=False, encoding="utf-8-sig")
    mat4 = src4.pivot(index="condition_id", columns="module", values="CW_jaccard").reindex(order)[list(MODULES)]
    mat4.index = labels
    mat4.columns = [ARCH_LABELS[m] for m in mat4.columns]
    observed_max = float(np.nanmax(mat4.to_numpy())) if np.isfinite(mat4.to_numpy()).any() else 1.0
    jaccard_vmax = max(0.01, math.ceil(observed_max * 100.0) / 100.0)
    fig, ax = plt.subplots(figsize=(3.7, 7.2))
    heatmap(ax, mat4, "Blues", None, 0, jaccard_vmax, "CW-set Jaccard")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_title("Damage-set overlap (Run1 + Run2 keys)")
    save_figure(fig, "fig4_cw_damage_set_jaccard_heatmap")

    # Figure 5: canonical behavior ΔFP / ΔFN for data and module ablation.
    src5 = canonical_deltas.copy()
    src5.to_csv(SOURCE_DIR / "fig5_delta_fp_fn.csv", index=False, encoding="utf-8-sig")
    data = src5[src5.effect_type == "data_ablation_full"].copy()
    data["column"] = data.behavior_label + " " + np.where(data.delta_FP.notna(), "", "")
    fp = data.pivot(index="effect_id", columns="behavior_label", values="delta_FP").reindex(order)
    fn = data.pivot(index="effect_id", columns="behavior_label", values="delta_FN").reindex(order)
    comb = pd.concat({"ΔFP": fp, "ΔFN": fn}, axis=1)
    comb = comb.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)
    comb.columns = [f"{b} {metric}" for b, metric in comb.columns]
    comb.index = labels
    lim = max(1.0, float(np.nanmax(np.abs(comb.to_numpy()))))
    fig, ax = plt.subplots(figsize=(7.8, 7.2))
    heatmap(ax, comb, "RdBu_r", 0, -lim, lim, "Count change")
    ax.tick_params(axis="x", rotation=45, labelsize=6)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_title("One-vs-rest error-count changes after data ablation")
    save_figure(fig, "fig5_delta_fp_fn_heatmaps")

    # Figure 6: three module panels of recall interaction.
    src6 = interaction_recall.copy()
    src6.to_csv(SOURCE_DIR / "fig6_double_ablation_interaction.csv", index=False, encoding="utf-8-sig")
    lim = max(1.0, float(np.nanmax(np.abs(src6.interaction_delta_recall_pp.to_numpy()))))
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 7.2), sharey=True)
    for ax, module in zip(axes, MODULES):
        matrix = src6[src6.module == module].pivot(index="condition_id", columns="behavior_label", values="interaction_delta_recall_pp").reindex(order)
        matrix.index = labels
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            vmin=-lim,
            vmax=lim,
            linewidths=0.25,
            linecolor="white",
            cbar=ax is axes[-1],
            cbar_kws={"label": "Interaction in Δ recall (pp)", "shrink": 0.72},
        )
        ax.set_title(ARCH_LABELS[module])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", labelsize=6)
    fig.suptitle("Double-ablation interaction: data effect in Full minus NoM", y=0.995)
    save_figure(fig, "fig6_double_ablation_interaction_heatmap")

    # Figure 7: strongest ΔPtrue sample associations.
    eligible = sample_assoc[(sample_assoc.metric == "true_class_probability") & sample_assoc.spearman.notna()].copy()
    selected = eligible.assign(abs_rho=eligible.spearman.abs()).sort_values("abs_rho", ascending=False).head(6)
    selected_keys = selected[["condition_id", "module", "stage"]].drop_duplicates()
    scatter_src = sample_joined.merge(selected_keys, on=["condition_id", "module", "stage"], how="inner")
    scatter_src.to_csv(SOURCE_DIR / "fig7_selected_delta_ptrue_scatter.csv", index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.8))
    color_map = {
        ("run1", 0): "#4C78A8",
        ("run1", 1): "#F2CF5B",
        ("run1", 2): "#E45756",
        ("run2", 0): "#72B7B2",
        ("run2", 1): "#B279A2",
        ("run2", 2): "#FF9DA6",
    }
    legend_handles: Dict[Tuple[str, int], object] = {}
    for ax, row in zip(axes.ravel(), selected.itertuples()):
        part = scatter_src[
            (scatter_src.condition_id == row.condition_id)
            & (scatter_src.module == row.module)
            & (scatter_src.stage == row.stage)
        ]
        for class_id, group in part.groupby("y_true", sort=True):
            key = (str(row.stage), int(class_id))
            handle = ax.scatter(
                group.true_class_probability_delta_data,
                group.true_class_probability_delta_module,
                s=10,
                alpha=0.6,
                color=color_map[key],
                label=f"{row.stage}: {STAGE_CLASS_NAMES[row.stage][int(class_id)]}",
                edgecolor="none",
            )
            legend_handles[key] = handle
        ax.axhline(0, color="#777777", lw=0.6)
        ax.axvline(0, color="#777777", lw=0.6)
        cond_label = conditions.set_index("condition_id").loc[row.condition_id, "display_label"]
        ax.set_title(f"{cond_label} × {ARCH_LABELS[row.module]} ({row.stage})\nρ={row.spearman:.2f}", fontsize=7)
        ax.set_xlabel("ΔPtrue: data mask")
        ax.set_ylabel("ΔPtrue: module ablation")
    if legend_handles:
        ordered_keys = sorted(legend_handles)
        fig.legend(
            [legend_handles[key] for key in ordered_keys],
            [f"{key[0]}: {STAGE_CLASS_NAMES[key[0]][key[1]]}" for key in ordered_keys],
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, -0.035),
        )
    fig.tight_layout()
    save_figure(fig, "fig7_selected_strong_pairs_delta_ptrue_scatter")

    # QA Figure 8: actual Run1 mask overlap.
    src8 = overlap[overlap.stage == "run1"].copy()
    src8.to_csv(SOURCE_DIR / "fig8_mask_overlap.csv", index=False, encoding="utf-8-sig")
    matrix = src8.pivot(index="condition_a", columns="condition_b", values="jaccard_overlap").reindex(index=order, columns=order)
    matrix.index = labels
    matrix.columns = labels
    fig, ax = plt.subplots(figsize=(8.0, 7.2))
    heatmap(ax, matrix, "mako", None, 0, 1, "Jaccard overlap")
    ax.tick_params(axis="x", rotation=90, labelsize=5.5)
    ax.tick_params(axis="y", labelsize=5.5)
    ax.set_title("Actual ±1 s mask overlap in legacy Run1 windows")
    save_figure(fig, "fig8_mask_overlap_heatmap")


def provenance(datasets: Mapping[str, PointwiseLegacyDataset], checkpoints: pd.DataFrame) -> None:
    try:
        nested_commit = subprocess.check_output(
            ["git", "-C", str(VIDEO_ROOT / "neural-behavior-dual-layer-model"), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        nested_commit = None
    aligned_files = sorted(Path(Config.data_dir).glob("*_aligned.csv.gz"), key=natural_key)
    payload = {
        "analysis_script": str(SCRIPT_PATH),
        "analysis_script_sha256": sha256_file(SCRIPT_PATH),
        "analysis_output_root": str(OUT_ROOT),
        "source_workbook": str(XLSX_PATH),
        "source_workbook_sha256": sha256_file(XLSX_PATH),
        "data_directory": str(Config.data_dir),
        "data_files": [{"path": str(path), "sha256": sha256_file(path)} for path in aligned_files],
        "legacy_model_directory": str(LEGACY_ROOT),
        "checkpoint_manifest": str(QA_DIR / "checkpoint_manifest_sha256.csv"),
        "code_commit": {
            "workspace_D_video": None,
            "workspace_note": "D:/video/.git contains no usable Git repository metadata",
            "nearby_repository": str(VIDEO_ROOT / "neural-behavior-dual-layer-model"),
            "nearby_repository_commit": nested_commit,
            "nearby_repository_note": "Recorded for context only; analysis executed from D:/video/raster and this output directory.",
        },
        "protocol": {
            "window_seconds": 5.0,
            "bin_ms": 10,
            "window_bins": 500,
            "step_bins": 250,
            "stage_label_constant": True,
            "mask_interval": "[center-1s, center+1s), clipped inside each 5-s window",
            "replacement": "per-channel maximum of the original unmodified 5-s window",
            "delta_definition": "ablated - baseline",
            "training_performed": False,
            "inference_device": "cpu (chosen to reproduce archived probability vectors)",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "matplotlib": mpl.__version__,
        },
        "datasets": {
            stage: {
                "n_windows": len(dataset),
                "n_neural_channels": int(dataset.x.shape[1]),
                "window_bins": int(dataset.x.shape[2]),
                "n_mixed_recording_windows": int(dataset.metadata.recording_mixed.sum()),
                "n_discontinuous_windows": int(dataset.metadata.discontinuous.sum()),
            }
            for stage, dataset in datasets.items()
        },
    }
    (QA_DIR / "provenance_and_config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_summary(
    conditions: pd.DataFrame,
    effects: pd.DataFrame,
    recall_corr: pd.DataFrame,
    set_assoc: pd.DataFrame,
    interaction: pd.DataFrame,
    sample_assoc: pd.DataFrame,
    overlap: pd.DataFrame,
    ttest_corr: pd.DataFrame,
) -> None:
    data_effect = effects[effects.effect_type == "data_ablation_full"]
    strongest_data = data_effect.iloc[data_effect.delta_recall_pp.abs().argmax()]
    module_effect = effects[effects.effect_type == "module_ablation_clean"]
    strongest_module = module_effect.iloc[module_effect.delta_recall_pp.abs().argmax()]
    best_corr = recall_corr.iloc[recall_corr.spearman.abs().fillna(-1).argmax()]
    best_j = set_assoc[set_assoc.stage == "all_stages"].sort_values("CW_jaccard", ascending=False).iloc[0]
    best_inter = interaction.iloc[interaction.interaction_delta_recall_pp.abs().argmax()]
    best_sample = sample_assoc.iloc[sample_assoc.spearman.abs().fillna(-1).argmax()]
    high_overlap = overlap[
        (overlap.stage == "run1")
        & (overlap.condition_a != overlap.condition_b)
        & (overlap.jaccard_overlap >= 0.90)
    ].drop_duplicates(["condition_a", "condition_b"])
    text = f"""# 单时间点神经数据消融 × 模块消融：结果摘要

## QA 与协议

- `Summary_P_long` 严格筛选得到 **{len(conditions)}** 个唯一显著时间点，与预设清单完全一致。
- 未训练模型、未重新划分数据。Full、NoConv、NoLSTM、NoAttention 均使用同一组 saved Run1/Run2 三折 test indices。
- legacy 协议保持为 5 s、10 ms/bin、500 bins、250-bin step、stage-label-constant selection。
- point-wise mask 将每个行为 onset 映射回 recording，只替换 `[center-1 s, center+1 s)`；fill 是每个神经通道在该原始 5 s window 内的 maximum。
- clean Full 预测逐样本复现旧保存结果；结构 clean 指标复现 coherent 三折表。详细证据见 `qa/`。

## 数据消融和模块消融的效应

- 绝对值最大的单个 ΔRecall 是 `{strongest_data.effect_id}` 对 `{strongest_data.behavior_label}`：{strongest_data.delta_recall_pp:+.3f} pp（定义始终为 ablated − baseline）。
- 绝对值最大的模块 ΔRecall 是 `{ARCH_LABELS[str(strongest_module.effect_id)]}` 对 `{strongest_module.behavior_label}`：{strongest_module.delta_recall_pp:+.3f} pp。
- 所有原始数值、符号、逐折值和 stage-wise 样本级转移均已保存；符号不是原始数值的替代品。

## Correlation / similarity（相关与相似性）

- 5-behavior ΔRecall 向量相似性中，绝对 Spearman 最大的组合是 `{best_corr.condition_id}` × `{ARCH_LABELS[str(best_corr.module)]}`，ρ={best_corr.spearman:.3f}。
- pooled Run1/Run2 key 的 CW damage-set Jaccard 最大组合是 `{best_j.condition_id}` × `{ARCH_LABELS[str(best_j.module)]}`，J={best_j.CW_jaccard:.3f}。
- 相同 OOF 样本上的 ΔPtrue/Δlogit-margin 关联按 stage 计算；最强绝对 Spearman 为 `{best_sample.condition_id}` × `{ARCH_LABELS[str(best_sample.module)]}`、{best_sample.stage}、{best_sample.metric}，ρ={best_sample.spearman:.3f}。
- 这些量只表示两类干预在输出效应上的一致程度。它们不证明某个模块“编码了”某个生理时间点。

## Double-ablation interaction（双消融交互）

- interaction 定义为 `(MaskFull − CleanFull) − (MaskNoM − CleanNoM)`。
- 绝对值最大的 5-behavior recall interaction 是 `{best_inter.condition_id}` × `{ARCH_LABELS[str(best_inter.module)]}` × `{best_inter.behavior_label}`：{best_inter.interaction_delta_recall_pp:+.3f} pp。
- 非零 interaction 表明数据掩码效应依赖模型结构背景；它与简单相关/相似性是不同证据，不应互相替代。

## Mask overlap 与统计限制

- Run1 中 Jaccard ≥ 0.90 的非对角 point-mask 配对记录数为 {len(high_overlap)}。相邻 L4/L5 显著点的 ±1 s 区间高度重叠，因此不能把每个点解释为独立的时间机制。
- 500-bin windows 以 250 bins 步长重叠。样本不是独立生物重复。本分析不提供把重叠窗口当独立样本的显著性 p 值。
- ΔPtrue/Δmargin 的相关 CI 使用 recording-file block bootstrap；跨 recording 或时间不连续的 legacy window 不进入 CI，但保留在点估计和逐样本 source-data 中。
- t-stat 与消融效应只作为探索性系数，见 `G_ttest_vs_data_ablation_correlations.csv`，不作因果或独立重复推断。

## Causal biological interpretation（因果生物学解释边界）

本结果支持的是 frozen classifier 在既定 legacy protocol 下的功能依赖与结构依赖。输入替换、模块结构消融和双消融交互均是对模型的干预，不是对动物神经回路的实验性因果操纵。即使某个时间点与某个模块呈高相关、高 Jaccard 或强 interaction，也只能表述为模型层面的相容性/依赖性证据。要提出生物学因果结论，仍需独立动物/recording 级验证和直接神经操纵实验。

## 文件导航

- `source_data/condition_predictions_stagewise.csv.gz`：四种结构、clean 与 24 个 point-mask 的逐样本输出。
- `source_data/paired_predictions_stagewise.csv.gz`：baseline/ablated 成对概率、margin、CC/CW/WC/WW、mask hit 与替换 bin 数。
- `tables/`：A–G 关联、ΔFP/ΔFN、confusion、interaction 和指标表。
- `figures/`：SVG/PDF/TIFF/PNG 论文图。
- `qa/`：baseline 复现、fold 对齐、checkpoint hash、mask overlap、路径与环境记录。
"""
    (OUT_ROOT / "RESULTS_SUMMARY_CN.md").write_text(text, encoding="utf-8")


def write_literature_notes() -> None:
    text = """# 实现前方法文献记录

本记录只用于设计与解释边界，不改变本项目已经锁定的 legacy protocol。

1. Zeiler & Fergus (ECCV 2014), *Visualizing and Understanding Convolutional Networks*. DOI: 10.1007/978-3-319-10590-1_53. 经典层/输入遮挡与网络消融工作，支持用预测变化衡量局部输入或模块贡献。
2. Adebayo et al. (NeurIPS 2018), *Sanity Checks for Saliency Maps*. 强调解释方法必须验证其对模型参数和数据关系的敏感性，不能只凭视觉吸引力。
3. Kriegeskorte, Mur & Bandettini (Frontiers in Systems Neuroscience 2008), *Representational similarity analysis—connecting the branches of systems neuroscience*. DOI: 10.3389/neuro.06.004.2008. 提供跨神经数据、行为和模型比较相似结构的框架；本项目用向量/混淆效应相似性，但不把相似性等同于因果。
4. Kornblith et al. (ICML 2019), *Similarity of Neural Network Representations Revisited*. 介绍 CKA 及表示比较的约束。当前任务聚焦输出效应和 intervention interaction，CKA 作为未来表征层扩展，不擅自加入主协议。
5. Šimić, Veas & Sabol (Scientific Reports 2025), *A comprehensive analysis of perturbation methods in explainable AI feature attribution validation for neural time series classifiers*. DOI: 10.1038/s41598-025-09538-2. 指出 time-series faithfulness 对 perturbation choice 和 region size 敏感，支持本项目固定并完整记录 replacement 与区间长度、同时报告 mask overlap。
6. Mercier et al. (2022), *Time to Focus: A Comprehensive Benchmark Using Time Series Attribution Methods*, arXiv:2202.03759. 对时序 attribution 的 perturbation/gradient 方法进行比较，强调不同评估维度没有单一万能方法。

在线来源：

- https://doi.org/10.1007/978-3-319-10590-1_53
- https://papers.nips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html
- https://doi.org/10.3389/neuro.06.004.2008
- https://proceedings.mlr.press/v97/kornblith19a.html
- https://doi.org/10.1038/s41598-025-09538-2
- https://arxiv.org/abs/2202.03759
"""
    (OUT_ROOT / "LITERATURE_NOTES.md").write_text(text, encoding="utf-8")


def write_figure_contract() -> None:
    text = """# Figure contract

Core conclusion: frozen-model single-timepoint perturbations and structural ablations show quantifiable but non-identical output-effect patterns, with structure-dependent double-ablation interactions and strong mask-overlap constraints.

Figure archetype: quantitative grid.

Target/output: manuscript-ready double-column figures; Python/matplotlib-seaborn only; editable SVG/PDF plus 600-dpi TIFF and PNG preview.

Panel map:

- Fig. 1: 24 × 5 data-ablation ΔRecall (primary input-effect evidence).
- Fig. 2: 3 × 5 module-ablation ΔRecall (structural comparison).
- Fig. 3: 24 × 3 recall-vector Spearman (similarity, not causality).
- Fig. 4: 24 × 3 CW-set Jaccard (sample damage overlap).
- Fig. 5: one-vs-rest ΔFP / ΔFN (error decomposition).
- Fig. 6: double-ablation recall interaction (structure dependence).
- Fig. 7: selected ΔPtrue scatter (sample-level association).
- Fig. 8: actual point-mask overlap (critical QA/control).

Statistics: three saved folds; same OOF indices; recording-file block bootstrap for sample-effect Spearman CI; no naive overlapping-window p-values.

Source data: one dedicated CSV per figure under `source_data/`.

Reviewer risks: high ±1 s overlap among adjacent 10-ms centers; hierarchical Run1/Run2 sample definitions; optimistic legacy split; perturbation may be out-of-distribution; model intervention is not biological causality.
"""
    (OUT_ROOT / "FIGURE_CONTRACT.md").write_text(text, encoding="utf-8")


def main() -> None:
    print("[1/12] significant-timepoint QA", flush=True)
    sig = read_significant_timepoints(XLSX_PATH)
    conditions = exact_condition_table(sig)
    write_literature_notes()
    write_figure_contract()

    print("[2/12] build exact legacy datasets with point-wise masks", flush=True)
    configs = {"run1": Config(), "run2": Config3()}
    # The archived prediction CSVs were reproduced to ~1e-6 on CPU, whereas
    # cuDNN produced small probability drift despite identical class outputs.
    # CPU inference is therefore fixed here for probability-level auditability.
    for config in configs.values():
        config.device = "cpu"
    datasets = {
        "run1": PointwiseLegacyDataset(configs["run1"], "run1", conditions),
        "run2": PointwiseLegacyDataset(configs["run2"], "run2", conditions),
    }
    pd.concat([dataset.mapping_detail for dataset in datasets.values()], ignore_index=True).to_csv(
        QA_DIR / "timepoint_mapping_by_recording.csv", index=False, encoding="utf-8-sig"
    )
    coverage = pd.concat([dataset.coverage for dataset in datasets.values()], ignore_index=True)
    coverage.to_csv(QA_DIR / "timepoint_mask_coverage.csv", index=False, encoding="utf-8-sig")
    for stage, dataset in datasets.items():
        dataset.metadata.to_csv(QA_DIR / f"window_recording_metadata_{stage}.csv", index=False, encoding="utf-8-sig")

    print("[3/12] dataset, mask, fold and checkpoint QA", flush=True)
    qa_dataset_equivalence(datasets, conditions)
    qa_fold_partitions(datasets)
    checkpoints = qa_checkpoint_manifest()
    overlap = mask_overlap_tables(datasets, conditions)
    provenance(datasets, checkpoints)

    print("[4/12] frozen-model clean and point-mask inference", flush=True)
    prediction_path = SOURCE_DIR / "condition_predictions_stagewise.csv.gz"
    if "--resume" in sys.argv and prediction_path.exists():
        print(f"[resume] loading {prediction_path}", flush=True)
        predictions = pd.read_csv(prediction_path)
    else:
        predictions = run_inference(datasets, configs, conditions)
    qa_clean_baseline(predictions)

    print("[5/12] condition metrics and clean structural QA", flush=True)
    metrics, confusion = condition_metrics(predictions)
    mean_vectors, effects = legacy_recall_vectors(metrics, conditions)
    qa_structural_metrics(mean_vectors, metrics)

    print("[6/12] paired predictions and metric deltas", flush=True)
    definitions = build_pair_definitions(conditions)
    pairs = paired_predictions(predictions, definitions)
    paired_metrics_df, ovr = paired_metric_tables(pairs)
    canonical_deltas = canonical_fp_fn(ovr, conditions)

    print("[7/12] A/E vector and confusion similarities", flush=True)
    recall_corr = recall_vector_correlations(effects, conditions)
    confusion_delta_similarity(paired_metrics_df, conditions)

    print("[8/12] B sample-effect correlations with recording bootstrap", flush=True)
    sample_assoc, sample_joined = sample_effect_associations(pairs, conditions)

    print("[9/12] C damage/recovery set associations", flush=True)
    set_assoc = transition_set_associations(pairs, conditions)

    print("[10/12] F double-ablation interactions", flush=True)
    interaction_recall, interaction_detail = double_ablation_interactions(predictions, mean_vectors, conditions)

    print("[11/12] G t-test / ablation exploratory association and figures", flush=True)
    ttest_detail, ttest_corr = ttest_ablation_association(conditions, effects, pairs)
    make_figures(
        conditions,
        effects,
        recall_corr,
        set_assoc,
        canonical_deltas,
        interaction_recall,
        sample_assoc,
        sample_joined,
        overlap,
    )

    print("[12/12] summary and completion manifest", flush=True)
    write_summary(
        conditions,
        effects,
        recall_corr,
        set_assoc,
        interaction_recall,
        sample_assoc,
        overlap,
        ttest_corr,
    )
    completion = {
        "status": "complete",
        "training_performed": False,
        "n_timepoints": int(len(conditions)),
        "architectures": list(ARCHITECTURES),
        "stages": list(STAGES),
        "folds": list(FOLDS),
        "prediction_rows": int(len(predictions)),
        "paired_prediction_rows": int(len(pairs)),
        "figures": sorted(path.name for path in FIGURE_DIR.glob("*.png")),
    }
    (QA_DIR / "completion_manifest.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
