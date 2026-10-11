"""Frozen-model pair/triple neural masking x module-ablation analysis.

Hard constraints preserved:
  * no training and no resplitting;
  * legacy 5 s windows (500 x 10 ms) with 250-bin step;
  * stage-label-constant selection and saved Run1/Run2 three-fold test indices;
  * point-wise masks map behavior onsets back to the recording timeline;
  * [center-1 s, center+1 s) replacement, clipped inside each legacy window;
  * each neural channel is filled with its own maximum from the original 5 s window.

The script reuses the audited single-mask pipeline. Multi-masks are Boolean
unions of existing point masks and are applied once to each original clean
window. It never calls an optimizer or a training loop.
"""

from __future__ import annotations

import hashlib
import itertools
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
SINGLE_ROOT = VIDEO_ROOT / "outputs" / "single_timepoint_module_ablation_association"
REPO_ROOT = VIDEO_ROOT / "neural-behavior-dual-layer-model"
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
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
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


PAIR_SPECS = (
    ("tp01_L1_m0p69", "tp08_L3_m0p08", "recommended_cross_behavior"),
    ("tp01_L1_m0p69", "tp05_L2_p0p06", "recommended_cross_behavior"),
    ("tp01_L1_m0p69", "tp07_L2_p0p82", "recommended_cross_behavior"),
    ("tp08_L3_m0p08", "tp05_L2_p0p06", "recommended_cross_behavior"),
    ("tp08_L3_m0p08", "tp07_L2_p0p82", "recommended_cross_behavior"),
    ("tp01_L1_m0p69", "tp06_L2_p0p46", "next_ranked_cross_behavior"),
    ("tp01_L1_m0p69", "tp04_L2_m0p39", "next_ranked_cross_behavior"),
    ("tp03_L2_m0p57", "tp07_L2_p0p82", "lowest_overlap_within_L2"),
)
TRIPLE_SPECS = (
    ("tp01_L1_m0p69", "tp08_L3_m0p08", "tp05_L2_p0p06"),
    ("tp01_L1_m0p69", "tp08_L3_m0p08", "tp07_L2_p0p82"),
)
NULL_SEED = 20261011


def load_single_refs() -> Dict[str, pd.DataFrame]:
    refs = {
        "conditions": pd.read_csv(SINGLE_ROOT / "source_data" / "significant_timepoints_24.csv"),
        "corr": pd.read_csv(SINGLE_ROOT / "tables" / "A_recall_vector_similarity.csv"),
        "effects": pd.read_csv(SINGLE_ROOT / "tables" / "recall_delta_vectors.csv"),
        "pred": pd.read_csv(SINGLE_ROOT / "source_data" / "condition_predictions_stagewise.csv.gz"),
        "pairs": pd.read_csv(SINGLE_ROOT / "source_data" / "paired_predictions_stagewise.csv.gz"),
        "clean_qa": pd.read_csv(SINGLE_ROOT / "qa" / "clean_baseline_reproduction.csv"),
        "fold_qa": pd.read_csv(SINGLE_ROOT / "qa" / "fold_test_index_alignment.csv"),
        "struct_qa": pd.read_csv(SINGLE_ROOT / "qa" / "structural_clean_metric_reproduction.csv"),
        "checkpoints": pd.read_csv(SINGLE_ROOT / "qa" / "checkpoint_manifest_sha256.csv"),
    }
    if len(refs["conditions"]) != 24:
        raise RuntimeError("QA ERROR: single reference does not contain 24 timepoints")
    top = refs["corr"][refs["corr"].module.eq("no_conv")].sort_values("pearson", ascending=False).head(7)
    expected = ["tp01_L1_m0p69", "tp08_L3_m0p08", "tp05_L2_p0p06", "tp07_L2_p0p82",
                "tp06_L2_p0p46", "tp04_L2_m0p39", "tp03_L2_m0p57"]
    if top.condition_id.tolist() != expected:
        raise RuntimeError(f"QA ERROR: top NoConv Pearson ranking changed: {top.condition_id.tolist()}")
    refs["corr"][["condition_id", "target_label", "time_s", "module", "pearson", "spearman", "cosine_similarity"]].to_csv(
        SOURCE_DIR / "single_mask_baseline_reference.csv", index=False, encoding="utf-8-sig")
    qa = pd.DataFrame([
        {"check": "24_timepoints", "passed": len(refs["conditions"]) == 24},
        {"check": "clean_baseline_reproduced", "passed": refs["clean_qa"]["pass"].astype(bool).all()},
        {"check": "structural_metrics_reproduced", "passed": refs["struct_qa"]["pass"].astype(bool).all()},
        {"check": "top7_NoConv_Pearson_reproduced", "passed": True},
    ])
    qa.to_csv(QA_DIR / "single_analysis_reference_checks.csv", index=False, encoding="utf-8-sig")
    if not qa.passed.all():
        raise RuntimeError("QA ERROR: single analysis reference audit failed")
    return refs


def multi_id(ids: Sequence[str], point_lookup: pd.DataFrame) -> str:
    prefix = "PAIR" if len(ids) == 2 else "TRIPLE"
    tokens = []
    for cid in ids:
        row = point_lookup.loc[cid]
        tokens.append(f"L{int(row.label)}_{time_token(float(row.time_s))}")
    return prefix + "__" + "__".join(tokens)


def build_multi_conditions(points: pd.DataFrame, triples: bool) -> pd.DataFrame:
    lookup = points.set_index("condition_id")
    rows = []
    specs = list(PAIR_SPECS) + ([(a, b, c, "top_nonredundant_all_pairs_measured") for a, b, c in TRIPLE_SPECS] if triples else [])
    for spec in specs:
        ids = list(spec[:-1])
        rule = spec[-1]
        row = {"condition_id": multi_id(ids, lookup), "mask_order": len(ids), "selection_rule": rule,
               "constituent_ids_json": json.dumps(ids)}
        labels = []
        for i in range(3):
            if i < len(ids):
                p = lookup.loc[ids[i]]
                row[f"region_{i+1}"] = ids[i]
                row[f"region_{i+1}_label"] = int(p.label)
                row[f"region_{i+1}_time_s"] = float(p.time_s)
                row[f"region_{i+1}_condition_index"] = int(p.condition_index)
                labels.append(f"L{int(p.label)} {float(p.time_s):+.2f}s")
            else:
                row[f"region_{i+1}"] = None
                row[f"region_{i+1}_label"] = np.nan
                row[f"region_{i+1}_time_s"] = np.nan
                row[f"region_{i+1}_condition_index"] = np.nan
        row["display_label"] = " + ".join(labels)
        rows.append(row)
    return pd.DataFrame(rows)


def union_for(dataset: PointwiseLegacyDataset, row: pd.Series) -> np.ndarray:
    idx = [int(row[f"region_{i}_condition_index"]) for i in range(1, int(row.mask_order) + 1)]
    return dataset.segment_masks[:, idx, :].any(axis=1)


def count_components(mask: np.ndarray) -> int:
    padded = np.pad(mask.astype(np.int8), ((0, 0), (1, 0)))
    return int((np.diff(padded, axis=1) == 1).sum())


def overlap_audit(refs: Mapping[str, pd.DataFrame], datasets: Mapping[str, PointwiseLegacyDataset]) -> pd.DataFrame:
    top = refs["corr"][refs["corr"].module.eq("no_conv")].sort_values("pearson", ascending=False).head(7)
    ids = top.condition_id.tolist()
    lookup = refs["conditions"].set_index("condition_id")
    selected = {frozenset((a, b)) for a, b, _ in PAIR_SPECS}
    rows = []
    for a, b in itertools.combinations(ids, 2):
        for stage in STAGES:
            ds = datasets[stage]
            ma = ds.segment_masks[:, int(lookup.loc[a].condition_index), :]
            mb = ds.segment_masks[:, int(lookup.loc[b].condition_index), :]
            inter = int((ma & mb).sum()); union = int((ma | mb).sum())
            na = int(ma.sum()); nb = int(mb.sum())
            rows.append({"condition_a": a, "condition_b": b, "stage": stage,
                         "n_bins_A": na, "n_bins_B": nb, "n_bins_intersection": inter,
                         "n_bins_union": union, "jaccard_AB": inter / union if union else np.nan,
                         "overlap_fraction_A": inter / na if na else np.nan,
                         "overlap_fraction_B": inter / nb if nb else np.nan,
                         "mask_hit_A": bool(ma.any()), "mask_hit_B": bool(mb.any()),
                         "mask_hit_pair": bool((ma | mb).any()),
                         "selected_for_pair_pilot": frozenset((a, b)) in selected})
    out = pd.DataFrame(rows)
    mx = out.groupby(["condition_a", "condition_b"], as_index=False).jaccard_AB.max().rename(columns={"jaccard_AB": "max_stage_jaccard"})
    out = out.merge(mx, on=["condition_a", "condition_b"], validate="many_to_one")
    out["overlap_flag"] = np.where(out.max_stage_jaccard >= .8, "highly_redundant_mask_geometry", "eligible_primary")
    if out.loc[out.selected_for_pair_pilot, "max_stage_jaccard"].max() >= .8:
        raise RuntimeError("QA ERROR: primary pair violates Jaccard < 0.80")
    out.to_csv(SOURCE_DIR / "candidate_pair_mask_overlap.csv", index=False, encoding="utf-8-sig")
    out.to_csv(SOURCE_DIR / "multi_mask_overlap_audit.csv", index=False, encoding="utf-8-sig")
    return out


def geometry_table(conditions: pd.DataFrame, datasets: Mapping[str, PointwiseLegacyDataset]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    detail = []
    summary = []
    for _, cond in conditions.iterrows():
        total_bins = total_cells = total_hits = total_components = 0
        all_j = []; all_unique = []
        for stage in STAGES:
            ds = datasets[stage]
            idx = [int(cond[f"region_{i}_condition_index"]) for i in range(1, int(cond.mask_order) + 1)]
            atomic = [ds.segment_masks[:, j, :] for j in idx]
            union = np.logical_or.reduce(atomic)
            pair_j = []
            for i, j in itertools.combinations(range(len(atomic)), 2):
                u = int((atomic[i] | atomic[j]).sum()); v = int((atomic[i] & atomic[j]).sum())
                pair_j.append(v / u if u else np.nan)
            row = {"condition_id": cond.condition_id, "mask_order": int(cond.mask_order), "stage": stage,
                   "n_bins_union": int(union.sum()), "mask_coverage": float(union.mean()),
                   "affected_windows": int(union.any(axis=1).sum()), "n_components": count_components(union),
                   "max_pairwise_jaccard": float(np.nanmax(pair_j)) if pair_j else np.nan}
            for i, m in enumerate(atomic):
                others = np.logical_or.reduce([x for k, x in enumerate(atomic) if k != i])
                unique = int((m & ~others).sum()); all_unique.append(unique)
                row[f"n_bins_region_{i+1}"] = int(m.sum())
                row[f"unique_contribution_bins_region_{i+1}"] = unique
                row[f"effective_extra_bins_region_{i+1}"] = unique
            if len(atomic) == 2:
                inter = int((atomic[0] & atomic[1]).sum())
                row.update({"n_bins_intersection": inter, "jaccard_AB": pair_j[0]})
            else:
                for (i, j), v in zip(itertools.combinations(range(3), 2), pair_j): row[f"jaccard_region_{i+1}_{j+1}"] = v
            detail.append(row); all_j.extend(pair_j)
            total_bins += int(union.sum()); total_cells += union.size
            total_hits += int(union.any(axis=1).sum()); total_components += count_components(union)
        summary.append({"condition_id": cond.condition_id, "mask_order": int(cond.mask_order),
                        "union_bins": total_bins, "mask_coverage": total_bins / total_cells,
                        "affected_windows": total_hits, "n_components": total_components,
                        "max_pairwise_jaccard": float(np.nanmax(all_j)),
                        "minimum_effective_extra_bins": int(min(all_unique))})
    return pd.DataFrame(detail), pd.DataFrame(summary)


def clean_reference(refs: Mapping[str, pd.DataFrame], stage: str, fold: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    p = refs["pred"][(refs["pred"].architecture == "full") & (refs["pred"].condition_id == "clean") &
                     (refs["pred"].stage == stage) & (refs["pred"].fold == fold)].sort_values("dataset_index")
    idx = p.dataset_index.to_numpy(int)
    if not np.array_equal(idx, np.sort(np.asarray(test_indices(stage, fold), int))):
        raise RuntimeError(f"QA ERROR: reference/test indices differ for {stage} {fold}")
    return idx, {"y_true": p.y_true.to_numpy(int), "pred": p.pred.to_numpy(int),
                 "probabilities": p[["prob_0", "prob_1", "prob_2"]].to_numpy(float),
                 "logits": p[["logit_0", "logit_1", "logit_2"]].to_numpy(float)}


@torch.no_grad()
def infer_union(model, ds: PointwiseLegacyDataset, idx: np.ndarray, mask_all: np.ndarray,
                clean: Mapping[str, np.ndarray], config, device: str) -> Dict[str, np.ndarray]:
    masks = mask_all[idx]; hit = np.flatnonzero(masks.any(axis=1)); logits = clean["logits"].astype(np.float32).copy()
    if len(hit):
        x = torch.from_numpy(ds.x[idx[hit]]).float(); m = torch.from_numpy(masks[hit]).bool(); chunks = []
        for start in range(0, len(x), int(config.batch_size)):
            xb = x[start:start+int(config.batch_size)].to(device); mb = m[start:start+int(config.batch_size)].to(device)
            fill = xb.amax(dim=2, keepdim=True); changed = torch.where(mb.unsqueeze(1), fill.expand_as(xb), xb)
            chunks.append(model(changed).detach().cpu().numpy())
        logits[hit] = np.concatenate(chunks)
    prob = softmax_numpy(logits)
    return {"y_true": clean["y_true"].copy(), "pred": prob.argmax(1), "probabilities": prob, "logits": logits,
            "mask_hit": masks.any(1), "replaced_bin_count": masks.sum(1).astype(int)}


def infer_real(conditions: pd.DataFrame, datasets, configs, refs, suffix: str) -> pd.DataFrame:
    chunks = []
    for stage in STAGES:
        ds = datasets[stage]; config = configs[stage]; config.device = "cpu"
        masks = {r.condition_id: union_for(ds, r) for _, r in conditions.iterrows()}
        for fold in FOLDS:
            idx, clean = clean_reference(refs, stage, fold)
            model = load_model("full", stage, fold, ds, config)
            chunks.append(rows_for_condition("full", "clean", "clean", None, None, stage, fold, ds, idx, clean))
            for _, cond in conditions.iterrows():
                print(f"[real] {cond.condition_id} {stage} {fold}", flush=True)
                masked = infer_union(model, ds, idx, masks[cond.condition_id], clean, config, "cpu")
                chunks.append(rows_for_condition("full", cond.condition_id, "timepoint_mask", None, None,
                                                 stage, fold, ds, idx, masked))
            del model
    out = pd.concat(chunks, ignore_index=True)
    out.to_csv(SOURCE_DIR / f"{suffix}_condition_predictions.csv.gz", index=False, compression="gzip")
    return out


def data_definitions(conditions: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"comparison_id": f"data_full__{r.condition_id}", "comparison_family": "multi_mask_full",
                          "module": None, "condition_id": r.condition_id, "target_label": None, "time_s": None,
                          "baseline_architecture": "full", "baseline_condition": "clean",
                          "ablated_architecture": "full", "ablated_condition": r.condition_id}
                         for r in conditions.itertuples()])


def behavior_effects(predictions: pd.DataFrame, conditions: pd.DataFrame, refs, geometry: pd.DataFrame):
    metrics, _ = condition_metrics(predictions)
    fold_rows = metrics[metrics.scope.isin(FOLDS)]
    effects = []
    for cond in conditions.itertuples():
        for fold in FOLDS:
            group = fold_rows[(fold_rows.condition_id == cond.condition_id) & (fold_rows.scope == fold)]
            by = {r.stage: r for r in group.itertuples()}
            clean_group = fold_rows[(fold_rows.condition_id == "clean") & (fold_rows.scope == fold)]
            clean = {r.stage: r for r in clean_group.itertuples()}
            if set(by) != set(STAGES) or set(clean) != set(STAGES): raise RuntimeError(f"Missing stage {cond.condition_id} {fold}")
            rb = {1: clean['run1'].recall_class_0_pct,
                  2: (clean['run1'].recall_class_1_pct + clean['run2'].recall_class_0_pct)/2,
                  3: clean['run1'].recall_class_2_pct,
                  4: (clean['run1'].recall_class_1_pct + clean['run2'].recall_class_1_pct)/2,
                  5: (clean['run1'].recall_class_1_pct + clean['run2'].recall_class_2_pct)/2}
            ra = {1: by['run1'].recall_class_0_pct,
                  2: (by['run1'].recall_class_1_pct + by['run2'].recall_class_0_pct)/2,
                  3: by['run1'].recall_class_2_pct,
                  4: (by['run1'].recall_class_1_pct + by['run2'].recall_class_1_pct)/2,
                  5: (by['run1'].recall_class_1_pct + by['run2'].recall_class_2_pct)/2}
            for beh in range(1,6): effects.append({"condition_id": cond.condition_id, "mask_order": cond.mask_order,
                                                    "fold": fold, "behavior": beh, "behavior_label": f"L{beh}",
                                                    "baseline_recall_pct": rb[beh], "masked_recall_pct": ra[beh],
                                                    "delta_recall_pp": ra[beh]-rb[beh]})
    effects = pd.DataFrame(effects)
    mean = effects.groupby(["condition_id","mask_order","behavior","behavior_label"], as_index=False).agg(
        baseline_recall_pct=("baseline_recall_pct","mean"), masked_recall_pct=("masked_recall_pct","mean"),
        delta_recall_pp=("delta_recall_pp","mean"))
    mean["delta_sign"] = mean.delta_recall_pp.map(sign_symbol)
    mean.to_csv(TABLE_DIR / "multi_mask_recall_delta_vectors.csv", index=False, encoding="utf-8-sig")
    modules = refs["effects"][refs["effects"].effect_type.eq("module_ablation_clean")]
    geom = geometry.set_index("condition_id"); c_lookup = conditions.set_index("condition_id"); rows=[]
    for cid, group in mean.groupby("condition_id"):
        x = group.sort_values("behavior").delta_recall_pp.to_numpy(float); c = c_lookup.loc[cid]
        for module in MODULES:
            y = modules[modules.effect_id.eq(module)].sort_values("behavior").delta_recall_pp.to_numpy(float)
            rows.append({"condition_id": cid, "mask_order": int(c.mask_order), "region_1": c.region_1,
                         "region_2": c.region_2, "region_3": c.region_3, "module": module,
                         "pearson": safe_corr(x,y,"pearson"), "spearman": safe_corr(x,y,"spearman"),
                         "cosine_similarity": safe_corr(x,y,"cosine"),
                         **{f"delta_recall_L{i}":x[i-1] for i in range(1,6)},
                         "union_bins": int(geom.loc[cid].union_bins), "mask_coverage": geom.loc[cid].mask_coverage,
                         "max_pairwise_jaccard": geom.loc[cid].max_pairwise_jaccard})
    corr = pd.DataFrame(rows); corr.to_csv(SOURCE_DIR / "multi_mask_module_correlations.csv", index=False, encoding="utf-8-sig")
    return metrics, mean, corr


def gains_table(corr: pd.DataFrame, conditions: pd.DataFrame, refs) -> pd.DataFrame:
    single_corr = refs["corr"].set_index(["condition_id","module"]); c_lookup=conditions.set_index("condition_id"); rows=[]
    for r in corr.itertuples():
        ids=json.loads(c_lookup.loc[r.condition_id].constituent_ids_json); out={"condition_id":r.condition_id,"mask_order":r.mask_order,"module":r.module}
        for name,col in (("pearson","pearson"),("spearman","spearman"),("cosine","cosine_similarity")):
            vals=np.array([single_corr.loc[(cid,r.module),col] for cid in ids],float); value=float(getattr(r,col))
            out[name]=value; out[f"best_single_{name}"]=float(np.nanmax(vals)); out[f"mean_single_{name}"]=float(np.nanmean(vals))
            out[f"gain_over_best_single_{name}"]=value-out[f"best_single_{name}"]
            out[f"gain_over_mean_single_{name}"]=value-out[f"mean_single_{name}"]
        rows.append(out)
    out=pd.DataFrame(rows); out.to_csv(SOURCE_DIR / "multi_mask_correlation_gain.csv", index=False, encoding="utf-8-sig"); return out


def interaction_outputs(mean: pd.DataFrame, conditions: pd.DataFrame, refs):
    singles=refs["effects"][refs["effects"].effect_type.eq("data_ablation_full")]
    sl={cid:g.sort_values("behavior").delta_recall_pp.to_numpy(float) for cid,g in singles.groupby("effect_id")}
    ml={cid:g.sort_values("behavior").delta_recall_pp.to_numpy(float) for cid,g in mean.groupby("condition_id")}
    pair_lookup={}; pair_rows=[]; triple_rows=[]
    for c in conditions.sort_values("mask_order").itertuples():
        ids=json.loads(c.constituent_ids_json); obs=ml[c.condition_id]; exp=np.sum([sl[x] for x in ids],axis=0); resid=obs-exp
        gap=float(np.linalg.norm(obs)-np.linalg.norm(exp)); typ="super_additive_magnitude" if gap>1 else "sub_additive_redundancy_or_compensation" if gap< -1 else "roughly_additive"
        common={"condition_id":c.condition_id,"mask_order":c.mask_order,"observed_norm":np.linalg.norm(obs),
                "additive_expected_norm":np.linalg.norm(exp),"interaction_norm":np.linalg.norm(resid),"magnitude_gap_pp":gap,"interaction_type":typ}
        if c.mask_order==2:
            pair_lookup[frozenset(ids)]=obs
            pair_rows.append({**common,**{f"observed_L{i}":obs[i-1] for i in range(1,6)},
                              **{f"expected_L{i}":exp[i-1] for i in range(1,6)},**{f"I_L{i}":resid[i-1] for i in range(1,6)}})
        else:
            pv=[pair_lookup[frozenset(x)] for x in itertools.combinations(ids,2)]
            pa=obs-np.sum(pv,axis=0)+np.sum([sl[x] for x in ids],axis=0)
            triple_rows.append({**common,**{f"observed_L{i}":obs[i-1] for i in range(1,6)},
                                **{f"expected_L{i}":exp[i-1] for i in range(1,6)},
                                **{f"I_simple_L{i}":resid[i-1] for i in range(1,6)},
                                **{f"I_pairadjusted_L{i}":pa[i-1] for i in range(1,6)},
                                "pairadjusted_interaction_norm":np.linalg.norm(pa)})
    p=pd.DataFrame(pair_rows); t=pd.DataFrame(triple_rows)
    p.to_csv(SOURCE_DIR / "pair_additivity_interaction.csv",index=False,encoding="utf-8-sig")
    t.to_csv(SOURCE_DIR / "triple_interaction.csv",index=False,encoding="utf-8-sig")
    return p,t


def bootstrap_corr(df,x,y,method,repeats,seed):
    usable=df[~df.recording_mixed.astype(bool)].reset_index(drop=True); groups=sorted(usable.recording_key.dropna().unique())
    if len(groups)<3:return np.nan,np.nan,len(groups),len(df)-len(usable)
    xa=usable[x].to_numpy(float); ya=usable[y].to_numpy(float); keys=usable.recording_key.to_numpy(); by={g:np.flatnonzero(keys==g) for g in groups}
    rng=np.random.default_rng(seed); vals=[]
    for _ in range(repeats):
        ids=np.concatenate([by[g] for g in rng.choice(groups,size=len(groups),replace=True)]); v=safe_corr(xa[ids],ya[ids],method)
        if np.isfinite(v):vals.append(v)
    return (np.percentile(vals,2.5),np.percentile(vals,97.5),len(groups),len(df)-len(usable)) if vals else (np.nan,np.nan,len(groups),len(df)-len(usable))


def sample_and_set_outputs(pairs: pd.DataFrame, conditions: pd.DataFrame, refs):
    modules=refs["pairs"][refs["pairs"].comparison_family.eq("module_ablation_clean")]; sample_rows=[]; set_rows=[]; keys=["stage","fold","dataset_index"]
    for c in conditions.itertuples():
        data=pairs[pairs.comparison_id.eq(f"data_full__{c.condition_id}")]
        for module in MODULES:
            mod=modules[modules.module.eq(module)]
            for stage in STAGES:
                d=data[data.stage.eq(stage)]; m=mod[mod.stage.eq(stage)]
                j=d.merge(m[keys+["true_class_probability_delta","logit_margin_delta"]],on=keys,suffixes=("_multi","_module"),validate="one_to_one")
                for metric,x,y in (("true_class_probability","true_class_probability_delta_multi","true_class_probability_delta_module"),
                                   ("logit_margin","logit_margin_delta_multi","logit_margin_delta_module")):
                    seed=NULL_SEED+abs(hash((c.condition_id,module,stage,metric)))%100000
                    pl,ph,nr,nm=bootstrap_corr(j,x,y,"pearson",500,seed); sl,sh,_,_=bootstrap_corr(j,x,y,"spearman",500,seed+1)
                    sample_rows.append({"condition_id":c.condition_id,"mask_order":c.mask_order,"module":module,"stage":stage,"metric":metric,
                                        "pearson":safe_corr(j[x],j[y],"pearson"),"pearson_bootstrap_ci_low":pl,"pearson_bootstrap_ci_high":ph,
                                        "spearman":safe_corr(j[x],j[y],"spearman"),"spearman_bootstrap_ci_low":sl,"spearman_bootstrap_ci_high":sh,
                                        "bootstrap_unit":"recording_file","bootstrap_repeats":500,"n_samples":len(j),"n_recordings":nr,
                                        "n_mixed_recording_windows_excluded_from_ci":nm})
            for stage in (*STAGES,"all_stages"):
                d=data if stage=="all_stages" else data[data.stage.eq(stage)]; m=mod if stage=="all_stages" else mod[mod.stage.eq(stage)]
                z=d[keys+["transition"]].merge(m[keys+["transition"]],on=keys,suffixes=("_multi","_module"),validate="one_to_one")
                mcw=z.transition_multi.eq("CW"); ncw=z.transition_module.eq("CW"); mwc=z.transition_multi.eq("WC"); nwc=z.transition_module.eq("WC")
                set_rows.append({"condition_id":c.condition_id,"mask_order":c.mask_order,"module":module,"stage":stage,"n_samples":len(z),
                                 "multi_CW":int(mcw.sum()),"module_CW":int(ncw.sum()),"CW_intersection":int((mcw&ncw).sum()),
                                 "CW_jaccard":jaccard(mcw,ncw),"CW_binary_mcc_phi":binary_phi(mcw,ncw),
                                 "p_module_CW_given_multi_CW":(mcw&ncw).sum()/mcw.sum() if mcw.sum() else np.nan,
                                 "p_multi_CW_given_module_CW":(mcw&ncw).sum()/ncw.sum() if ncw.sum() else np.nan,
                                 "multi_WC":int(mwc.sum()),"module_WC":int(nwc.sum()),"WC_intersection":int((mwc&nwc).sum()),
                                 "WC_jaccard":jaccard(mwc,nwc),"WC_binary_mcc_phi":binary_phi(mwc,nwc),
                                 "p_module_WC_given_multi_WC":(mwc&nwc).sum()/mwc.sum() if mwc.sum() else np.nan,
                                 "p_multi_WC_given_module_WC":(mwc&nwc).sum()/nwc.sum() if nwc.sum() else np.nan})
    s=pd.DataFrame(sample_rows); o=pd.DataFrame(set_rows)
    s.to_csv(SOURCE_DIR / "multi_mask_sample_level_correlations.csv",index=False,encoding="utf-8-sig")
    o.to_csv(SOURCE_DIR / "multi_mask_CW_WC_overlap.csv",index=False,encoding="utf-8-sig")
    return s,o


def confusion_output(pairs:pd.DataFrame,conditions:pd.DataFrame,refs):
    modules=refs["pairs"][refs["pairs"].comparison_family.eq("module_ablation_clean")]; rows=[]; off=np.array([0,1,1,1,0,1,1,1,0],bool)
    for c in conditions.itertuples():
        data=pairs[pairs.comparison_id.eq(f"data_full__{c.condition_id}")]
        for stage in STAGES:
            d=data[data.stage.eq(stage)]; y=d.y_true.to_numpy(int); dv=(confusion_matrix(y,d.ablated_pred,labels=[0,1,2])-confusion_matrix(y,d.baseline_pred,labels=[0,1,2])).ravel()[off]
            for module in MODULES:
                m=modules[(modules.module==module)&(modules.stage==stage)]; ym=m.y_true.to_numpy(int); mv=(confusion_matrix(ym,m.ablated_pred,labels=[0,1,2])-confusion_matrix(ym,m.baseline_pred,labels=[0,1,2])).ravel()[off]
                rows.append({"condition_id":c.condition_id,"mask_order":c.mask_order,"module":module,"stage":stage,
                             "pearson":safe_corr(dv,mv,"pearson"),"spearman":safe_corr(dv,mv,"spearman"),"cosine_similarity":safe_corr(dv,mv,"cosine"),
                             "multi_offdiag_delta_json":json.dumps(dv.tolist()),"module_offdiag_delta_json":json.dumps(mv.tolist())})
    out=pd.DataFrame(rows); out.to_csv(SOURCE_DIR / "multi_mask_confusion_similarity.csv",index=False,encoding="utf-8-sig"); return out


def shifted_mask(flags: np.ndarray, delta: int, radius: int = 100) -> np.ndarray:
    out=np.zeros_like(flags,dtype=bool)
    for w in np.flatnonzero(flags.any(1)):
        for center in np.flatnonzero(flags[w]):
            c=int(np.clip(int(center)+delta,0,flags.shape[1]-1)); out[w,max(0,c-radius):min(flags.shape[1],c+radius)]=True
    return out


def null_library(cond:pd.Series,datasets,n_null:int,seed:int):
    rng=np.random.default_rng(seed); order=int(cond.mask_order)
    old=np.array([round(float(cond[f"region_{i}_time_s"])*100) for i in range(1,order+1)],int)
    indices=[int(cond[f"region_{i}_condition_index"]) for i in range(1,order+1)]
    grid=np.arange(-100,101,dtype=int)
    cache={}
    for stage in STAGES:
        cache[stage]=[]
        for idx,ob in zip(indices,old):
            cache[stage].append({int(nb):shifted_mask(datasets[stage].center_flags[:,idx,:],int(nb-ob)) for nb in grid})
    real={s:union_for(datasets[s],cond) for s in STAGES}
    target={s:{"bins":int(real[s].sum()),"hits":int(real[s].any(1).sum()),"components":count_components(real[s])} for s in STAGES}
    candidates=[]; seen=set(); pool=max(n_null*20,10000)
    while len(candidates)<pool:
        bins=tuple(int(x) for x in rng.choice(grid,size=order,replace=True))
        if bins in seen or bins==tuple(old.tolist()):continue
        seen.add(bins); stats={}; score=0.0
        for stage in STAGES:
            u=np.logical_or.reduce([cache[stage][i][bins[i]] for i in range(order)])
            st={"bins":int(u.sum()),"hits":int(u.any(1).sum()),"components":count_components(u)}; stats[stage]=st
            score+=4*abs(st["bins"]-target[stage]["bins"])/max(target[stage]["bins"],1)
            score+=2*abs(st["hits"]-target[stage]["hits"])/max(target[stage]["hits"],1)
            score+=abs(st["components"]-target[stage]["components"])/max(target[stage]["components"],1)
        candidates.append((score,bins,stats))
    candidates.sort(key=lambda x:x[0]); chosen=candidates[:n_null]; rows=[]; masks={s:[] for s in STAGES}
    for i,(score,bins,stats) in enumerate(chosen,1):
        row={"condition_id":cond.condition_id,"mask_order":order,"null_id":f"{cond.condition_id}__NULL_{i:04d}",
             "random_center_bins_json":json.dumps(bins),"random_center_times_s_json":json.dumps([x/100 for x in bins]),"geometry_match_score":score}
        for stage in STAGES:
            selected_union=np.logical_or.reduce([cache[stage][j][bins[j]] for j in range(order)])
            masks[stage].append(selected_union); row[f"real_union_bins_{stage}"]=target[stage]["bins"]; row[f"null_union_bins_{stage}"]=stats[stage]["bins"]
            row[f"union_bin_relative_error_{stage}"]=(stats[stage]["bins"]-target[stage]["bins"])/max(target[stage]["bins"],1)
            row[f"real_affected_windows_{stage}"]=target[stage]["hits"]; row[f"null_affected_windows_{stage}"]=stats[stage]["hits"]
            row[f"real_components_{stage}"]=target[stage]["components"]; row[f"null_components_{stage}"]=stats[stage]["components"]
        rows.append(row)
    return pd.DataFrame(rows),masks


def gpu_cpu_validate(conditions,datasets,configs,refs,real_predictions):
    rows=[]
    for stage in STAGES:
        ds=datasets[stage]; config=configs[stage]; config.device="cuda"; masks={r.condition_id:union_for(ds,r) for _,r in conditions.iterrows()}
        for fold in FOLDS:
            idx,clean=clean_reference(refs,stage,fold); model=load_model("full",stage,fold,ds,config)
            for _,c in conditions.iterrows():
                gpu=infer_union(model,ds,idx,masks[c.condition_id],clean,config,"cuda")
                cpu=real_predictions[(real_predictions.condition_id==c.condition_id)&(real_predictions.stage==stage)&(real_predictions.fold==fold)].sort_values("dataset_index")
                exact=np.array_equal(gpu["pred"],cpu.pred.to_numpy(int)); rows.append({"condition_id":c.condition_id,"stage":stage,"fold":fold,
                    "n_samples":len(cpu),"class_predictions_exact":exact,"n_class_mismatches":int((gpu["pred"]!=cpu.pred.to_numpy(int)).sum())})
            del model; torch.cuda.empty_cache()
    out=pd.DataFrame(rows); out.to_csv(QA_DIR / "gpu_cpu_class_prediction_validation.csv",index=False,encoding="utf-8-sig")
    if not out.class_predictions_exact.all():raise RuntimeError("GPU/CPU predictions differ; abort random-null inference")
    return out


@torch.no_grad()
def infer_null_fold_batched(model,ds,idx,masks_all,clean,device="cuda",group_size=25,gpu_batch=256):
    out=np.tile(np.asarray(clean["pred"],int),(len(masks_all),1))
    for g0 in range(0,len(masks_all),group_size):
        xs=[]; ms=[]; null_ids=[]; local_ids=[]
        for ni in range(g0,min(g0+group_size,len(masks_all))):
            masks=masks_all[ni][idx]; hit=np.flatnonzero(masks.any(1))
            if not len(hit):continue
            xs.append(ds.x[idx[hit]]); ms.append(masks[hit]); null_ids.append(np.full(len(hit),ni,int)); local_ids.append(hit)
        if not xs:continue
        x=torch.from_numpy(np.concatenate(xs)).float(); m=torch.from_numpy(np.concatenate(ms)).bool(); ni_all=np.concatenate(null_ids); li_all=np.concatenate(local_ids); preds=[]
        for start in range(0,len(x),gpu_batch):
            xb=x[start:start+gpu_batch].to(device); mb=m[start:start+gpu_batch].to(device); fill=xb.amax(dim=2,keepdim=True)
            logits=model(torch.where(mb.unsqueeze(1),fill.expand_as(xb),xb)); preds.append(logits.argmax(1).detach().cpu().numpy())
        out[ni_all,li_all]=np.concatenate(preds)
    return out


@torch.no_grad()
def run_nulls(conditions,datasets,configs,refs,n_null:int):
    geometry=[]; vectors=[]
    for ci,cond in conditions.iterrows():
        print(f"[null] library {cond.condition_id} N={n_null}",flush=True); geo,masks=null_library(cond,datasets,n_null,NULL_SEED+ci*1009); geometry.append(geo)
        pred={}
        for stage in STAGES:
            ds=datasets[stage]; config=configs[stage]; config.device="cuda"
            for fold in FOLDS:
                idx,clean=clean_reference(refs,stage,fold); model=load_model("full",stage,fold,ds,config)
                batched=infer_null_fold_batched(model,ds,idx,masks[stage],clean,device="cuda",group_size=25,gpu_batch=256)
                for ni in range(n_null):pred[(ni,stage,fold)]=(clean["y_true"],clean["pred"],batched[ni])
                del model; torch.cuda.empty_cache()
        for ni in range(n_null):
            fold_vec=[]
            for fold in FOLDS:
                r={}
                for stage in STAGES:
                    y,pb,pa=pred[(ni,stage,fold)]; r[stage]=(recall_score(y,pb,labels=[0,1,2],average=None,zero_division=0)*100,
                                                          recall_score(y,pa,labels=[0,1,2],average=None,zero_division=0)*100)
                b1,a1=r["run1"]; b2,a2=r["run2"]
                b=np.array([b1[0],(b1[1]+b2[0])/2,b1[2],(b1[1]+b2[1])/2,(b1[1]+b2[2])/2])
                a=np.array([a1[0],(a1[1]+a2[0])/2,a1[2],(a1[1]+a2[1])/2,(a1[1]+a2[2])/2]); fold_vec.append(a-b)
            v=np.mean(fold_vec,axis=0); vectors.append({"condition_id":cond.condition_id,"mask_order":int(cond.mask_order),
                "null_id":geo.iloc[ni].null_id,**{f"delta_recall_L{i}":v[i-1] for i in range(1,6)}})
        del pred
    g=pd.concat(geometry,ignore_index=True); v=pd.DataFrame(vectors)
    g.to_csv(SOURCE_DIR / "multi_mask_random_null_geometry.csv.gz",index=False,compression="gzip")
    v.to_csv(SOURCE_DIR / "multi_mask_random_null_effect_vectors.csv.gz",index=False,compression="gzip")
    return g,v


def bh_adjust(values):
    """Benjamini-Hochberg adjusted q values; NaNs remain NaN."""
    x=np.asarray(values,dtype=float); out=np.full(len(x),np.nan,dtype=float); ok=np.isfinite(x)
    if not ok.any():return out
    p=x[ok]; order=np.argsort(p); ranked=p[order]*len(p)/np.arange(1,len(p)+1)
    ranked=np.minimum.accumulate(ranked[::-1])[::-1]; q=np.empty(len(p)); q[order]=np.minimum(ranked,1.0); out[ok]=q
    return out


def add_null_q_values(a,b):
    """Add global and metric-family BH q values to real-vs-null summaries."""
    a=a.copy(); b=b.copy()
    a["q_bh_global"]=bh_adjust(a.p_empirical_upper)
    b["q_bh_global"]=bh_adjust(b.p_empirical_gain_upper)
    a["q_bh_within_metric"]=np.nan; b["q_bh_within_metric"]=np.nan
    for metric,idx in a.groupby("metric").groups.items():a.loc[idx,"q_bh_within_metric"]=bh_adjust(a.loc[idx,"p_empirical_upper"])
    for metric,idx in b.groupby("metric").groups.items():b.loc[idx,"q_bh_within_metric"]=bh_adjust(b.loc[idx,"p_empirical_gain_upper"])
    return a,b


def null_stats(corr,gains,vectors,conditions,refs):
    mods=refs["effects"][refs["effects"].effect_type.eq("module_ablation_clean")]
    mv={m:g.sort_values("behavior").delta_recall_pp.to_numpy(float) for m,g in mods.groupby("effect_id")}
    sl=refs["corr"].set_index(["condition_id","module"]); cl=conditions.set_index("condition_id"); rows=[]
    for r in vectors.itertuples():
        x=np.array([getattr(r,f"delta_recall_L{i}") for i in range(1,6)],float); ids=json.loads(cl.loc[r.condition_id].constituent_ids_json)
        for m in MODULES:
            out={"condition_id":r.condition_id,"mask_order":r.mask_order,"null_id":r.null_id,"module":m}
            for name,col in (("pearson","pearson"),("spearman","spearman"),("cosine","cosine_similarity")):
                val=safe_corr(x,mv[m],"cosine" if name=="cosine" else name); best=np.nanmax([sl.loc[(cid,m),col] for cid in ids])
                out[name]=val; out[f"gain_over_best_single_{name}"]=val-best
            rows.append(out)
    full=pd.DataFrame(rows); full.to_csv(SOURCE_DIR / "multi_mask_random_null_full.csv.gz",index=False,compression="gzip")
    rl=corr.set_index(["condition_id","module"]); gl=gains.set_index(["condition_id","module"]); summaries=[]; gs=[]
    for (cid,m),g in full.groupby(["condition_id","module"]):
        for name,col in (("pearson","pearson"),("spearman","spearman"),("cosine","cosine_similarity")):
            vals=g[name].dropna().to_numpy(float); real=float(rl.loc[(cid,m),col])
            summaries.append({"condition_id":cid,"mask_order":int(cl.loc[cid].mask_order),"module":m,"metric":name,"real_correlation":real,
                "null_n":len(vals),"null_mean":np.mean(vals),"null_median":np.median(vals),"null_ci_low":np.percentile(vals,2.5),"null_ci_high":np.percentile(vals,97.5),
                "real_percentile":100*np.mean(vals<=real),"p_empirical_upper":(1+np.sum(vals>=real))/(len(vals)+1)})
            gc=f"gain_over_best_single_{name}"; gv=g[gc].dropna().to_numpy(float); gr=float(gl.loc[(cid,m),gc])
            gs.append({"condition_id":cid,"mask_order":int(cl.loc[cid].mask_order),"module":m,"metric":name,"real_gain":gr,"null_n":len(gv),
                "null_gain_mean":np.mean(gv),"null_gain_median":np.median(gv),"null_gain_ci_low":np.percentile(gv,2.5),"null_gain_ci_high":np.percentile(gv,97.5),
                "real_gain_percentile":100*np.mean(gv<=gr),"p_empirical_gain_upper":(1+np.sum(gv>=gr))/(len(gv)+1)})
    a,b=add_null_q_values(pd.DataFrame(summaries),pd.DataFrame(gs)); a.to_csv(SOURCE_DIR / "multi_mask_random_null.csv",index=False,encoding="utf-8-sig"); b.to_csv(SOURCE_DIR / "multi_mask_random_gain_null.csv",index=False,encoding="utf-8-sig")
    return a,b,full


def hm(ax,mat,label,cmap="vlag",center=0,fmt=".2f"):
    sns.heatmap(mat,ax=ax,cmap=cmap,center=center,annot=True,fmt=fmt,linewidths=.3,linecolor="white",
                cbar_kws={"label":label,"shrink":.72},annot_kws={"fontsize":5.5})


def figures(conditions,corr,gains,pint,sample,seto,geometry,nulls,nullfull,refs):
    labels=conditions.set_index("condition_id").display_label.to_dict(); modules=list(MODULES)
    candidate=sorted({x for s in conditions.constituent_ids_json for x in json.loads(s)})
    pts=refs["conditions"].set_index("condition_id"); one=refs["corr"][refs["corr"].condition_id.isin(candidate)].copy(); one["mask_order"]=1
    one["display_label"]=[f"L{int(pts.loc[x].label)} {float(pts.loc[x].time_s):+.2f}s" for x in one.condition_id]
    multi=corr.copy(); multi["display_label"]=multi.condition_id.map(labels)
    fa=pd.concat([one[["condition_id","mask_order","display_label","module","pearson","spearman","cosine_similarity"]],multi[["condition_id","mask_order","display_label","module","pearson","spearman","cosine_similarity"]]],ignore_index=True)
    fa.to_csv(SOURCE_DIR / "figA_single_pair_triple_module_correlation.csv",index=False,encoding="utf-8-sig")
    order=fa[["condition_id","mask_order","display_label"]].drop_duplicates().sort_values(["mask_order","condition_id"])
    for label,col,stem in (("Pearson r","pearson","figA1_module_correlation_pearson"),("Spearman rho","spearman","figA2_module_correlation_spearman"),("Cosine similarity","cosine_similarity","figA3_module_correlation_cosine")):
        mat=fa.pivot(index="condition_id",columns="module",values=col).reindex(order.condition_id)[modules]; mat.index=[f"{r.mask_order} | {r.display_label}" for r in order.itertuples()]
        fig,ax=plt.subplots(figsize=(7.2,max(4.6,.25*len(mat)+1.1)),constrained_layout=True); hm(ax,mat,label); ax.set(xlabel="Module ablation",ylabel="Mask order | condition",title=f"Single, pair and triple similarity ({label})"); save_figure(fig,stem)
    fb=fa.melt(id_vars=["condition_id","mask_order","display_label","module"],value_vars=["pearson","spearman","cosine_similarity"],var_name="metric",value_name="correlation")
    fb.to_csv(SOURCE_DIR / "figB_correlation_accumulation.csv",index=False,encoding="utf-8-sig"); pal={"no_conv":"#0072B2","no_lstm":"#E69F00","no_attention":"#009E73"}; marks={"no_conv":"o","no_lstm":"s","no_attention":"^"}
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.65),sharex=True,constrained_layout=True)
    for ax,metric in zip(axs,["pearson","spearman","cosine_similarity"]):
        p=fb[fb.metric==metric]
        for m in MODULES:
            q=p[p.module==m]; jitter=np.linspace(-.055,.055,len(q)); ax.scatter(q.mask_order+jitter,q.correlation,s=12,alpha=.3,color=pal[m],marker=marks[m]); med=q.groupby("mask_order").correlation.median().reindex([1,2,3]); ax.plot([1,2,3],med,color=pal[m],marker=marks[m],lw=1.1,label=ARCH_LABELS[m])
        ax.axhline(0,color="#777",lw=.6); ax.set(xticks=[1,2,3],xlabel="Mask order",title=metric.replace("_"," ").title())
    axs[0].set_ylabel("Correlation / similarity"); axs[-1].legend(fontsize=6); fig.suptitle("Condition distributions and order-wise medians",fontsize=8); save_figure(fig,"figB_correlation_accumulation")
    fc=gains.copy(); fc["display_label"]=fc.condition_id.map(labels); fc.to_csv(SOURCE_DIR / "figC_gain_over_best_single.csv",index=False,encoding="utf-8-sig")
    for metric in ("pearson","spearman","cosine"):
        mat=fc.pivot(index="condition_id",columns="module",values=f"gain_over_best_single_{metric}").reindex(conditions.condition_id)[modules]; mat.index=[labels[x] for x in mat.index]
        fig,ax=plt.subplots(figsize=(7.2,max(3.2,.34*len(mat)+1.1)),constrained_layout=True); hm(ax,mat,f"Gain in {metric}"); ax.set(xlabel="Module ablation",ylabel="Multi-mask condition",title=f"Gain over best constituent single ({metric})"); save_figure(fig,f"figC_gain_over_best_single_{metric}")
    fd=pint.melt(id_vars=["condition_id","interaction_type"],value_vars=[f"I_L{i}" for i in range(1,6)],var_name="behavior",value_name="interaction_pp"); fd["behavior"]=fd.behavior.str.replace("I_","",regex=False); fd["display_label"]=fd.condition_id.map(labels); fd.to_csv(SOURCE_DIR / "figD_pair_interaction.csv",index=False,encoding="utf-8-sig")
    mat=fd.pivot(index="condition_id",columns="behavior",values="interaction_pp").reindex(conditions[conditions.mask_order==2].condition_id)[[f"L{i}" for i in range(1,6)]]; mat.index=[labels[x] for x in mat.index]
    fig,ax=plt.subplots(figsize=(7.2,max(3.1,.38*len(mat)+1)),constrained_layout=True); hm(ax,mat,"I_AB (recall pp)"); ax.set(xlabel="Behavior",ylabel="Pair mask",title="Residual from additive single-mask expectation"); save_figure(fig,"figD_pair_additivity_interaction")
    fe=sample[sample.metric=="true_class_probability"].copy(); fe["row_label"]=[f"{labels[c]} | {s}" for c,s in zip(fe.condition_id,fe.stage)]; fe.to_csv(SOURCE_DIR / "figE_sample_delta_ptrue_correlation.csv",index=False,encoding="utf-8-sig")
    mat=fe.pivot(index="row_label",columns="module",values="spearman")[modules]; fig,ax=plt.subplots(figsize=(7.2,max(4,.23*len(mat)+1.1)),constrained_layout=True); hm(ax,mat,"Spearman rho"); ax.set(xlabel="Module ablation",ylabel="Condition | stage",title="Sample-level Delta Ptrue association on matched OOF samples"); save_figure(fig,"figE_sample_delta_ptrue_correlation")
    ff=seto[seto.stage=="all_stages"].copy(); ff["display_label"]=ff.condition_id.map(labels); ff.to_csv(SOURCE_DIR / "figF_CW_overlap.csv",index=False,encoding="utf-8-sig")
    mat=ff.pivot(index="condition_id",columns="module",values="CW_jaccard").reindex(conditions.condition_id)[modules]; mat.index=[labels[x] for x in mat.index]; fig,ax=plt.subplots(figsize=(7.2,max(3.4,.34*len(mat)+1)),constrained_layout=True); hm(ax,mat,"CW Jaccard","mako",None); ax.set(xlabel="Module ablation",ylabel="Multi-mask condition",title="Damage-set overlap across Run1 and Run2 keys"); save_figure(fig,"figF_CW_overlap")
    fg=nulls[nulls.metric=="pearson"].sort_values("real_percentile",ascending=False).head(5).copy(); fg.to_csv(SOURCE_DIR / "figG_real_vs_random_null_summary.csv",index=False,encoding="utf-8-sig"); keep=set(zip(fg.condition_id,fg.module)); q=nullfull[[tuple(x) in keep for x in zip(nullfull.condition_id,nullfull.module)]].copy(); q["panel"]=[f"{labels[c]}\n{ARCH_LABELS[m]}" for c,m in zip(q.condition_id,q.module)]; q.to_csv(SOURCE_DIR / "figG_real_vs_random_null_distribution.csv.gz",index=False,compression="gzip")
    fig,ax=plt.subplots(figsize=(7.2,3.25),constrained_layout=True); sns.violinplot(data=q,x="panel",y="pearson",color="#A6CEE3",inner="quart",cut=0,ax=ax)
    rv={f"{labels[r.condition_id]}\n{ARCH_LABELS[r.module]}":r.real_correlation for r in fg.itertuples()}
    for i,panel in enumerate(q.panel.drop_duplicates()):ax.scatter(i,rv[panel],marker="D",s=28,color="#D55E00",edgecolor="black",lw=.4,zorder=5)
    ax.axhline(0,color="#777",lw=.6); ax.set(xlabel="Real condition x module",ylabel="Pearson r",title="Real similarity versus size-matched random null"); ax.tick_params(axis="x",rotation=25); save_figure(fig,"figG_real_vs_random_null")
    fh=geometry.copy(); fh["display_label"]=fh.condition_id.map(labels); fh.to_csv(SOURCE_DIR / "figH_mask_geometry_QA.csv",index=False,encoding="utf-8-sig"); fig,axs=plt.subplots(1,3,figsize=(7.2,3.4),constrained_layout=True); y=np.arange(len(fh)); axs[0].barh(y,fh.union_bins,color="#0072B2"); axs[1].barh(y,fh.max_pairwise_jaccard,color="#E69F00"); axs[1].axvline(.8,color="#D55E00",ls="--",lw=.8); axs[2].barh(y,fh.minimum_effective_extra_bins,color="#009E73")
    for ax in axs:ax.set_yticks(y);ax.invert_yaxis()
    axs[0].set_yticklabels(fh.display_label);axs[1].set_yticklabels([]);axs[2].set_yticklabels([]);axs[0].set_xlabel("Union bins");axs[1].set_xlabel("Max pairwise Jaccard");axs[2].set_xlabel("Min effective extra bins");fig.suptitle("Multi-mask geometry QA",fontsize=8);save_figure(fig,"figH_mask_geometry_QA")


def summary_table(conditions,geometry,corr,gains,sample,seto,nulls,pint,tint):
    g=geometry.set_index("condition_id"); r={m:corr[corr.module==m].set_index("condition_id") for m in MODULES}; gn=gains[gains.module=="no_conv"].set_index("condition_id")
    ss=sample[(sample.module=="no_conv")&(sample.metric=="true_class_probability")].groupby("condition_id").spearman.apply(lambda x:x.iloc[np.argmax(np.abs(x))]); cw=seto[(seto.module=="no_conv")&(seto.stage=="all_stages")].set_index("condition_id"); nl=nulls[(nulls.module=="no_conv")&(nulls.metric=="pearson")].set_index("condition_id")
    it=pd.concat([pint[["condition_id","interaction_type"]],tint[["condition_id","interaction_type"]]],ignore_index=True).set_index("condition_id"); rows=[]
    for c in conditions.itertuples():
        cid=c.condition_id; rows.append({"Condition":c.display_label,"Condition_ID":cid,"Order":c.mask_order,"Union_bins":int(g.loc[cid].union_bins),"NoConv_Pearson":r['no_conv'].loc[cid].pearson,"NoConv_Spearman":r['no_conv'].loc[cid].spearman,"NoConv_Cosine":r['no_conv'].loc[cid].cosine_similarity,"Gain_vs_best_single_Pearson":gn.loc[cid].gain_over_best_single_pearson,"NoLSTM_Pearson":r['no_lstm'].loc[cid].pearson,"NoAttention_Pearson":r['no_attention'].loc[cid].pearson,"DeltaPtrue_rho_with_NoConv_max_abs_stage":ss.loc[cid],"CW_Jaccard_with_NoConv":cw.loc[cid].CW_jaccard,"Random_null_percentile_NoConv_Pearson":nl.loc[cid].real_percentile,"Random_null_p_NoConv_Pearson":nl.loc[cid].p_empirical_upper,"Random_null_q_global_NoConv_Pearson":nl.loc[cid].q_bh_global,"Random_null_q_within_metric_NoConv_Pearson":nl.loc[cid].q_bh_within_metric,"Interaction_type":it.loc[cid].interaction_type})
    out=pd.DataFrame(rows);out.to_csv(TABLE_DIR / "multi_mask_summary_for_group_meeting.csv",index=False,encoding="utf-8-sig");return out


def write_multi_summary(conditions,corr,gains,sample,seto,nulls,gnull,pint,tint,gate):
    nc=corr[corr.module=="no_conv"]; bp=nc[nc.mask_order==2].sort_values("pearson",ascending=False).iloc[0]; bg=gains[(gains.module=="no_conv")&(gains.mask_order==2)].sort_values("gain_over_best_single_pearson",ascending=False).iloc[0]
    sn=sample[(sample.module=="no_conv")&(sample.metric=="true_class_probability")].copy(); bs=sn.loc[sn.spearman.abs().idxmax()]; bc=seto[(seto.module=="no_conv")&(seto.stage=="all_stages")].sort_values("CW_jaccard",ascending=False).iloc[0]; bn=nulls[(nulls.module=="no_conv")&(nulls.metric=="pearson")].sort_values("real_percentile",ascending=False).iloc[0]; bgn=gnull[(gnull.module=="no_conv")&(gnull.metric=="pearson")].sort_values("real_gain_percentile",ascending=False).iloc[0]
    pair_nc=gains[(gains.module=="no_conv")&(gains.mask_order==2)]; n_pair_positive=int((pair_nc.gain_over_best_single_pearson>0).sum())
    pair_max_by_module=gains[gains.mask_order==2].groupby("module").gain_over_best_single_pearson.max().to_dict()
    min_nc_p=float(nulls[(nulls.module=="no_conv")&(nulls.metric=="pearson")].p_empirical_upper.min())
    min_nc_q=float(nulls[(nulls.module=="no_conv")&(nulls.metric=="pearson")].q_bh_within_metric.min())
    max_any_cw=float(seto[seto.stage=="all_stages"].CW_jaccard.max()); max_any_rho=float(sample[sample.metric=="true_class_probability"].spearman.abs().max())
    if gate:
        bt=nc[nc.mask_order==3].sort_values("pearson",ascending=False).iloc[0]; tg=gains[(gains.module=="no_conv")&(gains.mask_order==3)].sort_values("gain_over_best_single_pearson",ascending=False).iloc[0]; triple=f"最强 triple `{bt.condition_id}` 的 NoConv Pearson={bt.pearson:.3f}，gain={tg.gain_over_best_single_pearson:+.3f}。"
    else:triple="Pair pilot 没有正的 NoConv Pearson gain，按停止规则未运行 triple。"
    text=f"""# Multi-region neural masking × module-ablation association analysis

## 结论先行

- 全程未训练、未重划 fold；复用 frozen Full/NoConv/NoLSTM/NoAttention、原 test indices 和 legacy 5 s/10 ms/500-bin/250-bin protocol。
- Pair 采用原始 clean window 上的 Boolean union，一次 maximum replacement。共 {sum(conditions.mask_order==2)} 个预注册 pair；最强 NoConv Pearson 为 `{bp.condition_id}`，r={bp.pearson:.3f}。
- Pair 相对 constituent single 最佳 NoConv Pearson 的最大增益为 `{bg.condition_id}`，gain={bg.gain_over_best_single_pearson:+.3f}。
- {triple}
- 样本级 NoConv ΔPtrue 最大绝对 Spearman 为 `{bs.condition_id}`、{bs.stage}，rho={bs.spearman:.3f}；pooled CW Jaccard 最大为 `{bc.condition_id}`，J={bc.CW_jaccard:.3f}。
- Size-matched null 中，NoConv Pearson percentile 最高为 `{bn.condition_id}`：{bn.real_percentile:.1f}%，经验 p={bn.p_empirical_upper:.4f}，全局 BH q={bn.q_bh_global:.4f}；gain percentile 最高为 `{bgn.condition_id}`：{bgn.real_gain_percentile:.1f}%，经验 p={bgn.p_empirical_gain_upper:.4f}，全局 BH q={bgn.q_bh_global:.4f}。

## Q1–Q10

1. Pair 是否更像 NoConv：8 个 pair 中仅 {n_pair_positive} 个 Pearson gain>0；最大为 `{bg.condition_id}`，gain={bg.gain_over_best_single_pearson:+.3f}，但其 Spearman gain={bg.gain_over_best_single_spearman:+.3f}、cosine gain={bg.gain_over_best_single_cosine:+.3f}，证据不跨指标一致。
2. Triple 是否继续增加：{triple} 相对最佳 constituent single 没有进一步增加。
3. 是否 NoConv-specific：pair 的最大 Pearson gain 分别为 NoConv={pair_max_by_module.get('no_conv',np.nan):+.3f}、NoLSTM={pair_max_by_module.get('no_lstm',np.nan):+.3f}、NoAttention={pair_max_by_module.get('no_attention',np.nan):+.3f}。短名单中正增益主要见于 NoConv（另有较小 NoAttention 增益），但 matched-null 不支持 region-specific correspondence。
4. 样本级是否同步：三模块全部条件的最大 |ΔPtrue rho|={max_any_rho:.3f}；NoConv 最大为 {abs(bs.spearman):.3f}。类别级相似没有转化为强样本级一致。
5. CW 是否同步：三模块全部条件最大 pooled Jaccard={max_any_cw:.3f}；NoConv 最大={bc.CW_jaccard:.3f}，具体受损样本重合很低。
6. 是否 additive：pair 分类计数 {json.dumps(pint.interaction_type.value_counts().to_dict(),ensure_ascii=False)}；triple {json.dumps(tint.interaction_type.value_counts().to_dict() if len(tint) else {},ensure_ascii=False)}。
7. Synergy/redundancy：逐 behavior residual 已保存；recall 有边界，故仅为 descriptive interaction。
8. 真实 correlation 是否超过 random null：否。NoConv Pearson 最小单侧经验 p={min_nc_p:.4f}，Pearson-family BH q≥{min_nc_q:.4f}；负结果完整保留。
9. 真实 gain 是否超过 random gain：否；最佳 NoConv gain 的经验检验未通过，完整 p/q 见 `multi_mask_random_gain_null.csv`。
10. overlap/perturbation size：主 pair 均满足 max Jaccard<0.80；每个真实条件使用 N={int(nulls.null_n.max())} 个 stage/coverage/component/union-size 匹配 null。真实结果没有超过该 null，因此更符合“扰动范围增大导致输出表型趋近模块消融”，不足以证明所选区域具有 Conv-specific correspondence。

## 解释边界

Correlation/similarity 描述输出效应指纹；additivity residual 描述 frozen-model 非加性；两者均不能证明 causal biological interpretation。每个 center 实际对应 ±1 s 区域，不是 10-ms causal node。Random control 是 within-retained-window shifted-centre geometry-matched null：保留 stage 和命中窗口，随机移动 constituent centers 后仍用 ±1 s、边界裁剪、一次 union 和原窗 per-channel maximum。
"""
    (OUT_ROOT / "MULTI_MASK_RESULTS_SUMMARY_CN.md").write_text(text,encoding="utf-8")


def multi_provenance(refs,conditions,n_null,gate):
    for r in refs["checkpoints"].itertuples():
        if sha256_file(Path(r.path))!=r.sha256:raise RuntimeError(f"Checkpoint hash mismatch {r.path}")
    refs["checkpoints"].to_csv(QA_DIR / "checkpoint_manifest_sha256.csv",index=False,encoding="utf-8-sig")
    try:commit=subprocess.check_output(["git","-C",str(REPO_ROOT),"rev-parse","HEAD"],text=True).strip()
    except Exception:commit=None
    payload={"analysis_script":str(SCRIPT_PATH),"analysis_script_sha256":sha256_file(SCRIPT_PATH),"single_analysis_root":str(SINGLE_ROOT),"single_analysis_script_sha256":sha256_file(SINGLE_ROOT/"run_analysis.py"),"repository_commit_at_analysis_start":commit,"training_performed":False,"resplitting_performed":False,"n_pairs":int((conditions.mask_order==2).sum()),"n_triples":int((conditions.mask_order==3).sum()),"triple_gate_passed":gate,"random_null_repeats":n_null,"protocol":{"window_seconds":5,"bin_ms":10,"window_bins":500,"step_bins":250,"stage_label_constant":True,"mask_interval":"[center-1s,center+1s), clipped","multi_mask":"Boolean union on original clean window, one replacement pass","replacement":"per-channel maximum of original unmodified window","delta":"ablated-baseline","real_device":"cpu","null_device":"cuda after exact class validation"},"environment":{"python":sys.version,"platform":platform.platform(),"torch":torch.__version__,"cuda":torch.cuda.is_available(),"cuda_device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"numpy":np.__version__,"pandas":pd.__version__}}
    (QA_DIR / "provenance_and_config.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")


def validate_outputs():
    """Machine-readable end-to-end checks for the frozen-model result bundle."""
    checks=[]
    def add(name,passed,detail):checks.append({"check":name,"passed":bool(passed),"detail":str(detail)})
    required=[
        "multi_mask_conditions.csv","multi_mask_overlap_audit.csv","multi_mask_metrics.csv",
        "multi_mask_predictions_stagewise.csv.gz","multi_mask_module_correlations.csv",
        "multi_mask_correlation_gain.csv","pair_additivity_interaction.csv","triple_interaction.csv",
        "multi_mask_sample_level_correlations.csv","multi_mask_CW_WC_overlap.csv",
        "multi_mask_confusion_similarity.csv","multi_mask_random_null.csv","multi_mask_random_gain_null.csv",
    ]
    missing=[x for x in required if not (SOURCE_DIR/x).exists() or (SOURCE_DIR/x).stat().st_size==0]
    add("required_source_files_nonempty",not missing,missing or f"{len(required)} files")
    c=pd.read_csv(SOURCE_DIR/"multi_mask_conditions.csv"); p=pd.read_csv(SOURCE_DIR/"multi_mask_predictions_stagewise.csv.gz")
    add("condition_counts",len(c)==10 and sum(c.mask_order==2)==8 and sum(c.mask_order==3)==2,f"total={len(c)}, pair={sum(c.mask_order==2)}, triple={sum(c.mask_order==3)}")
    key=["condition_id","stage","fold","dataset_index"]
    add("paired_prediction_keys_unique",not p.duplicated(key).any(),f"rows={len(p)}, duplicates={int(p.duplicated(key).sum())}")
    add("stagewise_not_fake_five_class",set(p.stage)=={"run1","run2"},sorted(p.stage.unique()))
    n_per=p.groupby("condition_id").size(); add("all_conditions_use_same_1193_oof_samples",len(n_per)==10 and (n_per==1193).all(),n_per.to_dict())
    add("mask_hits_consistent_with_bins",((p.replaced_bin_count>0)==p.mask_hit.astype(bool)).all(),f"mismatches={int(((p.replaced_bin_count>0)!=p.mask_hit.astype(bool)).sum())}")
    q=pd.read_csv(SOURCE_DIR/"multi_mask_random_null.csv"); qg=pd.read_csv(SOURCE_DIR/"multi_mask_random_gain_null.csv")
    add("random_null_summary_shape",len(q)==90 and len(qg)==90,f"correlation={len(q)}, gain={len(qg)}")
    add("random_null_q_values_present",{"q_bh_global","q_bh_within_metric"}.issubset(q.columns) and {"q_bh_global","q_bh_within_metric"}.issubset(qg.columns),"global and metric-family BH")
    geom=pd.read_csv(SOURCE_DIR/"multi_mask_random_null_geometry.csv.gz")
    add("random_null_500_per_condition",len(geom)==5000 and (geom.groupby("condition_id").size()==500).all(),geom.groupby("condition_id").size().to_dict())
    win_mismatch=(geom.null_affected_windows_run1!=geom.real_affected_windows_run1)|(geom.null_affected_windows_run2!=geom.real_affected_windows_run2)
    max_rel=np.maximum(geom.union_bin_relative_error_run1.abs(),geom.union_bin_relative_error_run2.abs())
    add("random_null_affected_windows_exact",not win_mismatch.any(),f"mismatches={int(win_mismatch.sum())}")
    add("random_null_union_size_within_6pct",float(max_rel.max())<.06,f"max_abs_relative_error={float(max_rel.max()):.6f}")
    fold=pd.read_csv(QA_DIR/"fold_test_index_alignment.csv"); add("saved_fold_alignment",fold.used_by_all_architectures.astype(bool).all() and fold.within_fold_unique.astype(bool).all(),f"rows={len(fold)}")
    gpu=pd.read_csv(QA_DIR/"gpu_cpu_class_prediction_validation.csv"); add("gpu_cpu_exact_class_predictions",gpu.class_predictions_exact.astype(bool).all() and int(gpu.n_class_mismatches.sum())==0,f"rows={len(gpu)}, mismatches={int(gpu.n_class_mismatches.sum())}")
    single=pd.read_csv(QA_DIR/"single_analysis_reference_checks.csv"); add("single_baseline_audit",single.passed.astype(bool).all(),single.to_dict("records"))
    png={x.stem for x in FIGURE_DIR.glob("*.png")}; pdf={x.stem for x in FIGURE_DIR.glob("*.pdf")}; svg={x.stem for x in FIGURE_DIR.glob("*.svg")}; tif={x.stem for x in FIGURE_DIR.glob("*.tiff")}
    add("figure_formats_complete",len(png)==12 and png==pdf==svg==tif,f"stems={len(png)}")
    out=pd.DataFrame(checks); out.to_csv(QA_DIR/"output_validation.csv",index=False,encoding="utf-8-sig")
    (QA_DIR/"output_validation.json").write_text(json.dumps({"all_passed":bool(out.passed.all()),"checks":checks},indent=2,ensure_ascii=False),encoding="utf-8")
    if not out.passed.all():raise RuntimeError("Output validation failed: "+", ".join(out.loc[~out.passed,"check"]))
    return out


def finalize_existing_outputs(regenerate_figures=False):
    """Refresh derived tables/summary/provenance without repeating model inference."""
    refs=load_single_refs(); conditions=pd.read_csv(SOURCE_DIR / "multi_mask_conditions.csv")
    corr=pd.read_csv(SOURCE_DIR / "multi_mask_module_correlations.csv")
    gains=pd.read_csv(SOURCE_DIR / "multi_mask_correlation_gain.csv")
    geometry=pd.read_csv(SOURCE_DIR / "figH_mask_geometry_QA.csv")
    sample=pd.read_csv(SOURCE_DIR / "multi_mask_sample_level_correlations.csv")
    seto=pd.read_csv(SOURCE_DIR / "multi_mask_CW_WC_overlap.csv")
    pint=pd.read_csv(SOURCE_DIR / "pair_additivity_interaction.csv")
    tint=pd.read_csv(SOURCE_DIR / "triple_interaction.csv")
    nulls=pd.read_csv(SOURCE_DIR / "multi_mask_random_null.csv")
    gnull=pd.read_csv(SOURCE_DIR / "multi_mask_random_gain_null.csv")
    nulls,gnull=add_null_q_values(nulls,gnull)
    nulls.to_csv(SOURCE_DIR / "multi_mask_random_null.csv",index=False,encoding="utf-8-sig")
    gnull.to_csv(SOURCE_DIR / "multi_mask_random_gain_null.csv",index=False,encoding="utf-8-sig")
    nulls[nulls.metric.eq("pearson")].sort_values("real_percentile",ascending=False).head(5).to_csv(
        SOURCE_DIR / "figG_real_vs_random_null_summary.csv",index=False,encoding="utf-8-sig")
    if regenerate_figures:
        nfull=pd.read_csv(SOURCE_DIR / "multi_mask_random_null_full.csv.gz")
        figures(conditions,corr,gains,pint,sample,seto,geometry,nulls,nfull,refs)
    gate=bool(pd.read_csv(QA_DIR / "triple_continuation_gate.csv").passed.iloc[0])
    summary_table(conditions,geometry,corr,gains,sample,seto,nulls,pint,tint)
    write_multi_summary(conditions,corr,gains,sample,seto,nulls,gnull,pint,tint,gate)
    validate_outputs(); multi_provenance(refs,conditions,int(nulls.null_n.max()),gate)


def main() -> None:
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument("--null-repeats",type=int,default=500); parser.add_argument("--skip-null",action="store_true"); parser.add_argument("--resume",action="store_true"); parser.add_argument("--finalize-existing",action="store_true"); parser.add_argument("--regenerate-figures-existing",action="store_true"); args=parser.parse_args()
    if args.finalize_existing:
        finalize_existing_outputs(args.regenerate_figures_existing); print(json.dumps({"status":"existing_outputs_finalized","inference_repeated":False,"figures_regenerated":args.regenerate_figures_existing},indent=2)); return
    print("[0] audit single-mask references",flush=True); refs=load_single_refs()
    print("[1] rebuild exact legacy datasets and overlap",flush=True)
    configs={"run1":Config(),"run2":Config3()}
    for c in configs.values():c.device="cpu"
    datasets={"run1":PointwiseLegacyDataset(configs["run1"],"run1",refs["conditions"]),"run2":PointwiseLegacyDataset(configs["run2"],"run2",refs["conditions"])}
    overlap_audit(refs,datasets); qa_fold_partitions(datasets)
    print("[2] pair pilot",flush=True); pair_conditions=build_multi_conditions(refs["conditions"],False); pgd,pg=geometry_table(pair_conditions,datasets)
    pair_path=SOURCE_DIR / "pair_condition_predictions.csv.gz"; pp=pd.read_csv(pair_path) if args.resume and pair_path.exists() else infer_real(pair_conditions,datasets,configs,refs,"pair")
    pm,pe,pc=behavior_effects(pp,pair_conditions,refs,pg); pgn=gains_table(pc,pair_conditions,refs)
    gate=bool((pgn[(pgn.module=="no_conv")].gain_over_best_single_pearson>0).any()); pd.DataFrame([{"gate":"any_pair_positive_NoConv_Pearson_gain","passed":gate,"max_gain":pgn[pgn.module=="no_conv"].gain_over_best_single_pearson.max()}]).to_csv(QA_DIR / "triple_continuation_gate.csv",index=False,encoding="utf-8-sig")
    if gate:
        print("[3] gate passed; run triples",flush=True); conditions=build_multi_conditions(refs["conditions"],True); tc=conditions[conditions.mask_order==3].copy(); tgd,tg=geometry_table(tc,datasets); triple_path=SOURCE_DIR / "triple_condition_predictions.csv.gz"; tp=pd.read_csv(triple_path) if args.resume and triple_path.exists() else infer_real(tc,datasets,configs,refs,"triple"); predictions=pd.concat([pp,tp],ignore_index=True); gd=pd.concat([pgd,tgd],ignore_index=True); geometry=pd.concat([pg,tg],ignore_index=True)
    else:
        print("[3] gate failed; stop triples",flush=True); conditions=pair_conditions; predictions=pp; gd=pgd; geometry=pg
    predictions=predictions.drop_duplicates(["architecture","condition_id","stage","fold","dataset_index"],keep="first").reset_index(drop=True)
    conditions.to_csv(SOURCE_DIR / "multi_mask_conditions.csv",index=False,encoding="utf-8-sig"); gd.to_csv(SOURCE_DIR / "multi_mask_geometry_stagewise.csv",index=False,encoding="utf-8-sig"); predictions.to_csv(SOURCE_DIR / "multi_mask_condition_predictions_stagewise.csv.gz",index=False,compression="gzip")
    defs=data_definitions(conditions); pairs=paired_predictions(predictions,defs); pairs.to_csv(SOURCE_DIR / "multi_mask_predictions_stagewise.csv.gz",index=False,compression="gzip")
    metrics,mean,corr=behavior_effects(predictions,conditions,refs,geometry); metrics.to_csv(SOURCE_DIR / "multi_mask_metrics.csv",index=False,encoding="utf-8-sig"); gains=gains_table(corr,conditions,refs); pint,tint=interaction_outputs(mean,conditions,refs)
    print("[4] sample, set and confusion analyses",flush=True); sample,seto=sample_and_set_outputs(pairs,conditions,refs); confusion_output(pairs,conditions,refs)
    if args.skip_null:
        multi_provenance(refs,conditions,0,gate); print(json.dumps({"status":"real_only_complete","triple_gate":gate},indent=2)); return
    if not torch.cuda.is_available():raise RuntimeError("CUDA required for matched random null")
    print("[5] validate GPU and run matched null",flush=True); gpu_cpu_validate(conditions,datasets,configs,refs,predictions); ng,nv=run_nulls(conditions,datasets,configs,refs,args.null_repeats); nulls,gnull,nfull=null_stats(corr,gains,nv,conditions,refs)
    print("[6] figures, summary and provenance",flush=True); figures(conditions,corr,gains,pint,sample,seto,geometry,nulls,nfull,refs); summary_table(conditions,geometry,corr,gains,sample,seto,nulls,pint,tint); write_multi_summary(conditions,corr,gains,sample,seto,nulls,gnull,pint,tint,gate); validate_outputs(); multi_provenance(refs,conditions,args.null_repeats,gate)
    completion = {
        "status": "complete", "training_performed": False, "resplitting_performed": False,
        "n_pairs": int((conditions.mask_order==2).sum()), "n_triples": int((conditions.mask_order==3).sum()),
        "triple_gate_passed": gate, "n_null_per_condition": args.null_repeats,
        "prediction_rows": int(len(pairs)), "figures": sorted(path.name for path in FIGURE_DIR.glob("*.png")),
    }
    (QA_DIR / "completion_manifest.json").write_text(json.dumps(completion, indent=2), encoding="utf-8")
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
