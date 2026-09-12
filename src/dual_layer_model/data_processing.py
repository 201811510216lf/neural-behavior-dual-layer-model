import json
import re
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _natural_key(path: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def list_aligned_tables(data_dir: Path) -> List[Path]:
    files = sorted(Path(data_dir).glob("*_aligned.csv.gz"), key=_natural_key)
    if not files:
        raise FileNotFoundError(f"No *_aligned.csv.gz files found in {data_dir}")
    return files


def discover_neural_columns(files: Sequence[Path]) -> List[str]:
    cols = set()
    for path in files:
        header = pd.read_csv(path, nrows=0).columns
        cols.update(c for c in header if re.fullmatch(r"N\d+", str(c)))
    if not cols:
        raise ValueError("No neural columns like N1, N2, ... were found.")
    return sorted(cols, key=lambda c: int(c[1:]))


class TimeSeriesDataset(Dataset):

    def __init__(self, data, labels, window_size, step_size):
        self.window_size = int(window_size)
        self.step_size = int(step_size)
        self.windowed_data, self.labels = self.segment(data, labels)

    def segment(self, data, labels=None):
        n_samples = data.shape[0]
        windowed_data = []
        output_labels = []
        for start in range(0, n_samples - self.window_size + 1, self.step_size):
            segment = data[start:start + self.window_size, :]
            if labels is not None:
                seg_labels = labels[start:start + self.window_size]
                if np.all(seg_labels == seg_labels[0]):
                    windowed_data.append(segment.T)
                    output_labels.append(seg_labels[0])
            else:
                windowed_data.append(segment.T)
        return np.array(windowed_data, dtype=np.float32), np.array(output_labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.windowed_data[idx].astype(np.float32)
        y = int(self.labels[idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def _load_current_aligned_data(config):
    files = list_aligned_tables(config.data_dir)
    neural_columns = discover_neural_columns(files)
    all_features = []
    all_labels = []
    source_rows = []

    for path in files:
        df = pd.read_csv(path)
        for col in neural_columns:
            if col not in df.columns:
                df[col] = 0.0
        df = df.sort_values("time_bin").reset_index(drop=True)

        raw_labels = df["label"].astype(int).to_numpy()
        keep_mask = np.isin(raw_labels, list(config.valid_raw_labels))
        if not np.any(keep_mask):
            continue

        # User rule: remove label 0, then subtract 1.
        final_labels = np.vectorize(config.raw_label_to_final_class.get)(raw_labels[keep_mask])
        features = df.loc[keep_mask, neural_columns].to_numpy(dtype=np.float32)

        all_features.append(features)
        all_labels.append(final_labels.astype(np.int64))
        source_rows.append(
            {
                "file": str(path),
                "recording": str(df["recording"].iloc[0]) if "recording" in df.columns else path.stem,
                "rows_after_remove_0": int(keep_mask.sum()),
            }
        )

    if not all_features:
        raise ValueError("No valid rows after removing raw label 0.")

    features = np.vstack(all_features)
    labels = np.concatenate(all_labels)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    with open(config.output_dir / "source_rows_after_remove0.json", "w", encoding="utf-8") as f:
        json.dump(source_rows, f, ensure_ascii=False, indent=2)

    return features, labels, neural_columns


def prepare_data(config):
    """
    Stage 1 original logic:
    final classes 1/3/4 -> merged class 1;
    final classes 0 and 2 are kept as 0 and 2.
    """
    features, labels, neural_columns = _load_current_aligned_data(config)
    labels = np.where(np.isin(labels, list(config.merged_stage_classes)), 1, labels)
    dataset = TimeSeriesDataset(features, labels, config.window_size, config.step_size)
    _save_dataset_summary(config, "run1_dataset_summary.json", dataset, neural_columns)
    return dataset


def prepare_data3(config):
    """
    Stage 2 original logic:
    keep only final classes 1/3/4 and remap them to 0/1/2.
    """
    features, labels, neural_columns = _load_current_aligned_data(config)
    mask = np.isin(labels, list(config.merged_stage_classes))
    features = features[mask]
    labels = labels[mask]
    labels = np.vectorize(config.run2_label_map.get)(labels).astype(np.int64)
    dataset = TimeSeriesDataset(features, labels, config.window_size, config.step_size)
    _save_dataset_summary(config, "run2_dataset_summary.json", dataset, neural_columns)
    return dataset


def _save_dataset_summary(config, filename: str, dataset: TimeSeriesDataset, neural_columns: Sequence[str]) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    unique, counts = np.unique(dataset.labels, return_counts=True)
    summary = {
        "n_windows": int(len(dataset)),
        "input_size": int(dataset.windowed_data.shape[1]) if len(dataset) else 0,
        "window_size": int(config.window_size),
        "step_size": int(config.step_size),
        "neural_columns": list(neural_columns),
        "label_counts": {int(k): int(v) for k, v in zip(unique, counts)},
        "note": "Legacy mode: remove raw label 0, convert 1-5 to 0-4, concatenate recordings, then window.",
    }
    with open(config.output_dir / filename, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
