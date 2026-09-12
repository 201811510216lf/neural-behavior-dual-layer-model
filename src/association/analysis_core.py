import json
import importlib.util
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import config


sys.path.insert(0, str(config.LEGACY_PROJECT))
from model import ConvFeatureAttentionLSTM  # noqa: E402


def natural_key(path: Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def safe_corr(x: np.ndarray, y: np.ndarray, method: str = "spearman") -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3 or np.std(x[finite]) < 1e-12 or np.std(y[finite]) < 1e-12:
        return float("nan")
    if method == "pearson":
        return float(pearsonr(x[finite], y[finite])[0])
    return float(spearmanr(x[finite], y[finite])[0])


def fisher_mean(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return float("nan")
    return float(np.tanh(np.mean(np.arctanh(np.clip(x, -0.999999, 0.999999)))))


def bootstrap_ci(values: Iterable[float], seed: int = config.SEED, n_boot: int = 5000):
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def z_euclidean_rdm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    active = x.std(axis=0) > 1e-12
    if not np.any(active):
        return np.full((len(x), len(x)), np.nan)
    z = (x[:, active] - x[:, active].mean(axis=0)) / x[:, active].std(axis=0)
    return squareform(pdist(z, metric="euclidean")) / math.sqrt(int(active.sum()))


def correlation_rdm(x: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return squareform(pdist(np.asarray(x, dtype=float), metric="correlation"))


def rdm_vector(rdm: np.ndarray) -> np.ndarray:
    return rdm[np.tril_indices(rdm.shape[0], -1)]


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    k = x @ x.T
    l = y @ y.T
    denom = np.linalg.norm(k, "fro") * np.linalg.norm(l, "fro")
    return float(np.sum(k * l) / denom) if denom > 1e-12 else float("nan")


def partial_rank_corr(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(control, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if finite.sum() < 6:
        return float("nan")
    xr, yr, zr = rankdata(x[finite]), rankdata(y[finite]), rankdata(z[finite])
    design = np.column_stack([np.ones(len(zr)), zr])
    x_res = xr - design @ np.linalg.lstsq(design, xr, rcond=None)[0]
    y_res = yr - design @ np.linalg.lstsq(design, yr, rcond=None)[0]
    return safe_corr(x_res, y_res, method="pearson")


def bin_time(x: np.ndarray, axis: int) -> np.ndarray:
    x = np.asarray(x)
    x = np.moveaxis(x, axis, 1)
    n_bins = x.shape[1] // config.BIN_SIZE
    x = x[:, : n_bins * config.BIN_SIZE]
    new_shape = (x.shape[0], n_bins, config.BIN_SIZE) + x.shape[2:]
    return x.reshape(new_shape).mean(axis=2)


def _load_rows() -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    files = sorted(config.ALIGNED_DIR.glob("*_aligned.csv.gz"), key=natural_key)
    if not files:
        raise FileNotFoundError(config.ALIGNED_DIR)
    all_neural = set()
    for path in files:
        all_neural.update(
            c for c in pd.read_csv(path, nrows=0).columns
            if re.fullmatch(r"N\d+", str(c))
        )
    neural_cols = sorted(all_neural, key=lambda c: int(c[1:]))
    feature_parts, label_parts, meta_parts = [], [], []
    for path in files:
        df = pd.read_csv(path).sort_values("time_bin").reset_index(drop=True)
        present_neural = [c for c in neural_cols if c in df.columns]
        for col in neural_cols:
            if col not in df.columns:
                df[col] = 0.0
        raw = df["label"].astype(int).to_numpy()
        keep = np.isin(raw, [1, 2, 3, 4, 5])
        if not np.any(keep):
            continue
        recording = str(df["recording"].iloc[0]) if "recording" in df else path.stem.replace("_aligned", "")
        feature_parts.append(df.loc[keep, neural_cols].to_numpy(np.float32))
        label_parts.append(raw[keep].astype(np.int64) - 1)
        meta_parts.append(pd.DataFrame({
            "recording": recording,
            "source_file": str(path),
            "time_bin": df.loc[keep, "time_bin"].to_numpy(np.int64),
            "raw_label": raw[keep],
            "n_real_neurons": len(present_neural),
        }))
    return (
        np.vstack(feature_parts),
        np.concatenate(label_parts),
        pd.concat(meta_parts, ignore_index=True),
        neural_cols,
    )


def reconstruct_windows(run: str) -> Dict[str, object]:
    features, final_labels, row_meta, neural_cols = _load_rows()
    if run == "run1":
        labels = np.where(np.isin(final_labels, [1, 3, 4]), 1, final_labels)
    elif run == "run2":
        keep = np.isin(final_labels, [1, 3, 4])
        features = features[keep]
        final_labels = final_labels[keep]
        row_meta = row_meta.loc[keep].reset_index(drop=True)
        mapping = {1: 0, 3: 1, 4: 2}
        labels = np.vectorize(mapping.get)(final_labels).astype(np.int64)
    else:
        raise ValueError(run)

    windows, output_labels, records = [], [], []
    for start in range(0, len(labels) - config.WINDOW_SIZE + 1, config.STEP_SIZE):
        end = start + config.WINDOW_SIZE
        y = labels[start:end]
        if not np.all(y == y[0]):
            continue
        m = row_meta.iloc[start:end]
        one_recording = m["recording"].nunique() == 1
        contiguous = one_recording and bool(np.all(np.diff(m["time_bin"].to_numpy()) == 1))
        windows.append(features[start:end].T)
        output_labels.append(int(y[0]))
        records.append({
            "dataset_index": len(windows) - 1,
            "global_start": start,
            "global_end": end,
            "recording": str(m["recording"].iloc[0]) if one_recording else "CROSS_RECORDING",
            "start_time_bin": int(m["time_bin"].iloc[0]),
            "end_time_bin": int(m["time_bin"].iloc[-1]),
            "n_real_neurons": int(m["n_real_neurons"].iloc[0]) if one_recording else 0,
            "valid_biological_window": bool(one_recording and contiguous),
            "single_recording": bool(one_recording),
            "time_contiguous": bool(contiguous),
            "run": run,
            "run_label": int(y[0]),
        })
    return {
        "x": np.asarray(windows, dtype=np.float32),
        "y": np.asarray(output_labels, dtype=np.int64),
        "meta": pd.DataFrame(records),
        "neural_cols": neural_cols,
    }


def validate_reconstruction(run: str, payload: Dict[str, object]) -> Dict[str, object]:
    cfg_spec = importlib.util.spec_from_file_location(
        "legacy_model_config", config.LEGACY_PROJECT / "config.py"
    )
    legacy_cfg = importlib.util.module_from_spec(cfg_spec)
    cfg_spec.loader.exec_module(legacy_cfg)
    data_spec = importlib.util.spec_from_file_location(
        "legacy_data_processing", config.LEGACY_PROJECT / "data_processing.py"
    )
    legacy_data = importlib.util.module_from_spec(data_spec)
    data_spec.loader.exec_module(legacy_data)
    original = (
        legacy_data.prepare_data(legacy_cfg.Config())
        if run == "run1"
        else legacy_data.prepare_data3(legacy_cfg.Config3())
    )
    x_ok = np.array_equal(payload["x"], original.windowed_data)
    y_ok = np.array_equal(payload["y"], original.labels)
    if not (x_ok and y_ok):
        raise AssertionError(f"{run} reconstruction differs from legacy dataset")
    meta = payload["meta"]
    return {
        "run": run,
        "n_windows": int(len(payload["y"])),
        "features_exact_match": bool(x_ok),
        "labels_exact_match": bool(y_ok),
        "valid_biological_windows": int(meta["valid_biological_window"].sum()),
        "cross_recording_windows": int((~meta["single_recording"]).sum()),
        "time_discontinuous_windows": int((meta["single_recording"] & ~meta["time_contiguous"]).sum()),
    }


def load_model(run: str, fold: int, device: torch.device):
    model = ConvFeatureAttentionLSTM(
        input_size=12,
        hidden_size=256,
        attention_hidden_size=128,
        num_classes=3,
        conv_channels=64,
        kernel_size=3,
        num_layers=2,
        dropout=0.0,
    )
    path = config.LEGACY_OUTPUT / f"best_test_model_{run}_fold{fold:02d}.pth"
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def load_test_indices(run: str, fold: int) -> List[int]:
    path = config.LEGACY_OUTPUT / f"fold_split_{run}_fold{fold:02d}.json"
    return json.loads(path.read_text(encoding="utf-8"))["test_indices"]


@torch.no_grad()
def forward_states(
    model: ConvFeatureAttentionLSTM,
    x: np.ndarray,
    device: torch.device,
    intervention: str = "Full",
    batch_size: int = 64,
) -> Dict[str, np.ndarray]:
    outputs = defaultdict(list)
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start:start + batch_size]).to(device)
        conv = model.conv(xb)
        if intervention == "Zero Conv signal":
            conv = torch.zeros_like(conv)
        seq = conv.permute(0, 2, 1)
        lstm, _ = model.lstm(seq)
        if intervention == "Zero LSTM signal":
            lstm = torch.zeros_like(lstm)
        if intervention == "Uniform Attention":
            attn = torch.full(
                (len(xb), lstm.shape[1]), 1.0 / lstm.shape[1],
                device=device, dtype=lstm.dtype,
            )
        else:
            attn = torch.softmax(model.time_attention(lstm).squeeze(-1), dim=1)
        context = torch.bmm(attn.unsqueeze(1), lstm).squeeze(1)
        logits = model.fc(context)
        contribution = lstm * attn.unsqueeze(-1)
        for name, value in (
            ("logits", logits), ("conv", conv), ("lstm", lstm),
            ("attention", attn), ("contribution", contribution), ("context", context),
        ):
            outputs[name].append(value.detach().cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in outputs.items()}


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": 100 * float(accuracy_score(y_true, y_pred)),
        "precision": 100 * float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": 100 * float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": 100 * float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> List[float]:
    values = []
    for cls in range(n_classes):
        mask = y_true == cls
        values.append(100 * float(np.mean(y_pred[mask] == cls)) if np.any(mask) else float("nan"))
    return values


def merge_legacy_metrics(a: Dict, b: Dict) -> Dict:
    ac, bc = a["class_accuracy"], b["class_accuracy"]
    return {
        "accuracy": (a["accuracy"] + b["accuracy"]) / 2,
        "precision": (a["precision"] + b["precision"]) / 2,
        "recall": (a["recall"] + b["recall"]) / 2,
        "f1": (a["f1"] + b["f1"]) / 2,
        "L1": ac[0],
        "L2": (ac[1] + bc[0]) / 2,
        "L3": ac[2],
        "L4": (ac[1] + bc[1]) / 2,
        "L5": (ac[1] + bc[2]) / 2,
    }


def layer_features(states: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    conv = bin_time(states["conv"].transpose(0, 2, 1), axis=1)
    lstm = bin_time(states["lstm"], axis=1)
    contrib = bin_time(states["contribution"], axis=1)
    return {
        "Conv": conv.reshape(len(conv), -1),
        "LSTM": lstm.reshape(len(lstm), -1),
        "Attention contribution": contrib.reshape(len(contrib), -1),
        "Context": states["context"],
    }


def neural_features(x: np.ndarray, n_real: int) -> np.ndarray:
    values = x[:, :n_real, :].transpose(0, 2, 1)
    values = bin_time(values, axis=1)
    return values.reshape(len(values), -1)


def experiment1_baseline(datasets, device):
    rows, fold_cache = [], {}
    for run, payload in datasets.items():
        for fold in range(1, 4):
            idx = np.asarray(load_test_indices(run, fold), dtype=int)
            model = load_model(run, fold, device)
            states = forward_states(model, payload["x"][idx], device)
            pred = states["logits"].argmax(axis=1)
            metrics = metric_row(payload["y"][idx], pred)
            metrics.update({
                "run": run, "fold": f"fold{fold:02d}",
                "n_test": len(idx),
                "class_accuracy": per_class_accuracy(payload["y"][idx], pred),
            })
            rows.append(metrics)
            fold_cache[(run, fold)] = {
                "indices": idx, "states": states, "pred": pred, "model": model,
            }
    run_df = pd.DataFrame([{k: v for k, v in row.items() if k != "class_accuracy"} for row in rows])
    merged = []
    for fold in range(1, 4):
        a = next(r for r in rows if r["run"] == "run1" and r["fold"] == f"fold{fold:02d}")
        b = next(r for r in rows if r["run"] == "run2" and r["fold"] == f"fold{fold:02d}")
        row = merge_legacy_metrics(a, b)
        row["fold"] = f"fold{fold:02d}"
        merged.append(row)
    merged_df = pd.DataFrame(merged)
    return run_df, merged_df, fold_cache


def _recording_groups(payload, indices):
    meta = payload["meta"].iloc[indices].reset_index(drop=True)
    for recording, positions in meta.groupby("recording").groups.items():
        pos = np.asarray(list(positions), dtype=int)
        valid = meta.iloc[pos]["valid_biological_window"].to_numpy(bool)
        pos = pos[valid]
        if recording != "CROSS_RECORDING" and len(pos) >= config.MIN_WINDOWS_RSA:
            yield recording, pos, meta.iloc[pos]


def experiment2_and_3(datasets, fold_cache):
    rsa_rows, decision_rows = [], []
    for run, payload in datasets.items():
        for fold in range(1, 4):
            cached = fold_cache[(run, fold)]
            indices, states = cached["indices"], cached["states"]
            model_features = layer_features(states)
            for recording, pos, meta in _recording_groups(payload, indices):
                n_real = int(meta["n_real_neurons"].iloc[0])
                nf = neural_features(payload["x"][indices[pos]], n_real)
                neural_euc = z_euclidean_rdm(nf)
                neural_corr = correlation_rdm(nf)
                neural_vec = rdm_vector(neural_euc)
                y = payload["y"][indices[pos]]
                decision = (y[:, None] != y[None, :]).astype(float)
                decision_vec = rdm_vector(decision)
                enough_classes = len(np.unique(y)) >= 2
                decision_rows.append({
                    "run": run, "fold": f"fold{fold:02d}", "recording": recording,
                    "representation": "Neural", "n_windows": len(pos),
                    "n_real_neurons": n_real, "n_classes": len(np.unique(y)),
                    "decision_rsa": safe_corr(neural_vec, decision_vec) if enough_classes else np.nan,
                    "partial_decision_rsa_controlling_neural": np.nan,
                })
                for layer, full_feature in model_features.items():
                    mf = full_feature[pos]
                    model_euc = z_euclidean_rdm(mf)
                    model_corr = correlation_rdm(mf)
                    model_vec = rdm_vector(model_euc)
                    rsa_rows.append({
                        "run": run, "fold": f"fold{fold:02d}", "recording": recording,
                        "layer": layer, "n_windows": len(pos), "n_real_neurons": n_real,
                        "linear_cka": linear_cka(nf, mf),
                        "z_euclidean_rsa": safe_corr(neural_vec, model_vec),
                        "correlation_distance_rsa": safe_corr(
                            rdm_vector(neural_corr), rdm_vector(model_corr)
                        ),
                    })
                    decision_rows.append({
                        "run": run, "fold": f"fold{fold:02d}", "recording": recording,
                        "representation": layer, "n_windows": len(pos),
                        "n_real_neurons": n_real, "n_classes": len(np.unique(y)),
                        "decision_rsa": safe_corr(model_vec, decision_vec) if enough_classes else np.nan,
                        "partial_decision_rsa_controlling_neural": partial_rank_corr(
                            model_vec, decision_vec, neural_vec
                        ) if enough_classes else np.nan,
                    })
    return pd.DataFrame(rsa_rows), pd.DataFrame(decision_rows)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-12 else float("nan")


def experiment4_overlap(datasets, fold_cache):
    rows = []
    rng = np.random.default_rng(config.SEED)
    for run, payload in datasets.items():
        for fold in range(1, 4):
            cached = fold_cache[(run, fold)]
            indices, states = cached["indices"], cached["states"]
            meta = payload["meta"].iloc[indices].reset_index(drop=True)
            valid_pos = np.flatnonzero(meta["valid_biological_window"].to_numpy(bool))
            lookup = {
                (meta.iloc[p]["recording"], int(meta.iloc[p]["start_time_bin"])): int(p)
                for p in valid_pos
            }
            seqs = {
                "Conv": states["conv"].transpose(0, 2, 1),
                "LSTM": states["lstm"],
                "Attention contribution": states["contribution"],
                "Attention weights": states["attention"][..., None],
            }
            for p in valid_pos:
                row = meta.iloc[p]
                q = lookup.get((row["recording"], int(row["start_time_bin"]) + config.STEP_SIZE))
                if q is None or int(meta.iloc[q]["run_label"]) != int(row["run_label"]):
                    continue
                candidates = [
                    z for z in valid_pos
                    if z not in (p, q)
                    and meta.iloc[z]["recording"] == row["recording"]
                    and int(meta.iloc[z]["run_label"]) == int(row["run_label"])
                    and abs(int(meta.iloc[z]["start_time_bin"]) - int(row["start_time_bin"])) >= config.WINDOW_SIZE
                ]
                control = int(rng.choice(candidates)) if candidates else None
                for layer, seq in seqs.items():
                    rows.append({
                        "run": run, "fold": f"fold{fold:02d}", "recording": row["recording"],
                        "layer": layer, "pair_type": "Shared 2.5 s overlap",
                        "similarity": cosine_similarity(seq[p, 250:], seq[q, :250]),
                    })
                    if control is not None:
                        rows.append({
                            "run": run, "fold": f"fold{fold:02d}", "recording": row["recording"],
                            "layer": layer, "pair_type": "Same-label non-overlap control",
                            "similarity": cosine_similarity(seq[p, 250:], seq[control, :250]),
                        })
    return pd.DataFrame(rows)


def experiment5_interventions(datasets, device):
    accuracy_rows, alignment_rows = [], []
    run_condition_fold = {}
    for run, payload in datasets.items():
        for fold in range(1, 4):
            indices = np.asarray(load_test_indices(run, fold), dtype=int)
            model = load_model(run, fold, device)
            for condition in config.INTERVENTION_ORDER:
                states = forward_states(model, payload["x"][indices], device, intervention=condition)
                pred = states["logits"].argmax(axis=1)
                metrics = metric_row(payload["y"][indices], pred)
                metrics.update({
                    "run": run, "fold": f"fold{fold:02d}", "condition": condition,
                    "class_accuracy": per_class_accuracy(payload["y"][indices], pred),
                    "context_across_window_sd": float(states["context"].std(axis=0).mean()),
                })
                accuracy_rows.append(metrics)
                run_condition_fold[(run, condition, fold)] = metrics
                meta = payload["meta"].iloc[indices].reset_index(drop=True)
                for recording, positions in meta.groupby("recording").groups.items():
                    pos = np.asarray(list(positions), dtype=int)
                    pos = pos[meta.iloc[pos]["valid_biological_window"].to_numpy(bool)]
                    if recording == "CROSS_RECORDING" or len(pos) < config.MIN_WINDOWS_RSA:
                        continue
                    n_real = int(meta.iloc[pos]["n_real_neurons"].iloc[0])
                    nf = neural_features(payload["x"][indices[pos]], n_real)
                    cf = states["context"][pos]
                    alignment_rows.append({
                        "run": run, "fold": f"fold{fold:02d}", "recording": recording,
                        "condition": condition, "n_windows": len(pos),
                        "context_linear_cka": linear_cka(nf, cf),
                        "context_neural_rsa": safe_corr(
                            rdm_vector(z_euclidean_rdm(nf)),
                            rdm_vector(z_euclidean_rdm(cf)),
                        ),
                    })
    merged_rows = []
    for condition in config.INTERVENTION_ORDER:
        for fold in range(1, 4):
            row = merge_legacy_metrics(
                run_condition_fold[("run1", condition, fold)],
                run_condition_fold[("run2", condition, fold)],
            )
            row.update({"condition": condition, "fold": f"fold{fold:02d}"})
            merged_rows.append(row)
    clean_accuracy = pd.DataFrame([
        {k: v for k, v in row.items() if k != "class_accuracy"} for row in accuracy_rows
    ])
    return clean_accuracy, pd.DataFrame(merged_rows), pd.DataFrame(alignment_rows)


def aggregate_recordings(df: pd.DataFrame, group_cols: List[str], metrics: List[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols, sort=False):
        keys = (keys,) if not isinstance(keys, tuple) else keys
        row = dict(zip(group_cols, keys))
        row["n_recording_fold_rows"] = len(sub)
        for metric in metrics:
            row[metric] = fisher_mean(sub[metric]) if "rsa" in metric else float(sub[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_condition(df: pd.DataFrame, group_cols: List[str], metric: str, unit_col: str) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(group_cols, sort=False):
        keys = (keys,) if not isinstance(keys, tuple) else keys
        mean, low, high = bootstrap_ci(sub[metric])
        row = dict(zip(group_cols, keys))
        row.update({
            "metric": metric, "n_units": int(sub[unit_col].nunique()),
            "n_finite": int(np.isfinite(sub[metric]).sum()),
            "mean": mean, "sd": float(sub[metric].std(ddof=1)),
            "ci95_low": low, "ci95_high": high,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def split_overlap_audit(datasets) -> pd.DataFrame:
    rows = []
    for run, payload in datasets.items():
        meta = payload["meta"]
        for fold in range(1, 4):
            test = np.asarray(load_test_indices(run, fold), dtype=int)
            train = np.setdiff1d(np.arange(len(meta)), test)
            train_keys = defaultdict(list)
            for i in train:
                row = meta.iloc[i]
                if row["valid_biological_window"]:
                    train_keys[row["recording"]].append(int(row["start_time_bin"]))
            shared = []
            for i in test:
                row = meta.iloc[i]
                if not row["valid_biological_window"]:
                    continue
                starts = train_keys.get(row["recording"], [])
                shared.append(any(abs(int(row["start_time_bin"]) - s) < config.WINDOW_SIZE for s in starts))
            rows.append({
                "run": run,
                "fold": f"fold{fold:02d}",
                "n_valid_test_windows": len(shared),
                "n_test_windows_overlapping_a_train_window": int(np.sum(shared)),
                "fraction_overlapping_a_train_window": float(np.mean(shared)) if shared else np.nan,
            })
    return pd.DataFrame(rows)


def run_all_experiments():
    config.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {run: reconstruct_windows(run) for run in config.RUNS}
    audit = [validate_reconstruction(run, payload) for run, payload in datasets.items()]
    pd.DataFrame(audit).to_csv(config.TABLE_DIR / "data_window_audit.csv", index=False, encoding="utf-8-sig")
    for run, payload in datasets.items():
        payload["meta"].to_csv(config.TABLE_DIR / f"{run}_window_manifest.csv", index=False, encoding="utf-8-sig")
    split_overlap_audit(datasets).to_csv(
        config.TABLE_DIR / "split_overlap_leakage_audit.csv", index=False, encoding="utf-8-sig"
    )

    baseline_run, baseline_merged, fold_cache = experiment1_baseline(datasets, device)
    baseline_run.to_csv(config.TABLE_DIR / "experiment1_baseline_by_run_fold.csv", index=False, encoding="utf-8-sig")
    baseline_merged.to_csv(config.TABLE_DIR / "experiment1_baseline_merged.csv", index=False, encoding="utf-8-sig")

    rsa, decision = experiment2_and_3(datasets, fold_cache)
    rsa.to_csv(config.TABLE_DIR / "experiment2_rsa_cka_recording_fold.csv", index=False, encoding="utf-8-sig")
    decision.to_csv(config.TABLE_DIR / "experiment3_decision_rsa_recording_fold.csv", index=False, encoding="utf-8-sig")
    rsa_recording = aggregate_recordings(
        rsa, ["run", "recording", "layer"],
        ["linear_cka", "z_euclidean_rsa", "correlation_distance_rsa"],
    )
    decision_recording = aggregate_recordings(
        decision, ["run", "recording", "representation"],
        ["decision_rsa", "partial_decision_rsa_controlling_neural"],
    )
    rsa_recording.to_csv(config.TABLE_DIR / "experiment2_rsa_cka_by_recording.csv", index=False, encoding="utf-8-sig")
    decision_recording.to_csv(config.TABLE_DIR / "experiment3_decision_rsa_by_recording.csv", index=False, encoding="utf-8-sig")

    overlap = experiment4_overlap(datasets, fold_cache)
    overlap.to_csv(config.TABLE_DIR / "experiment4_overlap_pairs.csv", index=False, encoding="utf-8-sig")
    overlap_recording = (
        overlap.groupby(["run", "recording", "layer", "pair_type"], as_index=False)
        .agg(similarity=("similarity", "mean"), n_pairs=("similarity", "size"))
    )
    overlap_recording.to_csv(
        config.TABLE_DIR / "experiment4_overlap_by_recording.csv",
        index=False, encoding="utf-8-sig"
    )

    intervention_run, intervention_merged, intervention_alignment = experiment5_interventions(datasets, device)
    intervention_run.to_csv(config.TABLE_DIR / "experiment5_intervention_by_run_fold.csv", index=False, encoding="utf-8-sig")
    intervention_merged.to_csv(config.TABLE_DIR / "experiment5_intervention_merged.csv", index=False, encoding="utf-8-sig")
    intervention_alignment.to_csv(config.TABLE_DIR / "experiment5_intervention_alignment_recording_fold.csv", index=False, encoding="utf-8-sig")
    intervention_recording = aggregate_recordings(
        intervention_alignment, ["run", "recording", "condition"],
        ["context_linear_cka", "context_neural_rsa"],
    )
    intervention_recording.to_csv(
        config.TABLE_DIR / "experiment5_intervention_alignment_by_recording.csv",
        index=False, encoding="utf-8-sig",
    )

    summaries = {
        "experiment2": summarize_by_condition(
            rsa_recording, ["run", "layer"], "z_euclidean_rsa", "recording"
        ),
        "experiment2_cka": summarize_by_condition(
            rsa_recording, ["run", "layer"], "linear_cka", "recording"
        ),
        "experiment3": summarize_by_condition(
            decision_recording, ["run", "representation"], "decision_rsa", "recording"
        ),
        "experiment3_partial": summarize_by_condition(
            decision_recording[decision_recording["representation"] != "Neural"],
            ["run", "representation"], "partial_decision_rsa_controlling_neural", "recording"
        ),
        "experiment4": summarize_by_condition(
            overlap_recording, ["run", "layer", "pair_type"], "similarity", "recording"
        ),
        "experiment5": summarize_by_condition(
            intervention_recording, ["run", "condition"], "context_neural_rsa", "recording"
        ),
        "experiment5_cka": summarize_by_condition(
            intervention_recording, ["run", "condition"], "context_linear_cka", "recording"
        ),
    }
    for name, frame in summaries.items():
        frame.to_csv(config.TABLE_DIR / f"{name}_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "device": str(device),
        "protocol": "continuous 5 s, 50% overlap, whole-window label constant",
        "baseline_accuracy_mean": float(baseline_merged["accuracy"].mean()),
        "baseline_accuracy_sd": float(baseline_merged["accuracy"].std(ddof=1)),
        "baseline_recall_mean": float(baseline_merged["recall"].mean()),
        "baseline_f1_mean": float(baseline_merged["f1"].mean()),
        "audit": audit,
        "rsa_recordings": int(rsa_recording["recording"].nunique()),
        "overlap_pairs": int((overlap["pair_type"] == "Shared 2.5 s overlap").sum()),
        "note_experiment5": "Test-time functional interventions in the frozen full model; not retrained ablation models.",
    }
    (config.OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run_all_experiments(), ensure_ascii=False, indent=2))
