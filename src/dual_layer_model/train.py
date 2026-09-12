import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from config import Config, Config3
from data_processing import prepare_data, prepare_data3
from model import ConvFeatureAttentionLSTM


def set_seed(seed=25):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _paths(config, run_tag: str, fold_tag: Optional[str] = None) -> Dict[str, Path]:
    suffix = f"_{fold_tag}" if fold_tag else ""
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "jsonl": out / f"train_results_{run_tag}{suffix}.jsonl",
        "log": out / f"train_results_{run_tag}{suffix}.log",
        "best_model": out / f"best_test_model_{run_tag}{suffix}.pth",
        "best_json": out / f"best_results_{run_tag}{suffix}.json",
        "best_log": out / f"best_results_{run_tag}{suffix}.log",
    }


def _append_jsonl(path: Path, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _append_log(path: Path, text_block: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text_block.rstrip() + "\n")


def _compute_extra_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray], num_classes: int):
    if y_true.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mcc": 0.0, "auc": 0.0, "aupr": 0.0}
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    auc = 0.0
    aupr = 0.0
    if y_prob is not None and y_prob.shape[0] == y_true.shape[0] and y_prob.shape[1] == num_classes:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        except Exception:
            auc = 0.0
        try:
            aupr = average_precision_score(np.eye(num_classes)[y_true], y_prob, average="macro")
        except Exception:
            aupr = 0.0
    return {
        "precision": float(precision * 100.0),
        "recall": float(recall * 100.0),
        "f1": float(f1 * 100.0),
        "mcc": float(mcc * 100.0),
        "auc": float(auc * 100.0),
        "aupr": float(aupr * 100.0),
    }


def _format_epoch_text(epoch_idx_1based: int, loss: float, train_acc: float, eval_pack: dict):
    lines = [
        f"Epoch {epoch_idx_1based:02d} | Loss: {loss:.4f} | Train Acc: {train_acc:.2f}%",
        f"Test Accuracy at Epoch {epoch_idx_1based:02d}: {eval_pack['accuracy']:.2f}%",
        "Test Per-class Accuracy:",
    ]
    cls_ids = sorted(int(k.split("_")[1]) for k in eval_pack["per_class"].keys() if k.startswith("class_"))
    for cid in cls_ids:
        meta = eval_pack["per_class"].get(f"class_{cid}", {"acc": 0.0, "correct": 0, "total": 0})
        lines.append(f"  Class {cid}: {meta['acc']:.2f}% ({meta['correct']}/{meta['total']})")
    return "\n".join(lines)


@torch.no_grad()
def evaluate_model(model, loader, device, num_classes):
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    y_true, y_pred = [], []
    y_prob_chunks = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.squeeze().to(device)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
        y_pred.extend(preds.detach().cpu().numpy().astype(int).tolist())
        y_prob_chunks.append(probs.detach().cpu().numpy())
        for i in range(len(labels)):
            y = int(labels[i].item())
            p = int(preds[i].item())
            if y == p:
                class_correct[y] += 1
                correct += 1
            class_total[y] += 1
            total += 1

    acc = 100.0 * correct / max(total, 1)
    per_class = {
        f"class_{i}": {
            "acc": 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0,
            "correct": class_correct[i],
            "total": class_total[i],
        }
        for i in range(num_classes)
    }
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.concatenate(y_prob_chunks, axis=0) if y_prob_chunks else None
    return {"accuracy": acc, "per_class": per_class, "metrics": _compute_extra_metrics(y_true, y_pred, y_prob, num_classes)}


def _train_core_fixed_split(make_dataset_fn, config, run_tag: str, fold_tag: str, train_idx: List[int], test_idx: List[int]):
    set_seed(config.seed)
    paths = _paths(config, run_tag, fold_tag)
    dataset = make_dataset_fn(config)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, train_idx),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, test_idx),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = ConvFeatureAttentionLSTM(
        input_size=dataset[0][0].shape[0],
        hidden_size=config.hidden_size,
        attention_hidden_size=config.attention_hidden_size,
        num_classes=config.num_classes,
        conv_channels=config.conv_channels,
        kernel_size=config.kernel_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(config.device)

    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(dataset.labels), y=dataset.labels)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float, device=config.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    for k in ("jsonl", "log"):
        if paths[k].exists():
            paths[k].unlink()

    best = {"epoch": -1, "train_loss": None, "train_acc": -1.0, "test": {"accuracy": -1.0, "per_class": {}, "metrics": {}}}

    for epoch in range(config.epochs):
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs = inputs.to(config.device)
            labels = labels.squeeze().to(config.device)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            bs = labels.size(0)
            total_loss += float(loss.item())
            correct += int((torch.argmax(outputs, dim=1) == labels).sum().item())
            seen += bs

        epoch_loss = total_loss / max(len(train_loader), 1)
        train_acc = 100.0 * correct / max(seen, 1)
        eval_pack = evaluate_model(model, test_loader, config.device, config.num_classes)
        record = {"epoch": epoch + 1, "train_loss": epoch_loss, "train_acc": train_acc, "test": eval_pack}
        _append_jsonl(paths["jsonl"], record)
        _append_log(paths["log"], _format_epoch_text(epoch + 1, epoch_loss, train_acc, eval_pack))

        # Keep the original behaviour: select best epoch by test accuracy.
        if eval_pack["accuracy"] > best["test"]["accuracy"]:
            best = {"epoch": epoch + 1, "train_loss": epoch_loss, "train_acc": train_acc, "test": eval_pack}
            torch.save(model.state_dict(), paths["best_model"])

    with open(paths["best_json"], "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)
    _append_log(paths["best_log"], _format_epoch_text(best["epoch"], best["train_loss"], best["train_acc"], best["test"]))
    return paths


def _load_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@torch.no_grad()
def _predict_with_best_model(make_dataset_fn, config, run_tag: str, fold_tag: str, subset_indices: List[int], out_prefix: str):
    paths = _paths(config, run_tag, fold_tag)
    dataset = make_dataset_fn(config)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, subset_indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    model = ConvFeatureAttentionLSTM(
        input_size=dataset[0][0].shape[0],
        hidden_size=config.hidden_size,
        attention_hidden_size=config.attention_hidden_size,
        num_classes=config.num_classes,
        conv_channels=config.conv_channels,
        kernel_size=config.kernel_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(config.device)
    state = torch.load(paths["best_model"], map_location=config.device)
    model.load_state_dict(state)
    model.eval()

    all_true, all_pred, all_probs = [], [], []
    for inputs, labels in loader:
        inputs = inputs.to(config.device)
        labels = labels.squeeze()
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        all_true.extend(labels.cpu().numpy().astype(int).tolist())
        all_pred.extend(preds.cpu().numpy().astype(int).tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.asarray(all_true, dtype=int)
    y_pred = np.asarray(all_pred, dtype=int)
    y_prob = np.asarray(all_probs, dtype=float)
    metrics = _compute_extra_metrics(y_true, y_pred, y_prob, config.num_classes)
    per_class = {}
    for c in range(config.num_classes):
        mask = y_true == c
        total = int(mask.sum())
        correct = int((y_pred[mask] == c).sum()) if total else 0
        per_class[f"class_{c}"] = {"acc": 100.0 * correct / total if total else 0.0, "correct": correct, "total": total}
    acc = 100.0 * int((y_true == y_pred).sum()) / max(len(y_true), 1)

    pred_csv = Path(config.output_dir) / f"pred_{run_tag}_{fold_tag}__{out_prefix}.csv"
    with open(pred_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["global_index", "true", "pred"] + [f"prob_class_{i}" for i in range(config.num_classes)])
        for i, dataset_i in enumerate(subset_indices):
            writer.writerow([int(dataset_i), int(all_true[i]), int(all_pred[i])] + [float(v) for v in all_probs[i]])

    with open(paths["best_json"], "r", encoding="utf-8") as f:
        best = json.load(f)
    record = {"epoch": int(best["epoch"]), "train_loss": best["train_loss"], "train_acc": best["train_acc"], "test": {"accuracy": acc, "per_class": per_class, "metrics": metrics}}
    pred_jsonl = Path(config.output_dir) / f"predict_results_{run_tag}_{fold_tag}__{out_prefix}.jsonl"
    pred_log = Path(config.output_dir) / f"predict_results_{run_tag}_{fold_tag}__{out_prefix}.log"
    _append_jsonl(pred_jsonl, record)
    _append_log(pred_log, _format_epoch_text(record["epoch"], record["train_loss"], record["train_acc"], record["test"]))
    return {"jsonl": pred_jsonl, "csv_path": pred_csv, "metrics": {"accuracy": acc, **metrics}}


def _safe_get_class_acc(rec: dict, cls_id: int) -> float:
    return float(rec["test"]["per_class"].get(f"class_{cls_id}", {}).get("acc", 0.0))


def _safe_metrics(rec: dict) -> Dict[str, float]:
    m = rec["test"].get("metrics", {})
    return {k: float(m.get(k, 0.0)) for k in ("precision", "recall", "f1", "mcc", "auc", "aupr")}


def output_merged_predictions(file1_pred_jsonl: Path, file2_pred_jsonl: Path, log_path: Path):
    a = _load_jsonl(file1_pred_jsonl)[-1]
    b = _load_jsonl(file2_pred_jsonl)[-1]
    ep = int(a["epoch"])

    train_acc_avg = (float(a["train_acc"]) + float(b["train_acc"])) / 2.0
    acc_avg = (float(a["test"]["accuracy"]) + float(b["test"]["accuracy"])) / 2.0

    a_c0, a_c1, a_c2 = _safe_get_class_acc(a, 0), _safe_get_class_acc(a, 1), _safe_get_class_acc(a, 2)
    b_c0, b_c1, b_c2 = _safe_get_class_acc(b, 0), _safe_get_class_acc(b, 1), _safe_get_class_acc(b, 2)
    c0 = a_c0
    c2 = a_c2
    c1 = (a_c1 + b_c0) / 2.0
    c3 = (a_c1 + b_c1) / 2.0
    c4 = (a_c1 + b_c2) / 2.0

    ma, mb = _safe_metrics(a), _safe_metrics(b)
    merged = {
        "accuracy": acc_avg,
        "precision": (ma["precision"] + mb["precision"]) / 2.0,
        "recall": (ma["recall"] + mb["recall"]) / 2.0,
        "f1": (ma["f1"] + mb["f1"]) / 2.0,
        "mcc": (ma["mcc"] + mb["mcc"]) / 2.0,
        "auc": (ma["auc"] + mb["auc"]) / 2.0,
        "aupr": (ma["aupr"] + mb["aupr"]) / 2.0,
        "c0": c0,
        "c1": c1,
        "c2": c2,
        "c3": c3,
        "c4": c4,
    }
    text = (
        f"Train Acc: {train_acc_avg:.2f}%\n"
        f"Test Accuracy at Epoch {ep}: {acc_avg:.2f}%\n"
        f"F1 Score: {merged['f1']:.2f}%\n"
        f"Recall: {merged['recall']:.2f}%\n"
        f"MCC: {merged['mcc']:.2f}%\n"
        f"AUC: {merged['auc']:.2f}%\n"
        f"AUPR: {merged['aupr']:.2f}%\n"
        "Test Per-class Accuracy:\n"
        f"  Class 0: {c0:.2f}%\n"
        f"  Class 1: {c1:.2f}%\n"
        f"  Class 2: {c2:.2f}%\n"
        f"  Class 3: {c3:.2f}%\n"
        f"  Class 4: {c4:.2f}%"
    )
    print(text)
    _append_log(log_path, text)
    return merged


def _kfold_indices_from_labels(labels: np.ndarray, n_splits: int, seed: int):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(np.zeros_like(labels), labels), start=1):
        yield fold_id, train_idx.tolist(), test_idx.tolist()


def _save_fold_split(config, run_tag: str, labels: np.ndarray, train_idx: List[int], test_idx: List[int], fold_tag: str):
    path = Path(config.output_dir) / f"fold_split_{run_tag}_{fold_tag}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run": run_tag,
                "fold": fold_tag,
                "dataset_size": int(len(labels)),
                "train_indices": list(map(int, train_idx)),
                "test_indices": list(map(int, test_idx)),
                "train_labels": [int(labels[i]) for i in train_idx],
                "test_labels": [int(labels[i]) for i in test_idx],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def _summarize_cross_folds(results: List[Dict[str, float]], config):
    if not results:
        return
    keys = ["accuracy", "precision", "recall", "f1", "mcc", "auc", "aupr"]
    cls_keys = ["c0", "c1", "c2", "c3", "c4"]

    def mean_std(vals):
        arr = np.asarray(vals, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    lines = ["===== Legacy Cross-Fold Summary ====="]
    for key in keys:
        m, s = mean_std([d[key] for d in results])
        lines.append(f"{key}: {m:.2f}% ± {s:.2f}%")
    lines.append("")
    lines.append("Average Per-Class Accuracy:")
    for i, key in enumerate(cls_keys):
        m, s = mean_std([d[key] for d in results])
        lines.append(f"  Class {i}: {m:.2f}% ± {s:.2f}%")

    text = "\n".join(lines)
    print(text)
    with open(Path(config.output_dir) / "cross_fold_summary__legacy_on_fold_valid.log", "w", encoding="utf-8") as f:
        f.write(text + "\n")


def train_predict_kfold_legacy():
    set_seed(Config.seed)
    Path(Config.output_dir).mkdir(parents=True, exist_ok=True)

    merged_fold_results = []

    ds1 = prepare_data(Config())
    labels1 = np.asarray(ds1.labels)
    run1_pred_jsonl_by_fold = {}

    for fold_id, train_idx, test_idx in _kfold_indices_from_labels(labels1, Config.n_splits, Config.seed):
        fold_tag = f"fold{fold_id:02d}"
        print(f"===== {fold_tag} | run1 legacy =====")
        _save_fold_split(Config(), "run1", labels1, train_idx, test_idx, fold_tag)
        _train_core_fixed_split(prepare_data, Config(), "run1", fold_tag, train_idx, test_idx)
        pred_valid = _predict_with_best_model(prepare_data, Config(), "run1", fold_tag, test_idx, "on_fold_valid")
        run1_pred_jsonl_by_fold[fold_tag] = pred_valid["jsonl"]

    ds2 = prepare_data3(Config3())
    labels2 = np.asarray(ds2.labels)

    for fold_id, train_idx, test_idx in _kfold_indices_from_labels(labels2, Config3.n_splits, Config3.seed):
        fold_tag = f"fold{fold_id:02d}"
        print(f"===== {fold_tag} | run2 legacy =====")
        _save_fold_split(Config3(), "run2", labels2, train_idx, test_idx, fold_tag)
        _train_core_fixed_split(prepare_data3, Config3(), "run2", fold_tag, train_idx, test_idx)
        pred_valid_2 = _predict_with_best_model(prepare_data3, Config3(), "run2", fold_tag, test_idx, "on_fold_valid")

        if fold_tag in run1_pred_jsonl_by_fold:
            merged_log = Path(Config.output_dir) / f"merged_fiveclass_{fold_tag}__on_fold_valid.log"
            merged = output_merged_predictions(run1_pred_jsonl_by_fold[fold_tag], pred_valid_2["jsonl"], merged_log)
            merged_fold_results.append(merged)

    _summarize_cross_folds(merged_fold_results, Config())


if __name__ == "__main__":
    train_predict_kfold_legacy()
