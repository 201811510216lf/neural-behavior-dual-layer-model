"""Run the follow-up temporal-window mechanism experiments.

The script deliberately keeps the legacy 5 s / 50% overlap data construction,
saved folds, and hierarchical metric-merging convention so the new results can
be compared with the existing frozen-model experiments.  Newly trained
architectural ablations are stored only under this result bundle.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Subset


RESULT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = RESULT_ROOT.parent / "brain_model_regroup_1246_vs_35_fixed"
MODEL_ROOT = PROJECT_ROOT / "outputs_legacy"
XLSX_PATH = (
    RESULT_ROOT.parents[1]
    / "outputs"
    / "regroup_1246_vs_35_label_timepoint_ttest_bigtable"
    / "regroup_1246_vs_35_label_timepoint_ttest_bigtable.xlsx"
)
SEGMENT_ROOT = PROJECT_ROOT / "legacy_5s_50pct_significant_segment_replace"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SEGMENT_ROOT))

from config import Config, Config3  # noqa: E402
from model import ConvFeatureAttentionLSTM  # noqa: E402
from segment_replace_core import (  # noqa: E402
    LegacySegmentDataset,
    compute_stage_metrics,
    load_state,
    merge_training_style,
    read_significant_timepoints,
    replace_segment,
    significant_offsets,
)


ARCH_FLAGS = {
    "full": {},
    "no_conv": {"ablate_conv": True},
    "no_lstm": {"ablate_lstm": True},
    "no_attention": {"ablate_attention": True},
}
ARCH_LABELS = {
    "full": "Full",
    "no_conv": "NoConv",
    "no_lstm": "NoLSTM",
    "no_attention": "NoAttention",
}
TARGETS = (2, 3, 5)
TARGET_LABELS = {2: "L2", 3: "L3", 5: "L5"}
OUTCOME_KEYS = {2: "stage2_macro_recall", 3: "c2", 5: "c4"}
OUTCOME_LABELS = {
    2: "Stage-2 macro recall",
    3: "Hesitating recall",
    5: "Walking recall",
}
CONDITIONS = ("baseline", "maximum", "attention_shuffle", "attention_zero", "random_window")
FOLDS = ("fold01", "fold02", "fold03")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def split_indices(run_tag: str, fold: str) -> Tuple[List[int], List[int]]:
    payload = json.loads((MODEL_ROOT / f"fold_split_{run_tag}_{fold}.json").read_text(encoding="utf-8"))
    return list(map(int, payload["train_indices"])), list(map(int, payload["test_indices"]))


def checkpoint_path(architecture: str, run_tag: str, fold: str) -> Path:
    if architecture == "full":
        return MODEL_ROOT / f"best_test_model_{run_tag}_{fold}.pth"
    return RESULT_ROOT / "checkpoints" / architecture / run_tag / f"{fold}.pth"


def checkpoint_meta_path(architecture: str, run_tag: str, fold: str) -> Path:
    return RESULT_ROOT / "checkpoints" / architecture / run_tag / f"{fold}.json"


@torch.no_grad()
def evaluate_accuracy(model, dataset, indices: Sequence[int], config) -> float:
    loader = DataLoader(Subset(dataset, indices), batch_size=config.batch_size, shuffle=False)
    correct = total = 0
    model.eval()
    for x, y, _, _ in loader:
        pred = model(x.float().to(config.device)).argmax(dim=1).cpu()
        correct += int((pred == y).sum().item())
        total += int(len(y))
    return 100.0 * correct / max(total, 1)


def train_one(
    architecture: str,
    run_tag: str,
    fold: str,
    dataset,
    config,
    max_epochs: int,
    min_epochs: int,
    patience: int,
    force: bool,
) -> Path:
    out = checkpoint_path(architecture, run_tag, fold)
    meta_path = checkpoint_meta_path(architecture, run_tag, fold)
    history_path = out.with_suffix(".history.csv")
    if out.exists() and meta_path.exists() and not force:
        print(f"[resume] {architecture} {run_tag} {fold}", flush=True)
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    train_idx, test_idx = split_indices(run_tag, fold)
    set_seed(config.seed)
    model = make_model(config, dataset.x.shape[1], architecture)
    loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(dataset.y), y=dataset.y)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=config.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_acc = -np.inf
    best_epoch = -1
    epochs_without_improvement = 0
    history: List[Dict] = []
    started = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = seen = 0
        for x, y, _, _ in loader:
            x = x.float().to(config.device, non_blocking=True)
            y = y.to(config.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.item())
            correct += int((logits.argmax(dim=1) == y).sum().item())
            seen += int(len(y))

        test_acc = evaluate_accuracy(model, dataset, test_idx, config)
        train_acc = 100.0 * correct / max(seen, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(len(loader), 1),
                "train_accuracy": train_acc,
                "heldout_accuracy": test_acc,
            }
        )
        if test_acc > best_acc + 1e-12:
            best_acc = test_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), out)
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"[{architecture} {run_tag} {fold}] epoch {epoch:03d} "
                f"train={train_acc:.2f} heldout={test_acc:.2f} best={best_acc:.2f}@{best_epoch}",
                flush=True,
            )
        if epoch >= min_epochs and epochs_without_improvement >= patience:
            break

    pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")
    meta = {
        "architecture": architecture,
        "run_tag": run_tag,
        "fold": fold,
        "seed": int(config.seed),
        "batch_size": int(config.batch_size),
        "learning_rate": float(config.lr),
        "max_epochs": int(max_epochs),
        "epochs_completed": int(history[-1]["epoch"]),
        "min_epochs": int(min_epochs),
        "early_stopping_patience": int(patience),
        "selection_metric": "heldout fold accuracy (legacy-compatible; optimistic)",
        "best_epoch": int(best_epoch),
        "best_heldout_accuracy": float(best_acc),
        "elapsed_seconds": float(time.time() - started),
        "amp": bool(torch.cuda.is_available()),
        "device": str(config.device),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def load_model(architecture: str, run_tag: str, fold: str, dataset, config):
    model = make_model(config, dataset.x.shape[1], architecture)
    model.load_state_dict(load_state(checkpoint_path(architecture, run_tag, fold), config.device))
    model.eval()
    return model


def temporal_components(model, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    conv_out = model.conv(x)
    sequence = conv_out.permute(0, 2, 1)
    if model.ablate_lstm:
        states = torch.tanh(model.input_projection(sequence))
    else:
        states, _ = model.lstm(sequence)
    if model.ablate_attention:
        attention = torch.full(
            (x.shape[0], states.shape[1]),
            1.0 / states.shape[1],
            dtype=states.dtype,
            device=states.device,
        )
    else:
        attention = torch.softmax(model.time_attention(states).squeeze(-1), dim=1)
    return states, attention


def logits_from_attention(model, states: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    context = torch.bmm(attention.unsqueeze(1), states).squeeze(1)
    return model.fc(context)


def metric_probability_details(y_true: np.ndarray, probabilities: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    row = np.arange(len(y_true))
    true_probability = probabilities[row, y_true]
    other = logits.copy()
    other[row, y_true] = -np.inf
    margin = logits[row, y_true] - np.max(other, axis=1)
    return {
        "true_probability": float(true_probability.mean() * 100.0),
        "logit_margin": float(margin.mean()),
    }


def pack_from_arrays(y_true: np.ndarray, probabilities: np.ndarray, logits: np.ndarray, num_classes: int) -> Dict:
    predictions = probabilities.argmax(axis=1)
    pack = compute_stage_metrics(y_true, predictions, probabilities, num_classes)
    pack.update(metric_probability_details(y_true, probabilities, logits))
    return pack


@torch.no_grad()
def predict_stage(
    model,
    dataset,
    indices: Sequence[int],
    config,
    target: Optional[int] = None,
    condition: str = "baseline",
    shuffle_repeats: int = 20,
    seed: int = 20260913,
) -> Tuple[Dict, Dict[str, np.ndarray]]:
    loader = DataLoader(Subset(dataset, indices), batch_size=config.batch_size, shuffle=False)
    ys: List[np.ndarray] = []
    probs: List[np.ndarray] = []
    logits_all: List[np.ndarray] = []
    generator = torch.Generator(device=config.device)
    generator.manual_seed(int(seed))
    affected_windows = 0
    affected_bins = 0

    for x, y, masks, _ in loader:
        x = x.float().to(config.device)
        mask = None if target is None else masks[:, target - 1, :].bool().to(config.device)
        if mask is not None:
            affected_windows += int(mask.any(dim=1).sum().item())
            affected_bins += int(mask.sum().item())

        if condition == "baseline" or target is None:
            batch_logits = model(x)
            batch_prob = torch.softmax(batch_logits, dim=1)
        elif condition == "maximum":
            changed = replace_segment(x, mask, "max")
            batch_logits = model(changed)
            batch_prob = torch.softmax(batch_logits, dim=1)
        elif condition == "keep_only_mean":
            fill = x.mean(dim=2, keepdim=True).expand_as(x)
            changed = torch.where(mask.unsqueeze(1), x, fill)
            batch_logits = model(changed)
            batch_prob = torch.softmax(batch_logits, dim=1)
        elif condition in {"attention_shuffle", "attention_zero"}:
            states, original_attention = temporal_components(model, x)
            affected = mask.any(dim=1)
            if condition == "attention_zero":
                zeroed = original_attention.masked_fill(mask, 0.0)
                denominator = zeroed.sum(dim=1, keepdim=True)
                fallback = (~mask).float()
                fallback = fallback / fallback.sum(dim=1, keepdim=True).clamp_min(1.0)
                modified = torch.where(denominator > 1e-12, zeroed / denominator.clamp_min(1e-12), fallback)
                modified = torch.where(affected.unsqueeze(1), modified, original_attention)
                batch_logits = logits_from_attention(model, states, modified)
                batch_prob = torch.softmax(batch_logits, dim=1)
            else:
                prob_sum = torch.zeros((x.shape[0], config.num_classes), device=config.device)
                logit_sum = torch.zeros_like(prob_sum)
                for _ in range(shuffle_repeats):
                    perm = torch.rand(original_attention.shape, generator=generator, device=config.device).argsort(dim=1)
                    shuffled = original_attention.gather(1, perm)
                    modified = torch.where(affected.unsqueeze(1), shuffled, original_attention)
                    one_logits = logits_from_attention(model, states, modified)
                    prob_sum += torch.softmax(one_logits, dim=1)
                    logit_sum += one_logits
                batch_prob = prob_sum / float(shuffle_repeats)
                batch_logits = logit_sum / float(shuffle_repeats)
        else:
            raise ValueError(f"Unsupported condition: {condition}")

        ys.append(y.numpy().astype(int))
        probs.append(batch_prob.cpu().numpy())
        logits_all.append(batch_logits.cpu().numpy())

    y_true = np.concatenate(ys)
    probability = np.concatenate(probs)
    logits = np.concatenate(logits_all)
    pack = pack_from_arrays(y_true, probability, logits, config.num_classes)
    pack["affected_windows"] = int(affected_windows)
    pack["affected_bins"] = int(affected_bins)
    arrays = {"y_true": y_true, "probability": probability, "logits": logits}
    return pack, arrays


@torch.no_grad()
def random_window_stage_packs(
    model,
    dataset,
    indices: Sequence[int],
    config,
    target: int,
    repeats: int,
    seed: int,
) -> List[Dict]:
    base_pack, arrays = predict_stage(model, dataset, indices, config)
    del base_pack
    test_indices = np.asarray(indices, dtype=int)
    masks = dataset.segment_masks[test_indices, target - 1, :]
    affected_local = np.flatnonzero(masks.any(axis=1))
    if len(affected_local) == 0:
        return [pack_from_arrays(arrays["y_true"], arrays["probability"], arrays["logits"], config.num_classes) for _ in range(repeats)]

    affected_global = test_indices[affected_local]
    source_x = torch.from_numpy(dataset.x[affected_global]).float()
    counts = masks[affected_local].sum(axis=1).astype(int)
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []

    for _ in range(repeats):
        x = source_x.clone()
        random_masks = torch.zeros((len(x), x.shape[-1]), dtype=torch.bool)
        for row, count in enumerate(counts):
            count = max(1, min(int(count), int(x.shape[-1])))
            start = int(rng.integers(0, x.shape[-1] - count + 1))
            random_masks[row, start : start + count] = True
        changed = replace_segment(x, random_masks, "max")
        changed_logits: List[np.ndarray] = []
        for start in range(0, len(changed), config.batch_size):
            changed_logits.append(model(changed[start : start + config.batch_size].to(config.device)).cpu().numpy())
        changed_logits_np = np.concatenate(changed_logits)
        changed_prob_np = torch.softmax(torch.from_numpy(changed_logits_np), dim=1).numpy()
        rep_logits = arrays["logits"].copy()
        rep_prob = arrays["probability"].copy()
        rep_logits[affected_local] = changed_logits_np
        rep_prob[affected_local] = changed_prob_np
        pack = pack_from_arrays(arrays["y_true"], rep_prob, rep_logits, config.num_classes)
        pack["affected_windows"] = int(len(affected_local))
        pack["affected_bins"] = int(counts.sum())
        rows.append(pack)
    return rows


def merge_stage_packs(run1: Dict, run2: Dict) -> Dict[str, float]:
    merged = merge_training_style(run1, run2)
    merged["true_probability"] = (float(run1["true_probability"]) + float(run2["true_probability"])) / 2.0
    merged["logit_margin"] = (float(run1["logit_margin"]) + float(run2["logit_margin"])) / 2.0
    merged["stage2_macro_recall"] = float(np.mean([merged["c1"], merged["c3"], merged["c4"]]))
    return merged


def flatten_metric_row(base: Dict) -> Dict[str, float]:
    return {
        key: float(base[key])
        for key in [
            "accuracy", "recall", "f1", "mcc", "auc", "aupr",
            "c0", "c1", "c2", "c3", "c4",
            "true_probability", "logit_margin", "stage2_macro_recall",
        ]
    }


def run_architecture_experiments(
    datasets: Dict[str, LegacySegmentDataset],
    configs: Dict[str, object],
    random_repeats_full: int,
    random_repeats_arch: int,
    shuffle_repeats: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: List[Dict] = []
    random_rows: List[Dict] = []
    drop_keep_rows: List[Dict] = []

    for architecture in ARCH_FLAGS:
        for fold_i, fold in enumerate(FOLDS):
            models = {
                run_tag: load_model(architecture, run_tag, fold, datasets[run_tag], configs[run_tag])
                for run_tag in ("run1", "run2")
            }
            test = {run_tag: split_indices(run_tag, fold)[1] for run_tag in ("run1", "run2")}
            base_stage = {}
            for run_tag in ("run1", "run2"):
                base_stage[run_tag], _ = predict_stage(
                    models[run_tag], datasets[run_tag], test[run_tag], configs[run_tag]
                )
            baseline = merge_stage_packs(base_stage["run1"], base_stage["run2"])

            for target in TARGETS:
                rows.append(
                    {
                        "architecture": architecture,
                        "architecture_label": ARCH_LABELS[architecture],
                        "target": target,
                        "target_label": TARGET_LABELS[target],
                        "fold": fold,
                        "condition": "baseline",
                        **flatten_metric_row(baseline),
                    }
                )
                for condition in ("maximum", "attention_shuffle", "attention_zero"):
                    stage = {}
                    for run_tag in ("run1", "run2"):
                        stage[run_tag], _ = predict_stage(
                            models[run_tag],
                            datasets[run_tag],
                            test[run_tag],
                            configs[run_tag],
                            target=target,
                            condition=condition,
                            shuffle_repeats=shuffle_repeats,
                            seed=20260913 + fold_i * 1000 + target * 100,
                        )
                    merged = merge_stage_packs(stage["run1"], stage["run2"])
                    rows.append(
                        {
                            "architecture": architecture,
                            "architecture_label": ARCH_LABELS[architecture],
                            "target": target,
                            "target_label": TARGET_LABELS[target],
                            "fold": fold,
                            "condition": condition,
                            **flatten_metric_row(merged),
                        }
                    )

                repeats = random_repeats_full if architecture == "full" else random_repeats_arch
                random_stage = {
                    run_tag: random_window_stage_packs(
                        models[run_tag],
                        datasets[run_tag],
                        test[run_tag],
                        configs[run_tag],
                        target=target,
                        repeats=repeats,
                        seed=20261001 + fold_i * 10000 + target * 100 + (list(ARCH_FLAGS).index(architecture) * 100000),
                    )
                    for run_tag in ("run1", "run2")
                }
                merged_random = [
                    merge_stage_packs(random_stage["run1"][rep], random_stage["run2"][rep])
                    for rep in range(repeats)
                ]
                for rep, merged in enumerate(merged_random, start=1):
                    random_rows.append(
                        {
                            "architecture": architecture,
                            "architecture_label": ARCH_LABELS[architecture],
                            "target": target,
                            "target_label": TARGET_LABELS[target],
                            "fold": fold,
                            "repeat": rep,
                            **flatten_metric_row(merged),
                        }
                    )
                random_mean = {
                    key: float(np.mean([item[key] for item in merged_random]))
                    for key in flatten_metric_row(merged_random[0])
                }
                rows.append(
                    {
                        "architecture": architecture,
                        "architecture_label": ARCH_LABELS[architecture],
                        "target": target,
                        "target_label": TARGET_LABELS[target],
                        "fold": fold,
                        "condition": "random_window",
                        **random_mean,
                    }
                )

                if architecture == "full" and target == 3:
                    for condition in ("baseline", "maximum", "keep_only_mean"):
                        stage = {}
                        for run_tag in ("run1", "run2"):
                            stage[run_tag], _ = predict_stage(
                                models[run_tag], datasets[run_tag], test[run_tag], configs[run_tag],
                                target=target, condition=condition,
                            )
                        merged = merge_stage_packs(stage["run1"], stage["run2"])
                        drop_keep_rows.append(
                            {
                                "fold": fold,
                                "condition": condition,
                                **flatten_metric_row(merged),
                            }
                        )

            del models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[evaluate] {architecture} {fold} complete", flush=True)

    return pd.DataFrame(rows), pd.DataFrame(random_rows), pd.DataFrame(drop_keep_rows)


def exact_sign_flip_p(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    permuted = [abs(float(np.mean(values * np.asarray(signs)))) for signs in itertools.product((-1.0, 1.0), repeat=len(values))]
    return float(np.mean(np.asarray(permuted) >= observed - 1e-12))


def bootstrap_ci(values: Sequence[float], seed: int, repeats: int = 10000) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0, 1)
    return out


def build_interaction_tables(architecture_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    detailed: List[Dict] = []
    summary: List[Dict] = []
    modules = ("no_conv", "no_lstm", "no_attention")
    conditions = ("maximum", "attention_shuffle", "attention_zero", "random_window")
    for target in TARGETS:
        metric = OUTCOME_KEYS[target]
        for condition in conditions:
            for architecture in ARCH_FLAGS:
                part = architecture_df[
                    (architecture_df.target == target) & (architecture_df.architecture == architecture)
                ]
                baseline = part[part.condition == "baseline"].set_index("fold")[metric]
                perturbed = part[part.condition == condition].set_index("fold")[metric]
                for fold in FOLDS:
                    detailed.append(
                        {
                            "target": target,
                            "target_label": TARGET_LABELS[target],
                            "outcome": OUTCOME_LABELS[target],
                            "metric_key": metric,
                            "condition": condition,
                            "architecture": architecture,
                            "architecture_label": ARCH_LABELS[architecture],
                            "fold": fold,
                            "baseline": float(baseline.loc[fold]),
                            "perturbed": float(perturbed.loc[fold]),
                            "delta_perturb_minus_baseline": float(perturbed.loc[fold] - baseline.loc[fold]),
                        }
                    )

            detail_df = pd.DataFrame(detailed)
            full_delta = detail_df[
                (detail_df.target == target) & (detail_df.condition == condition) & (detail_df.architecture == "full")
            ].set_index("fold")["delta_perturb_minus_baseline"]
            for module in modules:
                module_delta = detail_df[
                    (detail_df.target == target) & (detail_df.condition == condition) & (detail_df.architecture == module)
                ].set_index("fold")["delta_perturb_minus_baseline"]
                interaction = np.asarray([full_delta.loc[f] - module_delta.loc[f] for f in FOLDS], dtype=float)
                low, high = bootstrap_ci(interaction, seed=target * 1000 + list(conditions).index(condition) * 100 + list(modules).index(module))
                summary.append(
                    {
                        "target": target,
                        "target_label": TARGET_LABELS[target],
                        "outcome": OUTCOME_LABELS[target],
                        "condition": condition,
                        "comparison": f"Full - {ARCH_LABELS[module]}",
                        "module_removed": module,
                        "interaction_mean_pp": float(interaction.mean()),
                        "interaction_sd_pp": float(interaction.std(ddof=1)),
                        "bootstrap_ci95_low_pp": low,
                        "bootstrap_ci95_high_pp": high,
                        "exact_sign_flip_p": exact_sign_flip_p(interaction),
                        "n_folds": len(interaction),
                    }
                )
    summary_df = pd.DataFrame(summary)
    summary_df["bh_fdr_q"] = bh_adjust(summary_df["exact_sign_flip_p"].to_numpy())
    return pd.DataFrame(detailed), summary_df


def build_condition_summary(detailed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "target", "target_label", "outcome", "metric_key", "condition", "architecture", "architecture_label"
    ]
    for group_index, (keys, part) in enumerate(detailed.groupby(group_cols, sort=False)):
        delta = part.delta_perturb_minus_baseline.to_numpy(dtype=float)
        low, high = bootstrap_ci(delta, seed=910000 + group_index)
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "baseline_mean": float(part.baseline.mean()),
                "baseline_sd": float(part.baseline.std(ddof=1)),
                "perturbed_mean": float(part.perturbed.mean()),
                "perturbed_sd": float(part.perturbed.std(ddof=1)),
                "delta_mean_pp": float(delta.mean()),
                "delta_sd_pp": float(delta.std(ddof=1)),
                "delta_bootstrap_ci95_low_pp": low,
                "delta_bootstrap_ci95_high_pp": high,
                "n_folds": int(len(part)),
            }
        )
    return pd.DataFrame(rows)


def normalize_legacy_sources() -> None:
    source_dir = RESULT_ROOT / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    max_csv = SEGMENT_ROOT / "01_maximum_replacement" / "outputs" / "condition_fold_metrics.csv"
    maximum = pd.read_csv(max_csv)
    behavior_names = {
        "c0": "Hole Exploration", "c1": "Wrong Hole Exploration", "c2": "Hesitating",
        "c3": "Changing Direction", "c4": "Walking",
    }
    heat_rows = []
    for fold in FOLDS:
        base = maximum[(maximum.fold == fold) & (maximum.condition == "baseline")].iloc[0]
        for target in range(1, 6):
            pert = maximum[(maximum.fold == fold) & (maximum.condition == f"ablate_l{target}")].iloc[0]
            for key, behavior in behavior_names.items():
                heat_rows.append(
                    {
                        "fold": fold,
                        "window": f"L{target}",
                        "target_raw_label": target,
                        "behavior": behavior,
                        "metric": "recall",
                        "baseline": float(base[key]),
                        "perturbed": float(pert[key]),
                        "delta_perturb_minus_baseline": float(pert[key] - base[key]),
                    }
                )
    pd.DataFrame(heat_rows).to_csv(source_dir / "fig_a_window_behavior_delta_recall.csv", index=False, encoding="utf-8-sig")

    mechanisms = {
        "Maximum": max_csv,
        "Attention shuffle": PROJECT_ROOT / "legacy_5s_50pct_attention_perturbation" / "02_attention_shuffle" / "outputs" / "condition_fold_metrics.csv",
        "Attention zero": PROJECT_ROOT / "legacy_5s_50pct_attention_perturbation" / "04_key_attention_zero" / "outputs" / "condition_fold_metrics.csv",
        "LSTM 50 ms": PROJECT_ROOT / "legacy_5s_50pct_lstm_perturbation" / "01_block_order_shuffle" / "outputs" / "block_50ms" / "condition_fold_metrics.csv",
        "LSTM 100 ms": PROJECT_ROOT / "legacy_5s_50pct_lstm_perturbation" / "01_block_order_shuffle" / "outputs" / "block_100ms" / "condition_fold_metrics.csv",
        "LSTM 200 ms": PROJECT_ROOT / "legacy_5s_50pct_lstm_perturbation" / "01_block_order_shuffle" / "outputs" / "block_200ms" / "condition_fold_metrics.csv",
    }
    contrast_rows = []
    for mechanism, path in mechanisms.items():
        data = pd.read_csv(path)
        for fold in FOLDS:
            base = data[(data.fold == fold) & (data.condition == "baseline")].iloc[0]
            for target in TARGETS:
                pert = data[(data.fold == fold) & (data.condition == f"ablate_l{target}")].iloc[0]
                stage2_base = float(np.mean([base.c1, base.c3, base.c4]))
                stage2_pert = float(np.mean([pert.c1, pert.c3, pert.c4]))
                target_key = {2: "stage2_macro_recall", 3: "c2", 5: "c4"}[target]
                baseline = stage2_base if target == 2 else float(base[target_key])
                perturbed = stage2_pert if target == 2 else float(pert[target_key])
                contrast_rows.append(
                    {
                        "mechanism": mechanism,
                        "fold": fold,
                        "target": target,
                        "target_label": TARGET_LABELS[target],
                        "outcome": OUTCOME_LABELS[target],
                        "baseline": baseline,
                        "perturbed": perturbed,
                        "delta_perturb_minus_baseline": perturbed - baseline,
                        "wrong_hole_delta": float(pert.c1 - base.c1),
                        "direction_delta": float(pert.c3 - base.c3),
                        "walking_delta": float(pert.c4 - base.c4),
                    }
                )
    pd.DataFrame(contrast_rows).to_csv(source_dir / "fig_c_mechanism_contrast_fold.csv", index=False, encoding="utf-8-sig")

    sig = pd.read_csv(SEGMENT_ROOT / "01_maximum_replacement" / "outputs" / "significant_10ms_timepoints_used.csv")
    sig.to_csv(source_dir / "significant_timepoints_used.csv", index=False, encoding="utf-8-sig")
    qc = json.loads((SEGMENT_ROOT / "01_maximum_replacement" / "outputs" / "dataset_mapping_qc.json").read_text(encoding="utf-8"))
    coverage = []
    for run_tag in ("run1", "run2"):
        for target in range(1, 6):
            coverage.append(
                {
                    "run": run_tag,
                    "target": target,
                    "target_label": f"L{target}",
                    "n_windows": qc[run_tag]["n_windows"],
                    "center_occurrences": qc[run_tag]["center_occurrences_across_all_windows"][str(target)],
                    "segment_bins": qc[run_tag]["segment_bins_across_all_windows"][str(target)],
                    "boundary_clipped_centers": qc[run_tag]["boundary_clipped_center_occurrences"][str(target)],
                    "cross_recording_windows": qc[run_tag]["cross_recording_windows"],
                    "time_discontinuous_windows": qc[run_tag]["time_discontinuous_windows"],
                }
            )
    pd.DataFrame(coverage).to_csv(source_dir / "mapping_coverage.csv", index=False, encoding="utf-8-sig")


def empirical_null_table(architecture_df: pd.DataFrame, random_df: pd.DataFrame) -> pd.DataFrame:
    output = []
    for target in TARGETS:
        metric = OUTCOME_KEYS[target]
        for fold in FOLDS:
            part = architecture_df[
                (architecture_df.architecture == "full") & (architecture_df.target == target) & (architecture_df.fold == fold)
            ]
            baseline = float(part[part.condition == "baseline"][metric].iloc[0])
            observed = float(part[part.condition == "maximum"][metric].iloc[0] - baseline)
            null = random_df[
                (random_df.architecture == "full") & (random_df.target == target) & (random_df.fold == fold)
            ].copy()
            null["delta_perturb_minus_baseline"] = null[metric] - baseline
            p = (1.0 + float((null.delta_perturb_minus_baseline <= observed).sum())) / (len(null) + 1.0)
            for _, row in null.iterrows():
                output.append(
                    {
                        "target": target,
                        "target_label": TARGET_LABELS[target],
                        "outcome": OUTCOME_LABELS[target],
                        "fold": fold,
                        "repeat": int(row["repeat"]),
                        "random_delta": float(row.delta_perturb_minus_baseline),
                        "observed_maximum_delta": observed,
                        "empirical_one_sided_p": p,
                    }
                )
    return pd.DataFrame(output)


def write_run_metadata(args, datasets: Dict[str, LegacySegmentDataset]) -> None:
    qa = RESULT_ROOT / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    metadata = {
        "python_environment": Path(sys.executable).parent.name,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "project_protocol": "legacy hierarchical 5 s windows, 50% overlap, saved three-fold splits",
        "model_selection_warning": "best epoch selected on heldout fold to remain legacy-compatible",
        "max_epochs": args.max_epochs,
        "min_epochs": args.min_epochs,
        "patience": args.patience,
        "attention_shuffle_repeats": args.shuffle_repeats,
        "random_repeats_full": args.random_repeats_full,
        "random_repeats_arch": args.random_repeats_arch,
        "dataset_windows": {run_tag: int(len(dataset)) for run_tag, dataset in datasets.items()},
        "input_channels": {run_tag: int(dataset.x.shape[1]) for run_tag, dataset in datasets.items()},
    }
    (qa / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--min-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--shuffle-repeats", type=int, default=20)
    parser.add_argument("--random-repeats-full", type=int, default=500)
    parser.add_argument("--random-repeats-arch", type=int, default=100)
    parser.add_argument("--force-training", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for subdir in ("checkpoints", "source_data", "tables", "figures", "qa"):
        (RESULT_ROOT / subdir).mkdir(parents=True, exist_ok=True)

    sig = read_significant_timepoints(XLSX_PATH)
    offsets = significant_offsets(sig)
    datasets = {
        "run1": LegacySegmentDataset(Config(), "run1", offsets, radius_bins=100),
        "run2": LegacySegmentDataset(Config3(), "run2", offsets, radius_bins=100),
    }
    configs = {"run1": Config(), "run2": Config3()}
    write_run_metadata(args, datasets)
    normalize_legacy_sources()

    if not args.skip_training:
        for architecture in ("no_conv", "no_lstm", "no_attention"):
            for run_tag in ("run1", "run2"):
                for fold in FOLDS:
                    train_one(
                        architecture,
                        run_tag,
                        fold,
                        datasets[run_tag],
                        configs[run_tag],
                        max_epochs=args.max_epochs,
                        min_epochs=args.min_epochs,
                        patience=args.patience,
                        force=args.force_training,
                    )

    missing = [
        str(checkpoint_path(architecture, run_tag, fold))
        for architecture in ARCH_FLAGS
        for run_tag in ("run1", "run2")
        for fold in FOLDS
        if not checkpoint_path(architecture, run_tag, fold).exists()
    ]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing))

    architecture_df, random_df, drop_keep_df = run_architecture_experiments(
        datasets,
        configs,
        random_repeats_full=args.random_repeats_full,
        random_repeats_arch=args.random_repeats_arch,
        shuffle_repeats=args.shuffle_repeats,
    )
    source = RESULT_ROOT / "source_data"
    tables = RESULT_ROOT / "tables"
    architecture_df.to_csv(source / "fig_b_architecture_perturbation_fold.csv", index=False, encoding="utf-8-sig")
    random_df.to_csv(source / "random_window_all_repeats.csv", index=False, encoding="utf-8-sig")
    drop_keep_df.to_csv(source / "fig_e_l3_drop_keep_fold.csv", index=False, encoding="utf-8-sig")

    detailed, interaction = build_interaction_tables(architecture_df)
    detailed.to_csv(tables / "table_a_fold_level_interactions.csv", index=False, encoding="utf-8-sig")
    build_condition_summary(detailed).to_csv(
        tables / "table_a_condition_summary.csv", index=False, encoding="utf-8-sig"
    )
    interaction.to_csv(tables / "table_a_interaction_summary.csv", index=False, encoding="utf-8-sig")
    empirical = empirical_null_table(architecture_df, random_df)
    empirical.to_csv(source / "fig_d_random_window_null.csv", index=False, encoding="utf-8-sig")
    empirical[
        ["target", "target_label", "outcome", "fold", "observed_maximum_delta", "empirical_one_sided_p"]
    ].drop_duplicates().to_csv(tables / "random_window_empirical_p.csv", index=False, encoding="utf-8-sig")
    print("Experiment suite complete.", flush=True)


if __name__ == "__main__":
    main()
