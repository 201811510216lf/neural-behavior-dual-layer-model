"""Four frozen-model Attention perturbations under the exact legacy protocol.

Protocol invariants:
  * 5 s = 500 bins per window
  * 50% overlap = 250-bin step
  * the stage label is constant across every retained window
  * the saved three-fold test indices and frozen run1/run2 checkpoints are reused
  * run1/run2 metrics are merged exactly like the legacy training report

The workbook's 10-ms p<0.05 points are mapped from each behavior onset back to
the original recording.  Affected windows are those containing at least one
mapped point for the target L1-L5 behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset


ANALYSIS_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ANALYSIS_ROOT.parents[1]
REPO_ROOT = SOURCE_ROOT.parent
MODEL_SOURCE_DIR = SOURCE_ROOT / "dual_layer_model"
SHARED_DIR = SOURCE_ROOT / "masking" / "segment_replacement"
sys.path.insert(0, str(MODEL_SOURCE_DIR))
sys.path.insert(0, str(SHARED_DIR))

from config import Config, Config3  # noqa: E402
from segment_replace_core import (  # noqa: E402
    LegacySegmentDataset,
    compute_stage_metrics,
    load_fold_models,
    load_saved_accuracy,
    load_test_indices,
    merge_training_style,
    read_significant_timepoints,
    significant_offsets,
)


SCHEME_DIRS = {
    "uniform": ANALYSIS_ROOT / "01_attention_uniform",
    "shuffle": ANALYSIS_ROOT / "02_attention_shuffle",
    "swap": ANALYSIS_ROOT / "03_key_nonkey_swap",
    "zero": ANALYSIS_ROOT / "04_key_attention_zero",
}
SCHEME_TITLES = {
    "uniform": "Attention weights uniformized",
    "shuffle": "Attention weights shuffled across time",
    "swap": "Critical/noncritical equal-length energy-matched exchange",
    "zero": "Critical-region Attention weights zeroed and renormalized",
}
METRIC_KEYS = ["accuracy", "recall", "f1", "c0", "c1", "c2", "c3", "c4"]
CLASS_NAMES = [
    "Hole Exploration",
    "Wrong Hole Exploration",
    "Hesitating",
    "Changing Direction",
    "Walking",
]


def mean_sd(values: Iterable[float]) -> Tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def mean_sd_text(values: Iterable[float]) -> str:
    mean, sd = mean_sd(values)
    return f"{mean:.2f} ± {sd:.2f}"


def temporal_states_and_attention(model, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the unchanged feature extractor/LSTM and return learned time weights."""
    if getattr(model, "ablate_conv", False):
        conv_out = model.input_projection(x)
    else:
        conv_out = model.conv(x)
    sequence = conv_out.permute(0, 2, 1)
    if getattr(model, "ablate_lstm", False):
        states = sequence
    else:
        states, _ = model.lstm(sequence)
    scores = model.time_attention(states).squeeze(-1)
    attention = torch.softmax(scores, dim=1)
    return states, attention


def logits_from_attention(model, states: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
    context = torch.bmm(attention.unsqueeze(1), states).squeeze(1)
    return model.fc(context)


def metric_probability_details(
    y_true: np.ndarray, probabilities: np.ndarray, logits: np.ndarray
) -> Dict[str, float]:
    row = np.arange(len(y_true))
    true_probability = probabilities[row, y_true]
    other = logits.copy()
    other[row, y_true] = -np.inf
    margin = logits[row, y_true] - np.max(other, axis=1)
    return {
        "true_probability": float(np.mean(true_probability) * 100.0),
        "logit_margin": float(np.mean(margin)),
    }


def find_energy_matched_destination(
    x: torch.Tensor,
    source: np.ndarray,
    full_critical_mask: np.ndarray,
) -> Tuple[np.ndarray, float, bool]:
    """Find a circular equal-length destination, prioritizing noncritical bins.

    The source is always a linear clipped +/-1-s interval.  A circular search
    guarantees a disjoint equal-length candidate even when the source lies in
    the middle of a 5-s window.  Candidate ranking is lexicographic:
    (critical-mask overlap, absolute signal-energy difference).
    """
    length = int(len(source))
    total = int(x.shape[-1])
    source_set = np.zeros(total, dtype=bool)
    source_set[source] = True
    energy_per_bin = x.detach().float().pow(2).mean(dim=0).cpu().numpy()
    source_energy = float(energy_per_bin[source].mean())
    offsets = np.arange(length, dtype=int)
    best_indices: Optional[np.ndarray] = None
    best_score: Optional[Tuple[float, float]] = None
    best_wrapped = False
    for start in range(total):
        candidate = (start + offsets) % total
        if source_set[candidate].any():
            continue
        overlap = float(full_critical_mask[candidate].mean())
        energy_gap = abs(float(energy_per_bin[candidate].mean()) - source_energy)
        score = (overlap, energy_gap)
        if best_score is None or score < best_score:
            best_indices = candidate
            best_score = score
            best_wrapped = bool(start + length > total)
    if best_indices is None or best_score is None:
        raise RuntimeError("No disjoint equal-length destination exists.")
    return best_indices, float(best_score[0]), best_wrapped


def build_swap_variants(
    x: torch.Tensor,
    segment_mask: torch.Tensor,
    center_flags: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[int], Dict[str, int | float]]:
    """One variant per mapped keypoint; variants from one window are averaged."""
    variants: List[torch.Tensor] = []
    owners: List[int] = []
    overlap_sum = 0.0
    wrapped = 0
    source_bins = 0
    for sample_index in range(x.shape[0]):
        centers = torch.nonzero(center_flags[sample_index], as_tuple=False).flatten().tolist()
        full_mask = segment_mask[sample_index].cpu().numpy().astype(bool)
        for center in centers:
            start = max(0, int(center) - 100)
            stop = min(int(x.shape[-1]), int(center) + 100)
            source = np.arange(start, stop, dtype=int)
            destination, overlap, did_wrap = find_energy_matched_destination(
                x[sample_index], source, full_mask
            )
            variant = x[sample_index].clone()
            source_values = variant[:, source].clone()
            destination_values = variant[:, destination].clone()
            variant[:, source] = destination_values
            variant[:, destination] = source_values
            variants.append(variant)
            owners.append(sample_index)
            overlap_sum += overlap
            wrapped += int(did_wrap)
            source_bins += int(len(source))
    return variants, owners, {
        "swap_variants": int(len(variants)),
        "swap_source_bins": int(source_bins),
        "swap_wrapped_destinations": int(wrapped),
        "swap_destination_critical_overlap_mean": float(overlap_sum / len(variants)) if variants else 0.0,
    }


@torch.no_grad()
def evaluate_stage(
    model,
    dataset: LegacySegmentDataset,
    indices: List[int],
    config,
    target_raw_label: Optional[int],
    scheme: Optional[str],
    shuffle_repeats: int,
    seed: int,
) -> Dict:
    loader = DataLoader(Subset(dataset, indices), batch_size=config.batch_size, shuffle=False)
    y_true_parts: List[np.ndarray] = []
    probability_parts: List[np.ndarray] = []
    logit_parts: List[np.ndarray] = []
    affected_windows = 0
    critical_time_bins = 0
    swap_totals = {
        "swap_variants": 0,
        "swap_source_bins": 0,
        "swap_wrapped_destinations": 0,
        "swap_destination_critical_overlap_weighted": 0.0,
    }
    generator = torch.Generator(device=config.device)
    generator.manual_seed(int(seed))

    for x, y, all_segment_masks, original_indices in loader:
        x = x.float()
        if target_raw_label is None:
            logits = model(x.to(config.device))
            probabilities = torch.softmax(logits, dim=1)
        else:
            label_index = int(target_raw_label) - 1
            segment_mask = all_segment_masks[:, label_index, :].bool()
            center_flags = torch.from_numpy(dataset.center_flags[original_indices.numpy(), label_index, :]).bool()
            affected = center_flags.any(dim=1)
            affected_windows += int(affected.sum().item())
            critical_time_bins += int(segment_mask.sum().item())

            if scheme == "swap":
                base_logits = model(x.to(config.device))
                base_probabilities = torch.softmax(base_logits, dim=1)
                variants, owners, stats = build_swap_variants(x, segment_mask, center_flags)
                probabilities = base_probabilities.clone()
                logits = base_logits.clone()
                if variants:
                    variant_tensor = torch.stack(variants)
                    variant_logits_parts = []
                    for start in range(0, len(variant_tensor), config.batch_size):
                        variant_logits_parts.append(model(variant_tensor[start:start + config.batch_size].to(config.device)))
                    variant_logits = torch.cat(variant_logits_parts, dim=0)
                    variant_probabilities = torch.softmax(variant_logits, dim=1)
                    owner_tensor = torch.as_tensor(owners, device=config.device, dtype=torch.long)
                    sums_p = torch.zeros_like(probabilities)
                    sums_l = torch.zeros_like(logits)
                    counts = torch.zeros((x.shape[0], 1), device=config.device)
                    sums_p.index_add_(0, owner_tensor, variant_probabilities)
                    sums_l.index_add_(0, owner_tensor, variant_logits)
                    counts.index_add_(0, owner_tensor, torch.ones((len(owners), 1), device=config.device))
                    has_variant = counts.squeeze(1) > 0
                    probabilities[has_variant] = sums_p[has_variant] / counts[has_variant]
                    logits[has_variant] = sums_l[has_variant] / counts[has_variant]
                swap_totals["swap_variants"] += int(stats["swap_variants"])
                swap_totals["swap_source_bins"] += int(stats["swap_source_bins"])
                swap_totals["swap_wrapped_destinations"] += int(stats["swap_wrapped_destinations"])
                swap_totals["swap_destination_critical_overlap_weighted"] += (
                    float(stats["swap_destination_critical_overlap_mean"]) * int(stats["swap_variants"])
                )
            else:
                states, original_attention = temporal_states_and_attention(model, x.to(config.device))
                affected_device = affected.to(config.device)
                mask_device = segment_mask.to(config.device)
                if scheme == "uniform":
                    uniform = torch.full_like(original_attention, 1.0 / original_attention.shape[1])
                    modified = torch.where(affected_device.unsqueeze(1), uniform, original_attention)
                    logits = logits_from_attention(model, states, modified)
                    probabilities = torch.softmax(logits, dim=1)
                elif scheme == "zero":
                    zeroed = original_attention.masked_fill(mask_device, 0.0)
                    denominator = zeroed.sum(dim=1, keepdim=True)
                    safe = zeroed / denominator.clamp_min(1e-12)
                    fallback = (~mask_device).float()
                    fallback = fallback / fallback.sum(dim=1, keepdim=True).clamp_min(1.0)
                    safe = torch.where(denominator > 1e-12, safe, fallback)
                    modified = torch.where(affected_device.unsqueeze(1), safe, original_attention)
                    logits = logits_from_attention(model, states, modified)
                    probabilities = torch.softmax(logits, dim=1)
                elif scheme == "shuffle":
                    probability_sum = torch.zeros(
                        (x.shape[0], config.num_classes), device=config.device, dtype=states.dtype
                    )
                    logit_sum = torch.zeros_like(probability_sum)
                    for _ in range(int(shuffle_repeats)):
                        permutation = torch.rand(
                            original_attention.shape,
                            generator=generator,
                            device=config.device,
                        ).argsort(dim=1)
                        shuffled = original_attention.gather(1, permutation)
                        modified = torch.where(affected_device.unsqueeze(1), shuffled, original_attention)
                        repeat_logits = logits_from_attention(model, states, modified)
                        probability_sum += torch.softmax(repeat_logits, dim=1)
                        logit_sum += repeat_logits
                    probabilities = probability_sum / float(shuffle_repeats)
                    logits = logit_sum / float(shuffle_repeats)
                else:
                    raise ValueError(f"Unsupported scheme: {scheme}")

        y_true_parts.append(y.numpy().astype(int))
        probability_parts.append(probabilities.detach().cpu().numpy())
        logit_parts.append(logits.detach().cpu().numpy())

    y_true = np.concatenate(y_true_parts)
    probabilities = np.concatenate(probability_parts)
    logits = np.concatenate(logit_parts)
    predictions = np.argmax(probabilities, axis=1)
    pack = compute_stage_metrics(y_true, predictions, probabilities, config.num_classes)
    pack.update(metric_probability_details(y_true, probabilities, logits))
    pack["affected_windows"] = int(affected_windows)
    pack["critical_time_bins"] = int(critical_time_bins)
    pack["shuffle_repeats"] = int(shuffle_repeats if scheme == "shuffle" else 0)
    pack["swap_variants"] = int(swap_totals["swap_variants"])
    pack["swap_source_bins"] = int(swap_totals["swap_source_bins"])
    pack["swap_wrapped_destinations"] = int(swap_totals["swap_wrapped_destinations"])
    denominator = max(1, int(swap_totals["swap_variants"]))
    pack["swap_destination_critical_overlap_mean"] = float(
        swap_totals["swap_destination_critical_overlap_weighted"] / denominator
    )
    return pack


def merge_with_probability_details(run1_pack: Dict, run2_pack: Dict) -> Dict:
    merged = merge_training_style(run1_pack, run2_pack)
    merged["true_probability"] = (
        float(run1_pack["true_probability"]) + float(run2_pack["true_probability"])
    ) / 2.0
    merged["logit_margin"] = (
        float(run1_pack["logit_margin"]) + float(run2_pack["logit_margin"])
    ) / 2.0
    return merged


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
                    for key in METRIC_KEYS + ["true_probability", "logit_margin"]
                },
            }
        )
    return output


def build_summary_tables(output_dir: Path, aggregate: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
                **{
                    CLASS_NAMES[index]: mean_sd_text(row[f"fold_c{index}"])
                    for index in range(5)
                },
            }
            for row in aggregate
        ]
    )
    overall.to_csv(output_dir / "table4_overall_mean_sd.csv", index=False, encoding="utf-8-sig")
    classes.to_csv(output_dir / "table5_class_accuracy_mean_sd.csv", index=False, encoding="utf-8-sig")

    baseline = aggregate[0]
    deltas = []
    for row in aggregate[1:]:
        item = {"condition": row["display_name"], "target_raw_label": row["target_raw_label"]}
        for key in METRIC_KEYS + ["true_probability", "logit_margin"]:
            drop = np.asarray(baseline[f"fold_{key}"]) - np.asarray(row[f"fold_{key}"])
            item[f"drop_{key}_mean"] = float(drop.mean())
            item[f"drop_{key}_sd"] = float(drop.std(ddof=1))
        deltas.append(item)
    pd.DataFrame(deltas).to_csv(output_dir / "perturbation_delta_vs_baseline.csv", index=False, encoding="utf-8-sig")
    return overall, classes


def write_outputs(
    scheme: str,
    sig: pd.DataFrame,
    ds1: LegacySegmentDataset,
    ds2: LegacySegmentDataset,
    fold_rows: List[Dict],
    aggregate: List[Dict],
    args,
) -> None:
    output_dir = SCHEME_DIRS[scheme] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(output_dir / "condition_fold_metrics.csv", index=False, encoding="utf-8-sig")
    sig.to_csv(output_dir / "significant_10ms_timepoints_used.csv", index=False, encoding="utf-8-sig")
    (output_dir / "attention_perturbation_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "dataset_mapping_qc.json").write_text(
        json.dumps({"run1": ds1.mapping_stats, "run2": ds2.mapping_stats}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    config = {
        "python": sys.executable,
        "scheme": scheme,
        "scheme_title": SCHEME_TITLES[scheme],
        "model_dir": str(args.model_dir),
        "xlsx_path": str(args.xlsx_path),
        "window_bins": 500,
        "window_seconds": 5.0,
        "step_bins": 250,
        "overlap_fraction": 0.5,
        "constant_stage_label_required": True,
        "critical_segment": "[significant center - 1 s, significant center + 1 s), clipped at window edge",
        "uniform_scope": "all 500 attention weights, only in affected windows",
        "shuffle_scope": "all 500 attention weights, only in affected windows",
        "shuffle_repeats": int(args.shuffle_repeats),
        "shuffle_seed": int(args.seed),
        "swap_scope": "one +/-1-s variant per mapped point; variants within a window averaged",
        "swap_candidate": "disjoint circular equal-length candidate; minimize critical overlap then energy gap",
        "zero_scope": "union of all +/-1-s critical masks; remaining weights renormalized",
        "retraining": False,
        "folds": args.folds,
    }
    (output_dir / "analysis_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    build_summary_tables(output_dir, aggregate)
    baseline = aggregate[0]
    lines = [
        f"# {SCHEME_TITLES[scheme]}",
        "",
        "Frozen legacy protocol: 5 s, 50% overlap, constant stage label, saved three-fold splits, no retraining.",
        "",
        "## Three-fold summary",
        "",
    ]
    for row in aggregate:
        lines.append(
            f"- {row['display_name']}: Acc {mean_sd_text(row['fold_accuracy'])}; "
            f"Recall {mean_sd_text(row['fold_recall'])}; F1 {mean_sd_text(row['fold_f1'])}."
        )
    lines += ["", "## Drop versus baseline", ""]
    for row in aggregate[1:]:
        drops = {
            key: np.asarray(baseline[f"fold_{key}"]) - np.asarray(row[f"fold_{key}"])
            for key in ["accuracy", "recall", "f1", "true_probability", "logit_margin"]
        }
        lines.append(
            f"- {row['display_name']}: Acc {drops['accuracy'].mean():.2f} pp; "
            f"Recall {drops['recall'].mean():.2f} pp; F1 {drops['f1'].mean():.2f} pp; "
            f"true-class probability {drops['true_probability'].mean():.2f} pp; "
            f"logit margin {drops['logit_margin'].mean():.4f}."
        )
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_schemes(args, schemes: Sequence[str]) -> None:
    invalid = [scheme for scheme in schemes if scheme not in SCHEME_DIRS]
    if invalid:
        raise ValueError(f"Unsupported schemes: {invalid}")
    sig = read_significant_timepoints(args.xlsx_path)
    offsets = significant_offsets(sig)
    ds1 = LegacySegmentDataset(Config(), "run1", offsets, radius_bins=100)
    ds2 = LegacySegmentDataset(Config3(), "run2", offsets, radius_bins=100)
    if ds1.x.shape[2] != 500 or ds2.x.shape[2] != 500:
        raise AssertionError("Legacy input must remain 500 bins.")
    if Config.step_size != 250 or Config3.step_size != 250:
        raise AssertionError("Legacy step must remain 250 bins (50% overlap).")

    rows_by_scheme = {scheme: [] for scheme in schemes}
    for fold_index, fold_tag in enumerate(args.folds):
        print(f"\nLoading {fold_tag} ...", flush=True)
        model1, model2 = load_fold_models(args.model_dir, fold_tag, ds1.x.shape[1])
        indices1 = load_test_indices(args.model_dir, "run1", fold_tag)
        indices2 = load_test_indices(args.model_dir, "run2", fold_tag)
        baseline1 = evaluate_stage(model1, ds1, indices1, Config(), None, None, args.shuffle_repeats, args.seed)
        baseline2 = evaluate_stage(model2, ds2, indices2, Config3(), None, None, args.shuffle_repeats, args.seed)
        saved1 = load_saved_accuracy(args.model_dir, "run1", fold_tag)
        saved2 = load_saved_accuracy(args.model_dir, "run2", fold_tag)
        if saved1 is not None and not np.isclose(baseline1["accuracy"], saved1, atol=1e-8):
            raise AssertionError(f"run1 {fold_tag} baseline mismatch: {baseline1['accuracy']} vs {saved1}")
        if saved2 is not None and not np.isclose(baseline2["accuracy"], saved2, atol=1e-8):
            raise AssertionError(f"run2 {fold_tag} baseline mismatch: {baseline2['accuracy']} vs {saved2}")
        baseline = merge_with_probability_details(baseline1, baseline2)

        for scheme in schemes:
            # The seed must not depend on whether a scheme is run alone or as
            # part of run_all.  Bind it to the canonical four-scheme order.
            scheme_index = list(SCHEME_DIRS).index(scheme)
            rows_by_scheme[scheme].append(
                {
                    "condition": "baseline",
                    "display_name": "Our Proposed",
                    "target_raw_label": None,
                    "fold": fold_tag,
                    **baseline,
                    "run1_accuracy": baseline1["accuracy"],
                    "run2_accuracy": baseline2["accuracy"],
                    "run1_affected_windows": 0,
                    "run2_affected_windows": 0,
                }
            )
            for target in range(1, 6):
                condition_seed = int(args.seed + fold_index * 10000 + scheme_index * 1000 + target * 100)
                pack1 = evaluate_stage(
                    model1, ds1, indices1, Config(), target, scheme, args.shuffle_repeats, condition_seed + 1
                )
                pack2 = evaluate_stage(
                    model2, ds2, indices2, Config3(), target, scheme, args.shuffle_repeats, condition_seed + 2
                )
                merged = merge_with_probability_details(pack1, pack2)
                rows_by_scheme[scheme].append(
                    {
                        "condition": f"ablate_l{target}",
                        "display_name": f"Ablate L{target}",
                        "target_raw_label": target,
                        "fold": fold_tag,
                        **merged,
                        **{
                            f"run1_{key}": pack1[key]
                            for key in [
                                "accuracy", "affected_windows", "critical_time_bins", "shuffle_repeats",
                                "swap_variants", "swap_source_bins", "swap_wrapped_destinations",
                                "swap_destination_critical_overlap_mean",
                            ]
                        },
                        **{
                            f"run2_{key}": pack2[key]
                            for key in [
                                "accuracy", "affected_windows", "critical_time_bins", "shuffle_repeats",
                                "swap_variants", "swap_source_bins", "swap_wrapped_destinations",
                                "swap_destination_critical_overlap_mean",
                            ]
                        },
                    }
                )
                print(f"  {scheme:7s} L{target} complete", flush=True)
        del model1, model2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for scheme in schemes:
        aggregate = aggregate_rows(rows_by_scheme[scheme])
        write_outputs(scheme, sig, ds1, ds2, rows_by_scheme[scheme], aggregate, args)
        table = pd.read_csv(SCHEME_DIRS[scheme] / "outputs" / "table4_overall_mean_sd.csv")
        print(f"\n[{scheme}] {SCHEME_TITLES[scheme]}\n{table.to_string(index=False)}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument(
        "--xlsx-path",
        type=Path,
        default=REPO_ROOT / "data" / "significant_timepoints.xlsx",
    )
    parser.add_argument("--folds", nargs="+", default=["fold01", "fold02", "fold03"])
    parser.add_argument("--shuffle-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser


def main(selected_schemes: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args()
    schemes = list(selected_schemes) if selected_schemes is not None else list(SCHEME_DIRS)
    run_schemes(args, schemes)


if __name__ == "__main__":
    main()
