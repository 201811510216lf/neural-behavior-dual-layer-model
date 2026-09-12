"""LSTM-oriented perturbations under the exact frozen legacy protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset


ANALYSIS_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ANALYSIS_ROOT.parents[1]
REPO_ROOT = SOURCE_ROOT.parent
MODEL_SOURCE_DIR = SOURCE_ROOT / "dual_layer_model"
MASKING_ROOT = SOURCE_ROOT / "masking"
SHARED_DIR = MASKING_ROOT / "segment_replacement"
ATTENTION_DIR = MASKING_ROOT / "attention"
CONV_DIR = MASKING_ROOT / "conv"
for directory in [MODEL_SOURCE_DIR, SHARED_DIR, ATTENTION_DIR, CONV_DIR]:
    sys.path.insert(0, str(directory))

from config import Config, Config3  # noqa: E402
from attention_perturbation_core import build_swap_variants  # noqa: E402
from conv_perturbation_core import (  # noqa: E402
    CLASS_NAMES,
    METRIC_KEYS,
    aggregate_rows,
    build_tables,
    mean_sd_text,
    merge_details,
    probability_details,
)
from segment_replace_core import (  # noqa: E402
    LegacySegmentDataset,
    compute_stage_metrics,
    load_fold_models,
    load_saved_accuracy,
    load_test_indices,
    read_significant_timepoints,
    significant_offsets,
)


SCHEME_DIRS = {
    "block": ANALYSIS_ROOT / "01_block_order_shuffle",
    "swap": ANALYSIS_ROOT / "02_distant_segment_exchange",
    "reset": ANALYSIS_ROOT / "03_hidden_state_reset",
    "history": ANALYSIS_ROOT / "04_effective_history_truncation",
}
SCHEME_TITLES = {
    "block": "Critical-region block-order shuffle",
    "swap": "Distant equal-length energy-matched segment exchange",
    "reset": "LSTM hidden/cell-state reset",
    "history": "Fixed-window effective history truncation",
}
BLOCK_LENGTHS_MS = [50, 100, 200, 400]
RESET_POSITIONS = ["start", "middle", "end"]
HISTORY_LENGTHS_MS = [50, 100, 200, 400, 800]


def connected_components(mask: torch.Tensor) -> List[Tuple[int, int]]:
    indices = torch.nonzero(mask, as_tuple=False).flatten().cpu().numpy()
    if len(indices) == 0:
        return []
    split_points = np.flatnonzero(np.diff(indices) > 1) + 1
    groups = np.split(indices, split_points)
    return [(int(group[0]), int(group[-1]) + 1) for group in groups]


def shuffle_blocks(
    x: torch.Tensor,
    components_by_sample: List[List[Tuple[int, int]]],
    block_bins: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Shuffle block order inside each critical component; preserve within-block order."""
    output = x.clone()
    for sample_index, components in enumerate(components_by_sample):
        for start, stop in components:
            blocks = [
                x[sample_index, :, block_start:min(block_start + block_bins, stop)]
                for block_start in range(start, stop, block_bins)
            ]
            if len(blocks) < 2:
                continue
            permutation = torch.randperm(len(blocks), generator=generator).tolist()
            output[sample_index, :, start:stop] = torch.cat(
                [blocks[index] for index in permutation], dim=-1
            )
    return output


def reset_positions_from_components(
    components_by_sample: List[List[Tuple[int, int]]],
    mode: str,
    history_bins: Optional[int] = None,
    total_bins: int = 500,
) -> List[List[int]]:
    positions_by_sample: List[List[int]] = []
    for components in components_by_sample:
        positions: List[int] = []
        for start, stop in components:
            if mode == "start":
                position = start
            elif mode == "middle":
                position = (start + stop) // 2
            elif mode == "end":
                position = stop
            elif mode == "history":
                if history_bins is None:
                    raise ValueError("history_bins is required for history mode")
                key_time = (start + stop) // 2
                position = max(0, key_time - int(history_bins))
            else:
                raise ValueError(f"Unsupported reset mode: {mode}")
            if 0 < int(position) < total_bins:
                positions.append(int(position))
        positions_by_sample.append(sorted(set(positions)))
    return positions_by_sample


@torch.no_grad()
def logits_with_state_resets(
    model,
    x: torch.Tensor,
    positions_by_sample: List[List[int]],
) -> torch.Tensor:
    """Run the original Conv/LSTM/Attention, zeroing h and c at selected times."""
    conv_out = model.conv(x)
    sequence = conv_out.permute(0, 2, 1)
    recurrent_outputs: List[torch.Tensor] = []
    total_bins = int(sequence.shape[1])
    for sample_index, reset_positions in enumerate(positions_by_sample):
        sample = sequence[sample_index:sample_index + 1]
        cursor = 0
        state = None
        parts: List[torch.Tensor] = []
        for position in sorted(set(p for p in reset_positions if 0 < p < total_bins)):
            if position > cursor:
                part, state = model.lstm(sample[:, cursor:position, :], state)
                parts.append(part)
            # Passing None to the next chunk is equivalent to zero h and c.
            state = None
            cursor = position
        if cursor < total_bins:
            part, state = model.lstm(sample[:, cursor:, :], state)
            parts.append(part)
        recurrent_outputs.append(torch.cat(parts, dim=1))
    lstm_out = torch.cat(recurrent_outputs, dim=0)
    attention = torch.softmax(model.time_attention(lstm_out).squeeze(-1), dim=1)
    context = torch.bmm(attention.unsqueeze(1), lstm_out).squeeze(1)
    return model.fc(context)


@torch.no_grad()
def logits_with_effective_history(
    model,
    x: torch.Tensor,
    components_by_sample: List[List[Tuple[int, int]]],
    history_bins: int,
) -> Tuple[torch.Tensor, int]:
    """At each critical midpoint, rebuild h/c from only the preceding L bins.

    Outputs before the critical midpoint remain on their current trajectory.
    The recurrent state used from the midpoint onward is recomputed from the
    raw Conv sequence [midpoint-L, midpoint) starting from zero state.
    """
    conv_out = model.conv(x)
    sequence = conv_out.permute(0, 2, 1)
    recurrent_outputs: List[torch.Tensor] = []
    event_count = 0
    total_bins = int(sequence.shape[1])
    for sample_index, components in enumerate(components_by_sample):
        sample = sequence[sample_index:sample_index + 1]
        key_times = sorted(
            set((start + stop) // 2 for start, stop in components if 0 < (start + stop) // 2 < total_bins)
        )
        cursor = 0
        state = None
        parts: List[torch.Tensor] = []
        for key_time in key_times:
            if key_time > cursor:
                part, state = model.lstm(sample[:, cursor:key_time, :], state)
                parts.append(part)
            history_start = max(0, key_time - int(history_bins))
            _, state = model.lstm(sample[:, history_start:key_time, :], None)
            event_count += 1
            cursor = key_time
        if cursor < total_bins:
            part, state = model.lstm(sample[:, cursor:, :], state)
            parts.append(part)
        recurrent_outputs.append(torch.cat(parts, dim=1))
    lstm_out = torch.cat(recurrent_outputs, dim=0)
    attention = torch.softmax(model.time_attention(lstm_out).squeeze(-1), dim=1)
    context = torch.bmm(attention.unsqueeze(1), lstm_out).squeeze(1)
    return model.fc(context), event_count


def variant_definitions(schemes: Sequence[str]) -> List[Dict]:
    variants: List[Dict] = []
    for scheme in schemes:
        if scheme == "block":
            for milliseconds in BLOCK_LENGTHS_MS:
                variants.append(
                    {
                        "key": f"block_{milliseconds}ms",
                        "scheme": "block",
                        "block_ms": milliseconds,
                        "block_bins": milliseconds // 10,
                    }
                )
        elif scheme == "swap":
            variants.append({"key": "swap", "scheme": "swap"})
        elif scheme == "reset":
            for position in RESET_POSITIONS:
                variants.append(
                    {"key": f"reset_{position}", "scheme": "reset", "reset_position": position}
                )
        elif scheme == "history":
            for milliseconds in HISTORY_LENGTHS_MS:
                variants.append(
                    {
                        "key": f"history_{milliseconds}ms",
                        "scheme": "history",
                        "history_ms": milliseconds,
                        "history_bins": milliseconds // 10,
                    }
                )
        else:
            raise ValueError(f"Unsupported scheme: {scheme}")
    return variants


def output_directory(variant: Dict) -> Path:
    scheme = variant["scheme"]
    if scheme == "block":
        return SCHEME_DIRS[scheme] / "outputs" / f"block_{variant['block_ms']}ms"
    if scheme == "reset":
        return SCHEME_DIRS[scheme] / "outputs" / str(variant["reset_position"])
    if scheme == "history":
        return SCHEME_DIRS[scheme] / "outputs" / f"history_{variant['history_ms']}ms"
    return SCHEME_DIRS[scheme] / "outputs"


@torch.no_grad()
def evaluate_stage(
    model,
    dataset: LegacySegmentDataset,
    indices: List[int],
    config,
    target_raw_label: Optional[int],
    variant: Optional[Dict],
    random_repeats: int,
    seed: int,
) -> Dict:
    loader = DataLoader(Subset(dataset, indices), batch_size=config.batch_size, shuffle=False)
    y_parts: List[np.ndarray] = []
    probability_parts: List[np.ndarray] = []
    logit_parts: List[np.ndarray] = []
    affected_windows = 0
    critical_time_bins = 0
    reset_events = 0
    swap_variants = 0
    swap_overlap_weighted = 0.0
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    for x, y, all_masks, original_indices in loader:
        x = x.float()
        base_logits = model(x.to(config.device))
        logits = base_logits.clone()
        probabilities = torch.softmax(base_logits, dim=1)

        if target_raw_label is not None and variant is not None:
            label_index = int(target_raw_label) - 1
            mask = all_masks[:, label_index, :].bool()
            center_flags = torch.from_numpy(
                dataset.center_flags[original_indices.numpy(), label_index, :]
            ).bool()
            affected = center_flags.any(dim=1)
            affected_windows += int(affected.sum().item())
            critical_time_bins += int(mask.sum().item())
            if affected.any():
                affected_x = x[affected]
                affected_mask = mask[affected]
                affected_centers = center_flags[affected]
                components = [connected_components(row) for row in affected_mask]
                scheme = str(variant["scheme"])

                if scheme == "block":
                    probability_sum = torch.zeros(
                        (affected_x.shape[0], config.num_classes), device=config.device
                    )
                    logit_sum = torch.zeros_like(probability_sum)
                    for _ in range(int(random_repeats)):
                        changed = shuffle_blocks(
                            affected_x, components, int(variant["block_bins"]), generator
                        )
                        repeat_logits = model(changed.to(config.device))
                        logit_sum += repeat_logits
                        probability_sum += torch.softmax(repeat_logits, dim=1)
                    affected_logits = logit_sum / float(random_repeats)
                    affected_probabilities = probability_sum / float(random_repeats)
                elif scheme == "swap":
                    swap_x, owners, stats = build_swap_variants(
                        affected_x, affected_mask, affected_centers
                    )
                    if swap_x:
                        stacked = torch.stack(swap_x)
                        variant_logits_parts = []
                        for start in range(0, len(stacked), config.batch_size):
                            variant_logits_parts.append(
                                model(stacked[start:start + config.batch_size].to(config.device))
                            )
                        variant_logits = torch.cat(variant_logits_parts, dim=0)
                        variant_probabilities = torch.softmax(variant_logits, dim=1)
                        owner_tensor = torch.as_tensor(owners, dtype=torch.long, device=config.device)
                        sums_logits = torch.zeros(
                            (affected_x.shape[0], config.num_classes), device=config.device
                        )
                        sums_probabilities = torch.zeros_like(sums_logits)
                        counts = torch.zeros((affected_x.shape[0], 1), device=config.device)
                        sums_logits.index_add_(0, owner_tensor, variant_logits)
                        sums_probabilities.index_add_(0, owner_tensor, variant_probabilities)
                        counts.index_add_(
                            0, owner_tensor, torch.ones((len(owners), 1), device=config.device)
                        )
                        fallback_logits = base_logits[affected.to(config.device)]
                        fallback_probabilities = torch.softmax(fallback_logits, dim=1)
                        valid = counts.squeeze(1) > 0
                        affected_logits = fallback_logits.clone()
                        affected_probabilities = fallback_probabilities.clone()
                        affected_logits[valid] = sums_logits[valid] / counts[valid]
                        affected_probabilities[valid] = sums_probabilities[valid] / counts[valid]
                    else:
                        affected_logits = base_logits[affected.to(config.device)]
                        affected_probabilities = torch.softmax(affected_logits, dim=1)
                    swap_variants += int(stats["swap_variants"])
                    swap_overlap_weighted += float(
                        stats["swap_destination_critical_overlap_mean"]
                    ) * int(stats["swap_variants"])
                elif scheme == "reset":
                    positions = reset_positions_from_components(
                        components, str(variant["reset_position"]), total_bins=500
                    )
                    reset_events += sum(len(item) for item in positions)
                    affected_logits = logits_with_state_resets(
                        model, affected_x.to(config.device), positions
                    )
                    affected_probabilities = torch.softmax(affected_logits, dim=1)
                elif scheme == "history":
                    affected_logits, events = logits_with_effective_history(
                        model,
                        affected_x.to(config.device),
                        components,
                        history_bins=int(variant["history_bins"]),
                    )
                    reset_events += int(events)
                    affected_probabilities = torch.softmax(affected_logits, dim=1)
                else:
                    raise ValueError(f"Unsupported scheme: {scheme}")

                affected_device = affected.to(config.device)
                logits[affected_device] = affected_logits
                probabilities[affected_device] = affected_probabilities

        y_parts.append(y.numpy().astype(int))
        probability_parts.append(probabilities.detach().cpu().numpy())
        logit_parts.append(logits.detach().cpu().numpy())

    y_true = np.concatenate(y_parts)
    probabilities_array = np.concatenate(probability_parts)
    logits_array = np.concatenate(logit_parts)
    predictions = np.argmax(probabilities_array, axis=1)
    pack = compute_stage_metrics(y_true, predictions, probabilities_array, config.num_classes)
    pack.update(probability_details(y_true, probabilities_array, logits_array))
    pack["confusion_matrix"] = confusion_matrix(
        y_true, predictions, labels=list(range(config.num_classes))
    ).astype(int).tolist()
    pack["affected_windows"] = int(affected_windows)
    pack["critical_time_bins"] = int(critical_time_bins)
    pack["random_repeats"] = int(
        random_repeats if variant is not None and variant["scheme"] == "block" else 0
    )
    pack["reset_events"] = int(reset_events)
    pack["swap_variants"] = int(swap_variants)
    pack["swap_destination_critical_overlap_mean"] = float(
        swap_overlap_weighted / max(1, swap_variants)
    )
    return pack


def write_outputs(
    variant: Dict,
    sig: pd.DataFrame,
    ds1: LegacySegmentDataset,
    ds2: LegacySegmentDataset,
    fold_rows: List[Dict],
    aggregate: List[Dict],
    args,
) -> None:
    out = output_directory(variant)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(out / "condition_fold_metrics.csv", index=False, encoding="utf-8-sig")
    sig.to_csv(out / "significant_10ms_timepoints_used.csv", index=False, encoding="utf-8-sig")
    (out / "lstm_perturbation_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "dataset_mapping_qc.json").write_text(
        json.dumps({"run1": ds1.mapping_stats, "run2": ds2.mapping_stats}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    config = {
        "python": sys.executable,
        "variant": variant,
        "scheme_title": SCHEME_TITLES[str(variant["scheme"])],
        "random_repeats": int(args.random_repeats),
        "seed": int(args.seed),
        "window_bins": 500,
        "window_seconds": 5.0,
        "step_bins": 250,
        "overlap_fraction": 0.5,
        "constant_stage_label_required": True,
        "critical_region": "union of [significant center-1 s, significant center+1 s), clipped at window edge",
        "conv_receptive_field_bins": 3,
        "conv_receptive_field_ms": 30,
        "history_implementation": (
            "at the critical midpoint, rebuild h/c from Conv outputs in [midpoint-L, midpoint) from zero state; "
            "pre-midpoint LSTM outputs, 500-bin input, and Attention remain unchanged"
            if variant["scheme"] == "history" else None
        ),
        "model_dir": str(args.model_dir),
        "xlsx_path": str(args.xlsx_path),
        "retraining": False,
        "folds": args.folds,
    }
    (out / "analysis_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    build_tables(out, aggregate)
    baseline = aggregate[0]
    lines = [
        f"# {SCHEME_TITLES[str(variant['scheme'])]} — {variant['key']}",
        "",
        "Frozen legacy protocol: 5 s, 50% overlap, constant stage labels, saved three-fold splits, no retraining.",
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
            f"true probability {drops['true_probability'].mean():.2f} pp; "
            f"logit margin {drops['logit_margin'].mean():.4f}."
        )
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_schemes(args, schemes: Sequence[str]) -> None:
    invalid = [scheme for scheme in schemes if scheme not in SCHEME_DIRS]
    if invalid:
        raise ValueError(f"Unsupported schemes: {invalid}")
    variants = variant_definitions(schemes)
    canonical_variants = variant_definitions(list(SCHEME_DIRS))
    canonical_index = {variant["key"]: index for index, variant in enumerate(canonical_variants)}
    sig = read_significant_timepoints(args.xlsx_path)
    offsets = significant_offsets(sig)
    ds1 = LegacySegmentDataset(Config(), "run1", offsets, radius_bins=100)
    ds2 = LegacySegmentDataset(Config3(), "run2", offsets, radius_bins=100)
    if ds1.x.shape[2] != 500 or ds2.x.shape[2] != 500:
        raise AssertionError("Legacy input must remain 500 bins.")
    if Config.step_size != 250 or Config3.step_size != 250:
        raise AssertionError("Legacy step must remain 250 bins (50% overlap).")

    rows_by_variant = {variant["key"]: [] for variant in variants}
    for fold_index, fold_tag in enumerate(args.folds):
        print(f"\nLoading {fold_tag} ...", flush=True)
        model1, model2 = load_fold_models(args.model_dir, fold_tag, ds1.x.shape[1])
        indices1 = load_test_indices(args.model_dir, "run1", fold_tag)
        indices2 = load_test_indices(args.model_dir, "run2", fold_tag)
        baseline1 = evaluate_stage(model1, ds1, indices1, Config(), None, None, 1, args.seed)
        baseline2 = evaluate_stage(model2, ds2, indices2, Config3(), None, None, 1, args.seed)
        saved1 = load_saved_accuracy(args.model_dir, "run1", fold_tag)
        saved2 = load_saved_accuracy(args.model_dir, "run2", fold_tag)
        if saved1 is not None and not np.isclose(baseline1["accuracy"], saved1, atol=1e-8):
            raise AssertionError(f"run1 {fold_tag} baseline mismatch: {baseline1['accuracy']} vs {saved1}")
        if saved2 is not None and not np.isclose(baseline2["accuracy"], saved2, atol=1e-8):
            raise AssertionError(f"run2 {fold_tag} baseline mismatch: {baseline2['accuracy']} vs {saved2}")
        baseline = merge_details(baseline1, baseline2)

        for variant in variants:
            rows = rows_by_variant[variant["key"]]
            rows.append(
                {
                    "condition": "baseline", "display_name": "Our Proposed", "target_raw_label": None,
                    "fold": fold_tag, **baseline,
                    "run1_accuracy": baseline1["accuracy"], "run2_accuracy": baseline2["accuracy"],
                    "run1_affected_windows": 0, "run2_affected_windows": 0,
                    "run1_confusion_matrix": json.dumps(baseline1["confusion_matrix"]),
                    "run2_confusion_matrix": json.dumps(baseline2["confusion_matrix"]),
                }
            )
            for target in range(1, 6):
                seed_base = int(
                    args.seed + fold_index * 100000 + canonical_index[variant["key"]] * 10000 + target * 100
                )
                pack1 = evaluate_stage(
                    model1, ds1, indices1, Config(), target, variant,
                    args.random_repeats, seed_base + 1,
                )
                pack2 = evaluate_stage(
                    model2, ds2, indices2, Config3(), target, variant,
                    args.random_repeats, seed_base + 2,
                )
                merged = merge_details(pack1, pack2)
                rows.append(
                    {
                        "condition": f"ablate_l{target}", "display_name": f"Ablate L{target}",
                        "target_raw_label": target, "fold": fold_tag, **merged,
                        "run1_accuracy": pack1["accuracy"], "run2_accuracy": pack2["accuracy"],
                        "run1_affected_windows": pack1["affected_windows"],
                        "run2_affected_windows": pack2["affected_windows"],
                        "run1_critical_time_bins": pack1["critical_time_bins"],
                        "run2_critical_time_bins": pack2["critical_time_bins"],
                        "run1_random_repeats": pack1["random_repeats"],
                        "run2_random_repeats": pack2["random_repeats"],
                        "run1_reset_events": pack1["reset_events"],
                        "run2_reset_events": pack2["reset_events"],
                        "run1_swap_variants": pack1["swap_variants"],
                        "run2_swap_variants": pack2["swap_variants"],
                        "run1_confusion_matrix": json.dumps(pack1["confusion_matrix"]),
                        "run2_confusion_matrix": json.dumps(pack2["confusion_matrix"]),
                    }
                )
                print(f"  {variant['key']:16s} L{target} complete", flush=True)
        del model1, model2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for variant in variants:
        aggregate = aggregate_rows(rows_by_variant[variant["key"]])
        write_outputs(variant, sig, ds1, ds2, rows_by_variant[variant["key"]], aggregate, args)
        table = pd.read_csv(output_directory(variant) / "table4_overall_mean_sd.csv")
        print(f"\n[{variant['key']}]\n{table.to_string(index=False)}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument(
        "--xlsx-path", type=Path,
        default=REPO_ROOT / "data" / "significant_timepoints.xlsx",
    )
    parser.add_argument("--folds", nargs="+", default=["fold01", "fold02", "fold03"])
    parser.add_argument("--random-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser


def main(selected_schemes: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args()
    schemes = list(selected_schemes) if selected_schemes is not None else list(SCHEME_DIRS)
    run_schemes(args, schemes)


if __name__ == "__main__":
    main()
