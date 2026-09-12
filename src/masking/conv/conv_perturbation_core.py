"""Conv-oriented input perturbations under the exact frozen legacy protocol.

The experiment preserves 5-s windows (500 x 10-ms bins), 50% overlap,
stage-label-constant selection, the saved three-fold test indices, frozen
run1/run2 checkpoints, and the original two-level metric merge.
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
from sklearn.metrics import confusion_matrix
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
    "phase": ANALYSIS_ROOT / "01_local_phase_randomization",
    "noise": ANALYSIS_ROOT / "02_multilevel_colored_noise",
    "band": ANALYSIS_ROOT / "03_high_band_ablation",
    "mean": ANALYSIS_ROOT / "04_mean_replacement_baseline",
}
SCHEME_TITLES = {
    "phase": "Local phase randomization",
    "noise": "Local spectrum-matched colored-noise injection",
    "band": "Local high-frequency-band ablation (15-50 Hz)",
    "mean": "Local per-neuron 5-s mean replacement",
}
NOISE_STRENGTHS = [0.25, 0.5, 1.0, 2.0]
CLASS_NAMES = [
    "Hole Exploration",
    "Wrong Hole Exploration",
    "Hesitating",
    "Changing Direction",
    "Walking",
]
METRIC_KEYS = ["accuracy", "recall", "f1", "c0", "c1", "c2", "c3", "c4"]


def mean_sd(values: Iterable[float]) -> Tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def mean_sd_text(values: Iterable[float]) -> str:
    mean, sd = mean_sd(values)
    return f"{mean:.2f} ± {sd:.2f}"


def connected_components(mask: torch.Tensor) -> List[Tuple[int, int]]:
    indices = torch.nonzero(mask, as_tuple=False).flatten().cpu().numpy()
    if len(indices) == 0:
        return []
    split_points = np.flatnonzero(np.diff(indices) > 1) + 1
    groups = np.split(indices, split_points)
    return [(int(group[0]), int(group[-1]) + 1) for group in groups]


def phase_randomize_segment(segment: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Preserve each channel's rFFT magnitude and randomize non-DC phases."""
    length = int(segment.shape[-1])
    if length < 3:
        return segment.clone()
    spectrum = torch.fft.rfft(segment, dim=-1)
    magnitude = torch.abs(spectrum)
    original_phase = torch.angle(spectrum)
    randomized_phase = original_phase.clone()
    last_exclusive = spectrum.shape[-1] - 1 if length % 2 == 0 else spectrum.shape[-1]
    if last_exclusive > 1:
        randomized_phase[:, 1:last_exclusive] = (
            torch.rand(
                (segment.shape[0], last_exclusive - 1),
                generator=generator,
                device=segment.device,
                dtype=segment.dtype,
            )
            * (2.0 * torch.pi)
            - torch.pi
        )
    randomized = torch.polar(magnitude, randomized_phase)
    return torch.fft.irfft(randomized, n=length, dim=-1)


def spectrum_matched_noise(segment: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Create zero-mean colored noise with the segment's spectral shape and unit SD."""
    centered = segment - segment.mean(dim=-1, keepdim=True)
    surrogate = phase_randomize_segment(centered, generator)
    surrogate = surrogate - surrogate.mean(dim=-1, keepdim=True)
    standard_deviation = surrogate.std(dim=-1, unbiased=False, keepdim=True)
    return torch.where(
        standard_deviation > 1e-8,
        surrogate / standard_deviation.clamp_min(1e-8),
        torch.zeros_like(surrogate),
    )


def perturb_input(
    x: torch.Tensor,
    mask: torch.Tensor,
    scheme: str,
    generator: torch.Generator,
    noise_strength: Optional[float] = None,
    sampling_hz: float = 100.0,
    band_hz: Tuple[float, float] = (15.0, 50.0),
    components_by_sample: Optional[List[List[Tuple[int, int]]]] = None,
) -> torch.Tensor:
    """Perturb only the union of clipped +/-1-s significant-point regions."""
    output = x.clone()
    if scheme == "mean":
        fill = x.mean(dim=2, keepdim=True).expand_as(x)
        return torch.where(mask.unsqueeze(1), fill, x)

    for sample_index in range(x.shape[0]):
        components = (
            components_by_sample[sample_index]
            if components_by_sample is not None
            else connected_components(mask[sample_index])
        )
        for start, stop in components:
            original = x[sample_index, :, start:stop]
            if scheme == "phase":
                changed = phase_randomize_segment(original, generator)
            elif scheme == "noise":
                if noise_strength is None:
                    raise ValueError("noise_strength is required for noise injection")
                noise = spectrum_matched_noise(original, generator)
                scale = original.std(dim=-1, unbiased=False, keepdim=True)
                changed = original + float(noise_strength) * scale * noise
            elif scheme == "band":
                length = int(original.shape[-1])
                spectrum = torch.fft.rfft(original, dim=-1)
                frequencies = torch.fft.rfftfreq(length, d=1.0 / sampling_hz).to(original.device)
                remove = (frequencies >= band_hz[0]) & (frequencies <= band_hz[1])
                spectrum[:, remove] = 0
                changed = torch.fft.irfft(spectrum, n=length, dim=-1)
            else:
                raise ValueError(f"Unsupported scheme: {scheme}")
            output[sample_index, :, start:stop] = changed
    return output


def probability_details(
    y_true: np.ndarray, probabilities: np.ndarray, logits: np.ndarray
) -> Dict[str, float]:
    rows = np.arange(len(y_true))
    true_probability = probabilities[rows, y_true]
    other_logits = logits.copy()
    other_logits[rows, y_true] = -np.inf
    margin = logits[rows, y_true] - np.max(other_logits, axis=1)
    return {
        "true_probability": float(np.mean(true_probability) * 100.0),
        "logit_margin": float(np.mean(margin)),
    }


@torch.no_grad()
def evaluate_stage(
    model,
    dataset: LegacySegmentDataset,
    indices: List[int],
    config,
    target_raw_label: Optional[int],
    scheme: Optional[str],
    repeats: int,
    seed: int,
    noise_strength: Optional[float] = None,
) -> Dict:
    loader = DataLoader(Subset(dataset, indices), batch_size=config.batch_size, shuffle=False)
    y_parts: List[np.ndarray] = []
    probability_parts: List[np.ndarray] = []
    logit_parts: List[np.ndarray] = []
    affected_windows = 0
    critical_time_bins = 0
    # FFT perturbations stay on CPU because this Windows PyTorch build lacks
    # the NVRTC complex-kernel runtime.  Only model inference is transferred
    # to the configured GPU; the numerical perturbation definition is unchanged.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    stochastic = scheme in {"phase", "noise"}
    actual_repeats = int(repeats if stochastic else 1)

    for x, y, all_masks, _ in loader:
        x = x.float()
        if target_raw_label is None:
            logits = model(x.to(config.device))
            probabilities = torch.softmax(logits, dim=1)
        else:
            mask = all_masks[:, int(target_raw_label) - 1, :].bool()
            affected = mask.any(dim=1)
            affected_windows += int(affected.sum().item())
            critical_time_bins += int(mask.sum().item())
            base_logits = model(x.to(config.device))
            logits = base_logits.clone()
            probabilities = torch.softmax(base_logits, dim=1)
            if affected.any():
                affected_x = x[affected]
                affected_mask = mask[affected]
                components_by_sample = [connected_components(row) for row in affected_mask]
                probability_sum = torch.zeros(
                    (affected_x.shape[0], config.num_classes), device=config.device
                )
                logit_sum = torch.zeros_like(probability_sum)
                for _ in range(actual_repeats):
                    changed = perturb_input(
                        affected_x,
                        affected_mask,
                        str(scheme),
                        generator,
                        noise_strength=noise_strength,
                        components_by_sample=components_by_sample,
                    )
                    repeat_logits = model(changed.to(config.device))
                    logit_sum += repeat_logits
                    probability_sum += torch.softmax(repeat_logits, dim=1)
                affected_device = affected.to(config.device)
                logits[affected_device] = logit_sum / float(actual_repeats)
                probabilities[affected_device] = probability_sum / float(actual_repeats)

        y_parts.append(y.numpy().astype(int))
        probability_parts.append(probabilities.detach().cpu().numpy())
        logit_parts.append(logits.detach().cpu().numpy())

    y_true = np.concatenate(y_parts)
    probabilities = np.concatenate(probability_parts)
    logits = np.concatenate(logit_parts)
    predictions = np.argmax(probabilities, axis=1)
    pack = compute_stage_metrics(y_true, predictions, probabilities, config.num_classes)
    pack.update(probability_details(y_true, probabilities, logits))
    pack["confusion_matrix"] = confusion_matrix(
        y_true, predictions, labels=list(range(config.num_classes))
    ).astype(int).tolist()
    pack["affected_windows"] = int(affected_windows)
    pack["critical_time_bins"] = int(critical_time_bins)
    pack["repeats"] = int(actual_repeats if target_raw_label is not None else 0)
    return pack


def merge_details(run1: Dict, run2: Dict) -> Dict:
    merged = merge_training_style(run1, run2)
    merged["true_probability"] = (
        float(run1["true_probability"]) + float(run2["true_probability"])
    ) / 2.0
    merged["logit_margin"] = (
        float(run1["logit_margin"]) + float(run2["logit_margin"])
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


def build_tables(output_dir: Path, aggregate: List[Dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "Method": row["display_name"],
                "Acc": mean_sd_text(row["fold_accuracy"]),
                "Recall": mean_sd_text(row["fold_recall"]),
                "F1": mean_sd_text(row["fold_f1"]),
            }
            for row in aggregate
        ]
    ).to_csv(output_dir / "table4_overall_mean_sd.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
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
    ).to_csv(output_dir / "table5_class_accuracy_mean_sd.csv", index=False, encoding="utf-8-sig")
    baseline = aggregate[0]
    delta_rows = []
    for row in aggregate[1:]:
        item = {"condition": row["display_name"], "target_raw_label": row["target_raw_label"]}
        for key in METRIC_KEYS + ["true_probability", "logit_margin"]:
            delta = np.asarray(baseline[f"fold_{key}"]) - np.asarray(row[f"fold_{key}"])
            item[f"drop_{key}_mean"] = float(delta.mean())
            item[f"drop_{key}_sd"] = float(delta.std(ddof=1))
        delta_rows.append(item)
    pd.DataFrame(delta_rows).to_csv(
        output_dir / "perturbation_delta_vs_baseline.csv", index=False, encoding="utf-8-sig"
    )


def output_directory(variant_key: str) -> Path:
    if variant_key.startswith("noise_"):
        strength = variant_key.split("_", 1)[1]
        return SCHEME_DIRS["noise"] / "outputs" / f"strength_{strength}"
    return SCHEME_DIRS[variant_key] / "outputs"


def write_variant_outputs(
    variant_key: str,
    scheme: str,
    noise_strength: Optional[float],
    sig: pd.DataFrame,
    ds1: LegacySegmentDataset,
    ds2: LegacySegmentDataset,
    fold_rows: List[Dict],
    aggregate: List[Dict],
    args,
) -> None:
    out = output_directory(variant_key)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(out / "condition_fold_metrics.csv", index=False, encoding="utf-8-sig")
    sig.to_csv(out / "significant_10ms_timepoints_used.csv", index=False, encoding="utf-8-sig")
    (out / "conv_perturbation_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "dataset_mapping_qc.json").write_text(
        json.dumps({"run1": ds1.mapping_stats, "run2": ds2.mapping_stats}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    config = {
        "python": sys.executable,
        "variant_key": variant_key,
        "scheme": scheme,
        "scheme_title": SCHEME_TITLES[scheme],
        "noise_strength_sigma_x": noise_strength,
        "random_repeats": int(args.random_repeats if scheme in {"phase", "noise"} else 1),
        "seed": int(args.seed),
        "sampling_hz": 100.0,
        "conv_kernel_bins": int(Config.kernel_size),
        "conv_receptive_field_ms": float(Config.kernel_size * 10),
        "frequency_band_removed_hz": [15.0, 50.0] if scheme == "band" else None,
        "mean_source": "each neural channel's original full 5-s window" if scheme == "mean" else None,
        "critical_region": "union of [significant center-1 s, significant center+1 s), clipped at 5-s boundary",
        "window_bins": 500,
        "window_seconds": 5.0,
        "step_bins": 250,
        "overlap_fraction": 0.5,
        "constant_stage_label_required": True,
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
        f"# {SCHEME_TITLES[scheme]}",
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


def variants_for_schemes(schemes: Sequence[str]) -> List[Tuple[str, str, Optional[float]]]:
    variants: List[Tuple[str, str, Optional[float]]] = []
    for scheme in schemes:
        if scheme == "noise":
            for strength in NOISE_STRENGTHS:
                text = str(strength).replace(".", "p")
                variants.append((f"noise_{text}", "noise", float(strength)))
        else:
            variants.append((scheme, scheme, None))
    return variants


def run_schemes(args, schemes: Sequence[str]) -> None:
    # Thousands of short FFTs are much faster without per-FFT thread-pool
    # fan-out.  Neural-network inference still runs on the configured GPU.
    torch.set_num_threads(1)
    invalid = [scheme for scheme in schemes if scheme not in SCHEME_DIRS]
    if invalid:
        raise ValueError(f"Unsupported schemes: {invalid}")
    variants = variants_for_schemes(schemes)
    sig = read_significant_timepoints(args.xlsx_path)
    offsets = significant_offsets(sig)
    ds1 = LegacySegmentDataset(Config(), "run1", offsets, radius_bins=100)
    ds2 = LegacySegmentDataset(Config3(), "run2", offsets, radius_bins=100)
    if ds1.x.shape[2] != 500 or ds2.x.shape[2] != 500:
        raise AssertionError("Legacy input must remain 500 bins.")
    if Config.step_size != 250 or Config3.step_size != 250:
        raise AssertionError("Legacy step must remain 250 bins (50% overlap).")
    if Config.kernel_size != 3 or Config3.kernel_size != 3:
        raise AssertionError("Expected the saved model's single 3-bin Conv receptive field.")

    rows_by_variant = {key: [] for key, _, _ in variants}
    canonical_variants = variants_for_schemes(list(SCHEME_DIRS))
    canonical_index = {key: index for index, (key, _, _) in enumerate(canonical_variants)}

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

        for variant_key, scheme, strength in variants:
            rows_by_variant[variant_key].append(
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
                    args.seed + fold_index * 100000 + canonical_index[variant_key] * 10000 + target * 100
                )
                pack1 = evaluate_stage(
                    model1, ds1, indices1, Config(), target, scheme,
                    args.random_repeats, seed_base + 1, noise_strength=strength,
                )
                pack2 = evaluate_stage(
                    model2, ds2, indices2, Config3(), target, scheme,
                    args.random_repeats, seed_base + 2, noise_strength=strength,
                )
                merged = merge_details(pack1, pack2)
                rows_by_variant[variant_key].append(
                    {
                        "condition": f"ablate_l{target}", "display_name": f"Ablate L{target}",
                        "target_raw_label": target, "fold": fold_tag, **merged,
                        "run1_accuracy": pack1["accuracy"], "run2_accuracy": pack2["accuracy"],
                        "run1_affected_windows": pack1["affected_windows"],
                        "run2_affected_windows": pack2["affected_windows"],
                        "run1_critical_time_bins": pack1["critical_time_bins"],
                        "run2_critical_time_bins": pack2["critical_time_bins"],
                        "run1_repeats": pack1["repeats"], "run2_repeats": pack2["repeats"],
                        "run1_confusion_matrix": json.dumps(pack1["confusion_matrix"]),
                        "run2_confusion_matrix": json.dumps(pack2["confusion_matrix"]),
                    }
                )
                print(f"  {variant_key:12s} L{target} complete", flush=True)
        del model1, model2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sweep_rows = []
    for variant_key, scheme, strength in variants:
        aggregate = aggregate_rows(rows_by_variant[variant_key])
        write_variant_outputs(
            variant_key, scheme, strength, sig, ds1, ds2, rows_by_variant[variant_key], aggregate, args
        )
        table = pd.read_csv(output_directory(variant_key) / "table4_overall_mean_sd.csv")
        print(f"\n[{variant_key}] {SCHEME_TITLES[scheme]}\n{table.to_string(index=False)}", flush=True)
        if scheme == "noise":
            baseline = aggregate[0]
            for row in aggregate[1:]:
                sweep_rows.append(
                    {
                        "noise_strength_sigma_x": strength,
                        "condition": row["display_name"],
                        "drop_accuracy_mean": float(
                            (np.asarray(baseline["fold_accuracy"]) - np.asarray(row["fold_accuracy"])).mean()
                        ),
                        "drop_recall_mean": float(
                            (np.asarray(baseline["fold_recall"]) - np.asarray(row["fold_recall"])).mean()
                        ),
                        "drop_f1_mean": float(
                            (np.asarray(baseline["fold_f1"]) - np.asarray(row["fold_f1"])).mean()
                        ),
                    }
                )
    if sweep_rows:
        noise_output = SCHEME_DIRS["noise"] / "outputs"
        noise_output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(sweep_rows).to_csv(
            noise_output / "noise_strength_dose_response.csv", index=False, encoding="utf-8-sig"
        )


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
