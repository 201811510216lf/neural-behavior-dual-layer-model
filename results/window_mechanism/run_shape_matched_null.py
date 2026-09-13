"""Run an exact-mask-geometry circular-shift null for the frozen Full model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_experiments as core


ROOT = Path(__file__).resolve().parent
REPEATS = 500


@torch.no_grad()
def shifted_stage_packs(model, dataset, indices, config, target: int, repeats: int, seed: int):
    _, arrays = core.predict_stage(model, dataset, indices, config)
    test_indices = np.asarray(indices, dtype=int)
    masks = dataset.segment_masks[test_indices, target - 1, :]
    affected_local = np.flatnonzero(masks.any(axis=1))
    if len(affected_local) == 0:
        return [
            core.pack_from_arrays(arrays["y_true"], arrays["probability"], arrays["logits"], config.num_classes)
            for _ in range(repeats)
        ]

    affected_global = test_indices[affected_local]
    source_x = torch.from_numpy(dataset.x[affected_global]).float()
    source_masks = masks[affected_local]
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(repeats):
        shifted = np.zeros_like(source_masks, dtype=bool)
        for row in range(len(source_masks)):
            shift = int(rng.integers(1, source_masks.shape[1]))
            shifted[row] = np.roll(source_masks[row], shift)
        changed = core.replace_segment(source_x, torch.from_numpy(shifted), "max")
        logits_parts = []
        for start in range(0, len(changed), config.batch_size):
            logits_parts.append(model(changed[start : start + config.batch_size].to(config.device)).cpu().numpy())
        changed_logits = np.concatenate(logits_parts)
        changed_probability = torch.softmax(torch.from_numpy(changed_logits), dim=1).numpy()
        rep_logits = arrays["logits"].copy()
        rep_probability = arrays["probability"].copy()
        rep_logits[affected_local] = changed_logits
        rep_probability[affected_local] = changed_probability
        pack = core.pack_from_arrays(arrays["y_true"], rep_probability, rep_logits, config.num_classes)
        rows.append(pack)
    return rows


def main() -> None:
    sig = core.read_significant_timepoints(core.XLSX_PATH)
    offsets = core.significant_offsets(sig)
    datasets = {
        "run1": core.LegacySegmentDataset(core.Config(), "run1", offsets, radius_bins=100),
        "run2": core.LegacySegmentDataset(core.Config3(), "run2", offsets, radius_bins=100),
    }
    configs = {"run1": core.Config(), "run2": core.Config3()}
    architecture = pd.read_csv(ROOT / "source_data" / "fig_b_architecture_perturbation_fold.csv")
    output = []
    for fold_i, fold in enumerate(core.FOLDS):
        models = {
            run_tag: core.load_model("full", run_tag, fold, datasets[run_tag], configs[run_tag])
            for run_tag in ("run1", "run2")
        }
        test = {run_tag: core.split_indices(run_tag, fold)[1] for run_tag in ("run1", "run2")}
        for target in core.TARGETS:
            stage = {
                run_tag: shifted_stage_packs(
                    models[run_tag], datasets[run_tag], test[run_tag], configs[run_tag],
                    target=target, repeats=REPEATS,
                    seed=20261301 + fold_i * 10000 + target * 100,
                )
                for run_tag in ("run1", "run2")
            }
            merged = [core.merge_stage_packs(stage["run1"][i], stage["run2"][i]) for i in range(REPEATS)]
            metric = core.OUTCOME_KEYS[target]
            observed_rows = architecture[
                (architecture.architecture == "full") & (architecture.target == target) & (architecture.fold == fold)
            ]
            baseline = float(observed_rows[observed_rows.condition == "baseline"][metric].iloc[0])
            observed = float(observed_rows[observed_rows.condition == "maximum"][metric].iloc[0] - baseline)
            deltas = np.asarray([row[metric] - baseline for row in merged], dtype=float)
            p = (1.0 + float((deltas <= observed).sum())) / (REPEATS + 1.0)
            for repeat, delta in enumerate(deltas, start=1):
                output.append(
                    {
                        "target": target,
                        "target_label": core.TARGET_LABELS[target],
                        "outcome": core.OUTCOME_LABELS[target],
                        "fold": fold,
                        "repeat": repeat,
                        "random_delta": float(delta),
                        "observed_maximum_delta": observed,
                        "empirical_one_sided_p": p,
                        "null_type": "circular_shift_exact_mask_geometry",
                    }
                )
        del models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[shape-matched null] {fold} complete", flush=True)

    table = pd.DataFrame(output)
    table.to_csv(ROOT / "source_data" / "fig_d_shape_matched_null.csv", index=False, encoding="utf-8-sig")
    table[
        ["target", "target_label", "outcome", "fold", "observed_maximum_delta", "empirical_one_sided_p", "null_type"]
    ].drop_duplicates().to_csv(
        ROOT / "tables" / "shape_matched_random_window_empirical_p.csv", index=False, encoding="utf-8-sig"
    )
    (ROOT / "qa" / "shape_matched_null_metadata.json").write_text(
        json.dumps(
            {
                "repeats_per_target_per_fold": REPEATS,
                "null": "independent nonzero circular shift of each affected sample's exact binary mask",
                "replacement": "channel-wise maximum from the unchanged 5 s sample",
                "model": "frozen Full",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

