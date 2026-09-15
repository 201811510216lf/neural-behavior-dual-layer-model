"""Trace historical ablation-table values and build a single consistency ledger."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
VIDEO = ROOT.parents[1]
OLD_ROOT = VIDEO / "multi_classification" / "ablationstudy"
PUBLIC_ROOT = VIDEO / "neural-behavior-dual-layer-model"
SOURCE = ROOT / "source_data"
TABLES = ROOT / "tables"
QA = ROOT / "qa"


def extract_tenfold() -> pd.DataFrame:
    text = (OLD_ROOT / "新建 文本文档.txt").read_text(encoding="utf-8", errors="replace")
    headers = list(re.finditer(r"^=+([A-Za-z][A-Za-z0-9]+)=+\s*$", text, flags=re.M))
    rows: List[Dict] = []
    for index, header in enumerate(headers):
        model = header.group(1).strip()
        stop = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        body = text[header.end():stop]
        summaries = list(
            re.finditer(
                r"Average Accuracy:\s*([0-9.]+)%\s*±\s*([0-9.]+)%.*?"
                r"Class 0:\s*([0-9.]+)%\s*±\s*([0-9.]+)%.*?"
                r"Class 1:\s*([0-9.]+)%\s*±\s*([0-9.]+)%.*?"
                r"Class 2:\s*([0-9.]+)%\s*±\s*([0-9.]+)%.*?"
                r"Class 3:\s*([0-9.]+)%\s*±\s*([0-9.]+)%.*?"
                r"Class 4:\s*([0-9.]+)%\s*±\s*([0-9.]+)%",
                body,
                flags=re.S,
            )
        )
        if not summaries:
            continue
        values = list(map(float, summaries[-1].groups()))
        row: Dict = {"source_protocol": "historical text 10-fold", "source_model": model}
        row["accuracy_mean"], row["accuracy_sd"] = values[0], values[1]
        for class_id in range(5):
            row[f"c{class_id}_mean"] = values[2 + class_id * 2]
            row[f"c{class_id}_sd"] = values[3 + class_id * 2]
        rows.append(row)
    return pd.DataFrame(rows)


def read_fivefold() -> pd.DataFrame:
    rows: List[Dict] = []
    for path in sorted((OLD_ROOT / "results").glob("*_fold5_summary.csv")):
        item = pd.read_csv(path).iloc[0]
        row: Dict = {
            "source_protocol": "historical CSV 5-fold",
            "source_model": path.name.replace("_fold5_summary.csv", ""),
            "accuracy_mean": float(item["Fold_Avg_Accuracy"]),
            "accuracy_sd": float(item["Fold_Std_Accuracy"]),
        }
        for class_id in range(5):
            row[f"c{class_id}_mean"] = float(item[f"Class_{class_id}_Avg"])
            row[f"c{class_id}_sd"] = float(item[f"Class_{class_id}_Std"])
        rows.append(row)
    return pd.DataFrame(rows)


def match_public_cells(candidates: pd.DataFrame) -> pd.DataFrame:
    public = pd.read_csv(PUBLIC_ROOT / "results" / "model" / "structural_ablation_per_class_ppt.csv")
    value_columns = [
        ("hole_exploration", 0),
        ("wrong_hole", 1),
        ("hesitating", 2),
        ("changing_direction", 3),
        ("walking", 4),
    ]
    rows: List[Dict] = []
    for _, public_row in public.iterrows():
        for public_key, class_id in value_columns:
            for statistic in ("mean", "sd"):
                value = float(public_row[f"{public_key}_{statistic}"])
                candidate_key = f"c{class_id}_{statistic}"
                matches = candidates[np.isclose(candidates[candidate_key], value, atol=0.005)]
                rows.append(
                    {
                        "public_method": public_row["method"],
                        "class_id": class_id,
                        "class_name": public_key,
                        "statistic": statistic,
                        "public_value": value,
                        "matched_sources": " | ".join(
                            f"{row.source_protocol}: {row.source_model}"
                            for row in matches.itertuples()
                        ),
                        "match_count": int(len(matches)),
                    }
                )
    return pd.DataFrame(rows)


def current_structural_summary() -> pd.DataFrame:
    data = pd.read_csv(SOURCE / "fig_b_architecture_perturbation_fold.csv")
    data = data[(data["condition"] == "baseline") & (data["target"] == 2)].copy()
    labels = {
        "full": "Full",
        "no_conv": "NoConv",
        "no_lstm": "NoLSTM",
        "no_attention": "NoAttention",
    }
    rows: List[Dict] = []
    for architecture, group in data.groupby("architecture", sort=False):
        row: Dict = {
            "protocol": "coherent legacy-hierarchical three-fold",
            "architecture": architecture,
            "display_name": labels[architecture],
            "n_folds": int(group["fold"].nunique()),
        }
        for metric in ("accuracy", "recall", "f1", "c0", "c1", "c2", "c3", "c4"):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def baseline_comparison() -> pd.DataFrame:
    structural = pd.read_csv(PUBLIC_ROOT / "results" / "model" / "structural_ablation_per_class_ppt.csv")
    structural_overall = pd.read_csv(PUBLIC_ROOT / "results" / "model" / "structural_ablation_overall_ppt.csv")
    mask_class = pd.read_csv(
        PUBLIC_ROOT
        / "results"
        / "masking"
        / "segment_replacement"
        / "maximum"
        / "table5_class_accuracy_mean_sd.csv"
    )
    mask_overall = pd.read_csv(
        PUBLIC_ROOT
        / "results"
        / "masking"
        / "segment_replacement"
        / "maximum"
        / "table4_overall_mean_sd.csv"
    )
    old_full = structural[structural["method"] == "Full model"].iloc[0]
    old_overall = structural_overall[structural_overall["method"] == "Full model"].iloc[0]
    mask_full = mask_class[mask_class["Method"] == "Our Proposed"].iloc[0]
    mask_overall_full = mask_overall[mask_overall["Method"] == "Our Proposed"].iloc[0]
    rows = [
        {
            "metric": "accuracy",
            "historical_structural_full": float(old_overall["accuracy_mean"]),
            "threefold_mask_baseline": float(str(mask_overall_full["Acc"]).split("±")[0].strip()),
        }
    ]
    mapping = [
        ("c0", "hole_exploration_mean", "Hole Exploration"),
        ("c1", "wrong_hole_mean", "Wrong Hole Exploration"),
        ("c2", "hesitating_mean", "Hesitating"),
        ("c3", "changing_direction_mean", "Changing Direction"),
        ("c4", "walking_mean", "Walking"),
    ]
    for key, old_key, mask_key in mapping:
        rows.append(
            {
                "metric": key,
                "historical_structural_full": float(old_full[old_key]),
                "threefold_mask_baseline": float(str(mask_full[mask_key]).split("±")[0].strip()),
            }
        )
    out = pd.DataFrame(rows)
    out["mask_minus_historical_pp"] = (
        out["threefold_mask_baseline"] - out["historical_structural_full"]
    )
    out["comparison_valid"] = False
    out["reason"] = "different protocol/model generation; retain as separate historical result"
    return out


def main() -> None:
    QA.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    tenfold = extract_tenfold()
    fivefold = read_fivefold()
    candidates = pd.concat([tenfold, fivefold], ignore_index=True)
    tenfold.to_csv(QA / "legacy_10fold_summary_extracted.csv", index=False, encoding="utf-8-sig")
    fivefold.to_csv(QA / "legacy_5fold_summary_files.csv", index=False, encoding="utf-8-sig")
    match_public_cells(candidates).to_csv(
        QA / "public_table5_cell_source_matches.csv", index=False, encoding="utf-8-sig"
    )
    current_structural_summary().to_csv(
        TABLES / "table_e_current_structural_ablation_coherent.csv",
        index=False,
        encoding="utf-8-sig",
    )
    baseline_comparison().to_csv(
        TABLES / "table_f_baseline_protocol_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("Legacy consistency audit complete.", flush=True)


if __name__ == "__main__":
    main()
