"""Run classical baselines on label-constant temporal windows.

The private aligned hippocampal tables are intentionally absent from this
repository. Supply a directory containing CSV/CSV.GZ files whose final column
is the integer behavior label and whose preceding columns are neural features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def load_windows(data_dir: Path, window_size: int, step_size: int):
    windows, labels = [], []
    for path in sorted(data_dir.glob("*.csv*")):
        frame = pd.read_csv(path)
        values = frame.iloc[:, :-1].to_numpy(dtype=np.float32)
        target = frame.iloc[:, -1].to_numpy(dtype=int)
        for start in range(0, len(frame) - window_size + 1, step_size):
            y = target[start : start + window_size]
            if len(y) and np.all(y == y[0]) and y[0] in {1, 2, 3, 4, 5}:
                windows.append(values[start : start + window_size].T.reshape(-1))
                labels.append(int(y[0] - 1))
    if not windows:
        raise ValueError("No label-constant windows were found")
    return np.asarray(windows), np.asarray(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("baseline_comparison.csv"))
    parser.add_argument("--window-size", type=int, default=500)
    parser.add_argument("--step-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=25)
    args = parser.parse_args()

    x, y = load_windows(args.data_dir, args.window_size, args.step_size)
    models = {
        "SVM": make_pipeline(StandardScaler(), SVC()),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=args.seed),
        "Linear": make_pipeline(StandardScaler(), SGDClassifier(random_state=args.seed)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000, random_state=args.seed)
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=args.seed),
        "Three-layer MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=1000, random_state=args.seed),
        ),
    }
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    rows = []
    for name, model in models.items():
        fold_scores = []
        for train_idx, test_idx in splitter.split(x, y):
            model.fit(x[train_idx], y[train_idx])
            pred = model.predict(x[test_idx])
            fold_scores.append(
                (
                    accuracy_score(y[test_idx], pred),
                    recall_score(y[test_idx], pred, average="macro", zero_division=0),
                    f1_score(y[test_idx], pred, average="macro", zero_division=0),
                )
            )
        scores = np.asarray(fold_scores) * 100
        rows.append(
            {
                "method": name,
                "accuracy_mean": scores[:, 0].mean(),
                "accuracy_sd": scores[:, 0].std(ddof=1),
                "recall_mean": scores[:, 1].mean(),
                "recall_sd": scores[:, 1].std(ddof=1),
                "f1_mean": scores[:, 2].mean(),
                "f1_sd": scores[:, 2].std(ddof=1),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
