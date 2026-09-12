from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[1]
LEGACY_PROJECT = PROJECT_DIR.parent / "dual_layer_model"
ALIGNED_DIR = REPO_ROOT / "data" / "aligned_tables"
LEGACY_OUTPUT = REPO_ROOT / "checkpoints"

OUTPUT_DIR = REPO_ROOT / "results" / "association"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
ARRAY_DIR = OUTPUT_DIR / "arrays"

WINDOW_SIZE = 500
STEP_SIZE = 250
BIN_SIZE = 10  # 10 ms samples -> 100 ms bins
SEED = 25
N_SPLITS = 3
MIN_WINDOWS_RSA = 4

RUNS = {
    "run1": {
        "num_classes": 3,
        "label_mode": "coarse",
    },
    "run2": {
        "num_classes": 3,
        "label_mode": "fine",
    },
}

FINAL_CLASS_NAMES = {
    0: "L1 Hole Exploration",
    1: "L2 Wrong Hole Exploration",
    2: "L3 Hesitating",
    3: "L4 Changing Direction",
    4: "L5 Walking",
}

LAYER_ORDER = ["Conv", "LSTM", "Attention contribution", "Context"]
INTERVENTION_ORDER = ["Full", "Zero Conv signal", "Zero LSTM signal", "Uniform Attention"]


def ensure_dirs() -> None:
    for path in (OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, ARRAY_DIR):
        path.mkdir(parents=True, exist_ok=True)
