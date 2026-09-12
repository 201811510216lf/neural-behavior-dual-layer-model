from pathlib import Path

import torch


class CommonConfig:
    project_dir = Path(__file__).resolve().parent
    repo_root = project_dir.parents[1]
    data_dir = repo_root / "data" / "aligned_tables"
    output_dir = repo_root / "checkpoints"

    # Current raster data rule:
    # raw label 0 is removed; raw labels 1-5 are converted to classes 0-4.
    valid_raw_labels = (1, 2, 3, 4, 5)
    raw_label_to_final_class = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    final_class_to_raw_label = {v: k for k, v in raw_label_to_final_class.items()}
    final_class_names = {
        0: "L1: Hole Exploration",
        1: "L2: Wrong Hole Exploration",
        2: "L3: Hesitating",
        3: "L4: Changing Direction",
        4: "L5: Walking",
    }

    # Old hierarchical strategy after label conversion to 0-4:
    # run1: keep classes 0 and 2, merge classes 1/3/4 into class 1.
    # run2: split only original final classes 1/3/4.
    merged_stage_classes = (1, 3, 4)
    run2_label_map = {1: 0, 3: 1, 4: 2}
    run2_inverse_label_map = {0: 1, 1: 3, 2: 4}

    window_size = 500
    step_size = int(0.5 * window_size)

    hidden_size = 256
    attention_hidden_size = 128
    conv_channels = 64
    kernel_size = 3
    num_layers = 2
    dropout = 0.0

    epochs = 500
    weight_decay = 1e-5
    grad_clip = 1.0
    seed = 25
    n_splits = 3
    num_workers = 0

    # Works on Windows/Linux/macOS. On Apple Silicon, PyTorch can use MPS.
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"


class Config(CommonConfig):
    """Stage 1: 3-class coarse model."""

    run_tag = "run1"
    num_classes = 3
    batch_size = 32
    lr = 0.001


class Config3(CommonConfig):
    """Stage 2: 3-class model inside final classes 1/3/4."""

    run_tag = "run2"
    num_classes = 3
    batch_size = 64
    lr = 0.0005
