import json
import platform
import sys

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

import config
from analysis_core import run_all_experiments
from make_figures import make_all_figures


def main():
    config.ensure_dirs()
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }
    (config.OUTPUT_DIR / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = run_all_experiments()
    make_all_figures()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

