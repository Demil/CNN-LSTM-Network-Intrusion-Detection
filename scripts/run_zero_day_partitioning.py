"""CLI Runner for executing Zero-Day Partitioning and Feature Normalization (Option B)."""

import sys
from pathlib import Path

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from configs.project_config import MODELS_DIR, PROCESSED_DATA_DIR
from src.preprocessing.zero_day_partitioner import ZeroDayPartitioner


def main():
    input_csv = PROCESSED_DATA_DIR / "central_mixed_dataset.csv"
    partitioner = ZeroDayPartitioner(
        input_csv_path=input_csv,
        output_dir=PROCESSED_DATA_DIR,
        models_dir=MODELS_DIR,
        split_ratios=(0.70, 0.15, 0.15),
    )
    partitioner.run()


if __name__ == "__main__":
    main()
