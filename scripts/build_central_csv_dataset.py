"""CLI runner for generating the single central combined reference CSV dataset."""

import sys
from pathlib import Path

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from configs.project_config import (
    CICIDS2017_DIR,
    CSE_CIC_IDS2018_DIR,
    PROCESSED_DATA_DIR,
)
from src.preprocessing.central_csv_builder import CentralCSVBuilder


def main():
    builder = CentralCSVBuilder(
        cicids2017_dir=CICIDS2017_DIR,
        csecicids2018_dir=CSE_CIC_IDS2018_DIR,
        output_dir=PROCESSED_DATA_DIR,
    )
    builder.run()


if __name__ == "__main__":
    main()
