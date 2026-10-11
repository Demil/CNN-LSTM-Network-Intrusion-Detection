"""Central Configuration for CNN-LSTM Network Intrusion Detection Research.

Defines all canonical project paths, dataset directories, and artifact targets
using pathlib for cross-platform robustness and central maintainability.
"""

from pathlib import Path

# Base Project Root
PROJECT_DIR: Path = Path(__file__).resolve().parent.parent

# Discovered Dataset Locations (Read-Only)
CICIDS2017_DIR: Path = PROJECT_DIR / "CICIDS2017 Dataset"
CSE_CIC_IDS2018_DIR: Path = PROJECT_DIR / "CSE-CIC-IDS2018"

# Dataset Roles & Research Architecture
# CSE-CIC-IDS2018: Primary dataset for model development, temporal sequence learning,
#                  and controlled low-frequency intrusion experiments.
PRIMARY_DATASET_NAME: str = "CSE-CIC-IDS2018"
PRIMARY_DATASET_DIR: Path = CSE_CIC_IDS2018_DIR

# CICIDS2017: Reserved primarily for cross-dataset generalization and carefully
#             designed unseen-attack / zero-day evaluation.
EVAL_DATASET_NAME: str = "CICIDS2017"
EVAL_DATASET_DIR: Path = CICIDS2017_DIR

# Approved Research Feature Directives
# Strictly excluded to prevent shortcut service-port memorization and identifier leakage
EXCLUDED_FEATURES: list = [
    "Dst Port",
    "Destination Port",
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
]
DEDUPLICATION_REQUIRED: bool = True

# Data Subdirectories
DATA_DIR: Path = PROJECT_DIR / "data"
AUDIT_DIR: Path = DATA_DIR / "audit"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
SEQUENCE_DATA_DIR: Path = DATA_DIR / "sequences"

# Model and Experiment Output Directories
MODELS_DIR: Path = PROJECT_DIR / "models"
RESULTS_DIR: Path = PROJECT_DIR / "results"
REPORTS_DIR: Path = PROJECT_DIR / "reports"

# Ensure runtime output directories exist (never raw dataset directories)
ALL_OUTPUT_DIRS = [
    DATA_DIR,
    AUDIT_DIR,
    PROCESSED_DATA_DIR,
    SEQUENCE_DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    REPORTS_DIR,
]


def init_project_directories() -> None:
    """Creates output and artifact directories if they do not exist.
    
    Leaves raw dataset directories untouched.
    """
    for d in ALL_OUTPUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    init_project_directories()
    print("Project directories initialized successfully.")
    print(f"PROJECT_DIR: {PROJECT_DIR}")
    print(f"CICIDS2017_DIR: {CICIDS2017_DIR} (Exists: {CICIDS2017_DIR.exists()})")
    print(f"CSE_CIC_IDS2018_DIR: {CSE_CIC_IDS2018_DIR} (Exists: {CSE_CIC_IDS2018_DIR.exists()})")
    print(f"AUDIT_DIR: {AUDIT_DIR}")
