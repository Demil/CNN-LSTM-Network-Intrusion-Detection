"""Comprehensive Dataset Audit Engine for CICIDS2017 and CSE-CIC-IDS2018.

This module performs streaming, read-only discovery, schema inspection, label distribution
auditing, temporal analysis, data quality checks, feature statistics, potential leakage
identification, and cross-dataset compatibility analysis without modifying any raw files.

Designed for memory-efficient execution on resource-constrained hardware (e.g. 8GB RAM).
"""

import csv
import glob
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple


@dataclass
class FileMetadata:
    """Metadata for an audited CSV file."""
    dataset_name: str
    filename: str
    relative_path: str
    file_size_bytes: int
    file_size_mb: float
    extension: str
    num_rows: int
    num_cols: int
    column_names: List[str]
    label_column: str
    has_repeated_headers: bool
    repeated_header_count: int
    in_file_duplicates: int
    in_file_duplicate_pct: float


class DatasetAuditor:
    """Read-only dataset audit suite for intrusion detection research."""

    def __init__(self, cicids2017_dir: Path, csecic2018_dir: Path, audit_dir: Path):
        """Initialize the auditor with dataset and output directories.

        Parameters
        ----------
        cicids2017_dir : Path
            Path to the read-only CICIDS2017 directory.
        csecic2018_dir : Path
            Path to the read-only CSE-CIC-IDS2018 directory.
        audit_dir : Path
            Directory where audit artifacts and reports will be saved.
        """
        self.cicids2017_dir = Path(cicids2017_dir)
        self.csecic2018_dir = Path(csecic2018_dir)
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_hardware_and_software_environment() -> Dict[str, Any]:
        """Collect current system hardware and software versions.

        Purpose
        -------
        Record the exact computing environment to ensure audit reproducibility.

        Inputs
        ------
        None.

        Outputs
        -------
        Dict[str, Any]
            Dictionary containing OS, Python, packages, CPU, RAM, and GPU info.

        Important Assumptions
        ---------------------
        Assumes standard Python libraries and optional torch/tensorflow installations.
        """
        env: Dict[str, Any] = {
            "os": platform.platform(),
            "python_version": sys.version,
            "executable": sys.executable,
        }

        # Check key scientific libraries
        for pkg in ["pandas", "numpy", "torch", "tensorflow", "sklearn", "scipy"]:
            try:
                mod = __import__(pkg)
                env[f"{pkg}_version"] = getattr(mod, "__version__", "Installed")
            except ImportError:
                env[f"{pkg}_version"] = "NOT INSTALLED"

        # Hardware metrics
        try:
            import psutil
            mem = psutil.virtual_memory()
            env["ram_total_gb"] = round(mem.total / (1024 ** 3), 2)
            env["ram_available_gb"] = round(mem.available / (1024 ** 3), 2)
            env["cpu_count_logical"] = psutil.cpu_count(logical=True)
            env["cpu_count_physical"] = psutil.cpu_count(logical=False)
        except ImportError:
            env["ram_total_gb"] = "Unknown (psutil not installed)"
            env["ram_available_gb"] = "Unknown"
            env["cpu_count_logical"] = os.cpu_count()
            env["cpu_count_physical"] = "Unknown"

        # GPU / CUDA detection
        try:
            import torch
            env["cuda_available"] = torch.cuda.is_available()
            env["cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
            env["gpu_model"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
        except ImportError:
            env["cuda_available"] = False
            env["cuda_device_count"] = 0
            env["gpu_model"] = "PyTorch not available to query GPU"

        return env

    @staticmethod
    def detect_label_column(headers: List[str]) -> Tuple[int, str]:
        """Identify the label column index and exact string in header.

        Purpose
        -------
        Locates the Ground Truth label column irrespective of leading/trailing spaces.

        Inputs
        ------
        headers : List[str]
            Raw column header strings from CSV.

        Outputs
        -------
        Tuple[int, str]
            (column_index, exact_column_header)

        Important Assumptions
        ---------------------
        Assumes label column name contains 'label' (case-insensitive, stripped).
        """
        for idx, col in enumerate(headers):
            if col.strip().lower() == "label":
                return idx, col
        raise ValueError(f"No Label column found in headers: {headers}")

    def audit_single_file(
        self, file_path: Path, dataset_name: str
    ) -> Tuple[FileMetadata, Counter, Dict[str, Any], Counter]:
        """Perform a streaming audit on a single CSV file.

        Purpose
        -------
        Collects file-level dimensions, exact column names, label frequency,
        duplicate counts, missing/inf values, and temporal samples in one streaming pass.

        Inputs
        ------
        file_path : Path
            Path to the CSV file.
        dataset_name : str
            Identifier ('CICIDS2017' or 'CSE-CIC-IDS2018').

        Outputs
        -------
        Tuple[FileMetadata, Counter, Dict[str, Any], Counter]
            Metadata dataclass, label distribution Counter, data quality metrics dict,
            and sample temporal values Counter.

        Important Assumptions
        ---------------------
        Reads files in binary/utf-8 with replacement to tolerate encoding anomalies
        like Windows-1252 symbols in CICIDS2017 without crashing.
        """
        file_size_bytes = file_path.stat().st_size
        file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
        relative_path = str(file_path.relative_to(file_path.parent.parent))

        label_counts: Counter = Counter()
        temporal_samples: Counter = Counter()
        seen_hashes: Set[bytes] = set()
        in_file_dups = 0
        repeated_header_count = 0
        total_data_rows = 0

        nan_count = 0
        pos_inf_count = 0
        neg_inf_count = 0

        # Read first line to extract headers
        with open(file_path, "r", encoding="utf-8", errors="replace") as fp:
            reader = csv.reader(fp)
            raw_headers = next(reader)

        num_cols = len(raw_headers)
        label_idx, label_col_name = self.detect_label_column(raw_headers)

        has_timestamp = any(h.strip().lower() == "timestamp" for h in raw_headers)
        ts_idx = -1
        if has_timestamp:
            for i, h in enumerate(raw_headers):
                if h.strip().lower() == "timestamp":
                    ts_idx = i
                    break

        # Streaming line-by-line pass
        with open(file_path, "r", encoding="utf-8", errors="replace") as fp:
            reader = csv.reader(fp)
            _ = next(reader)  # skip header

            for row in reader:
                if not row:
                    continue

                # Check if this row is an embedded duplicate header row
                if len(row) > label_idx and row[label_idx].strip().lower() == "label":
                    repeated_header_count += 1
                    continue

                total_data_rows += 1

                # Hash line content for exact duplicate detection
                line_repr = "\t".join(row).encode("utf-8")
                h = hashlib.md5(line_repr).digest()[:8]
                if h in seen_hashes:
                    in_file_dups += 1
                else:
                    seen_hashes.add(h)

                # Label extraction (preserve exact original string)
                if len(row) > label_idx:
                    lbl = row[label_idx].strip()
                    label_counts[lbl] += 1

                # Temporal extraction
                if ts_idx >= 0 and len(row) > ts_idx:
                    ts_val = row[ts_idx].strip()
                    if total_data_rows <= 10 or total_data_rows % 50000 == 0:
                        temporal_samples[ts_val] += 1

                # Quality: Check for NaN and Infs in row tokens
                for token in row:
                    t_lower = token.strip().lower()
                    if t_lower in ("nan", ""):
                        nan_count += 1
                    elif "infinity" in t_lower or "inf" in t_lower:
                        if "-" in t_lower:
                            neg_inf_count += 1
                        else:
                            pos_inf_count += 1

        dup_pct = round((in_file_dups / total_data_rows * 100), 2) if total_data_rows > 0 else 0.0

        metadata = FileMetadata(
            dataset_name=dataset_name,
            filename=file_path.name,
            relative_path=relative_path,
            file_size_bytes=file_size_bytes,
            file_size_mb=file_size_mb,
            extension=file_path.suffix,
            num_rows=total_data_rows,
            num_cols=num_cols,
            column_names=raw_headers,
            label_column=label_col_name,
            has_repeated_headers=(repeated_header_count > 0),
            repeated_header_count=repeated_header_count,
            in_file_duplicates=in_file_dups,
            in_file_duplicate_pct=dup_pct,
        )

        quality_stats = {
            "nan_count": nan_count,
            "pos_inf_count": pos_inf_count,
            "neg_inf_count": neg_inf_count,
            "total_non_finite": nan_count + pos_inf_count + neg_inf_count,
        }

        return metadata, label_counts, quality_stats, temporal_samples
