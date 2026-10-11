"""Memory-Safe Streaming Sequence Dataset & DataLoader for CNN-LSTM Intrusion Detection.

Engineered for 8 GB RAM constraints:
- Pure streaming chunk ingestion with on-the-fly RobustScaler normalization.
- Formulates sliding temporal windows of length L = 10 consecutive flows.
- Yields normalized 3D tensors: [Batch_Size, Seq_Len=10, Features=66].
- Keeps RAM footprint strictly < 100 MB during training.
"""

import csv
import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

# 12 Known Class Label-to-ID Mapping (+ Negative Tracking IDs for Zero-Day)
DEFAULT_LABEL_MAP: Dict[str, int] = {
    "Benign": 0,
    "DDoS": 1,
    "DoS-Hulk": 2,
    "Bot": 3,
    "Infiltration": 4,
    "SSH-Bruteforce": 5,
    "PortScan": 6,
    "DoS-GoldenEye": 7,
    "FTP-BruteForce": 8,
    "DoS-SlowHTTPTest": 9,
    "DoS-Slowloris": 10,
    "Brute Force -Web": 11,
    # Zero-day holdout evaluation tracking IDs
    "Heartbleed": -101,
    "SQL Injection": -102,
    "Brute Force -XSS": -103,
}


class StreamingSequenceDataset(IterableDataset):
    """Memory-safe iterable PyTorch dataset that streams CSV flows into temporal sequence windows."""

    def __init__(
        self,
        csv_file_path: Path,
        scaler_path: Optional[Path] = None,
        label_map: Optional[Dict[str, int]] = None,
        seq_len: int = 10,
        num_features: int = 66,
        max_rows: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.csv_file_path = Path(csv_file_path)
        self.seq_len = seq_len
        self.num_features = num_features
        self.max_rows = max_rows
        self.label_map = label_map or DEFAULT_LABEL_MAP

        # Load fitted RobustScaler if provided
        self.scaler = None
        self.center = None
        self.scale = None
        if scaler_path and Path(scaler_path).exists():
            scaler_obj = joblib.load(scaler_path)
            if hasattr(scaler_obj, "center_") and hasattr(scaler_obj, "scale_"):
                self.center = scaler_obj.center_.astype(np.float32)
                scale_arr = scaler_obj.scale_.copy().astype(np.float32)
                scale_arr[scale_arr == 0] = 1.0
                self.scale = scale_arr
            else:
                self.scaler = scaler_obj

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int]]:
        """Yields (sequence_tensor [Seq_Len, Features], target_label_id)."""
        feature_indices = list(range(1, 67))  # 66 canonical behavioral features

        window_features: List[np.ndarray] = []
        rows_read = 0

        with open(self.csv_file_path, "r", encoding="utf-8") as fp:
            reader = csv.reader(fp)
            headers = next(reader)
            label_col_idx = headers.index("Label")

            for row in reader:
                rows_read += 1
                if self.max_rows and rows_read > self.max_rows:
                    break

                # Extract and sanitize 66 numeric features
                raw_feats = np.empty(self.num_features, dtype=np.float32)
                for f_idx, col_idx in enumerate(feature_indices):
                    val_str = row[col_idx].strip()
                    try:
                        v = float(val_str)
                        # Sanitize negative TCP window artifacts (-1) to 0.0
                        raw_feats[f_idx] = max(v, 0.0) if f_idx in (54, 55) else v
                    except ValueError:
                        raw_feats[f_idx] = 0.0

                # Label encoding
                lbl_str = row[label_col_idx].strip()
                lbl_id = self.label_map.get(lbl_str, -1)

                window_features.append(raw_feats)

                # Once we reach sequence length L, yield window
                if len(window_features) == self.seq_len:
                    window_arr = np.array(window_features, dtype=np.float32)

                    # Fast vectorized RobustScaler normalization
                    if self.center is not None and self.scale is not None:
                        window_arr = (window_arr - self.center) / self.scale
                        window_arr = np.clip(window_arr, -10.0, 10.0)
                    elif self.scaler is not None:
                        window_arr = self.scaler.transform(window_arr)
                        window_arr = np.clip(window_arr, -10.0, 10.0)

                    seq_tensor = torch.from_numpy(window_arr)
                    yield seq_tensor, lbl_id

                    # Slide window by 1 time step
                    window_features.pop(0)


def create_streaming_dataloader(
    csv_file_path: Path,
    scaler_path: Optional[Path] = None,
    batch_size: int = 256,
    seq_len: int = 10,
    max_rows: Optional[int] = None,
) -> DataLoader:
    """Creates a memory-safe PyTorch DataLoader for streaming training/evaluation."""
    dataset = StreamingSequenceDataset(
        csv_file_path=csv_file_path,
        scaler_path=scaler_path,
        seq_len=seq_len,
        max_rows=max_rows,
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=0)
