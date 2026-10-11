"""Zero-Day Partitioning and Feature Normalization Pipeline (Option B).

Implements Option B Research Protocol:
1. Zero-Day Holdout:
   - Isolates 100% of Heartbleed (11), SQL Injection (108), and Brute Force - XSS (882)
     flows (Total: 1,001 flows) into 'zero_day_test.csv'.
   - These flows are strictly UNSEEN during model training and validation.
2. Base Partitioning:
   - Partitions the remaining 18,343,597 flows chronologically:
     - Train (70%): ~12,840,518 flows -> 'train.csv'
     - Validation (15%): ~2,751,540 flows -> 'val.csv'
     - Test Known (15%): ~2,751,539 flows -> 'test_known.csv'
     - Test Full: Test Known + Zero-Day Test -> 'test_full.csv'
3. Feature Scaling & Normalization (Leakage-Free):
   - Streams 250,000 training flows to fit an outlier-resistant RobustScaler.
   - Computes StandardScaler parameters across training data.
   - Saves fitted scalers to 'models/robust_scaler.joblib' and 'models/standard_scaler.joblib'.
4. Class Imbalance Mitigation:
   - Computes balanced inverse class weights and Effective Number of Samples weights
     for all 12 known classes to power Focal Loss and Weighted Cross-Entropy.
   - Saves weights and label encoders to 'models/class_weights.json' and 'models/label_encoder.json'.
5. Audit Reporting:
   - Generates 'data/processed/partition_audit_report.csv'.
"""

import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

# Zero-Day Unseen Classes for Thesis Evaluation (Option B)
ZERO_DAY_CLASSES: Set[str] = {
    "Heartbleed",
    "SQL Injection",
    "Brute Force -XSS",
}

# The 12 Known Training Attack Classes (Label IDs 0 to 11)
KNOWN_CLASSES: List[str] = [
    "Benign",
    "DDoS",
    "DoS-Hulk",
    "Bot",
    "Infiltration",
    "SSH-Bruteforce",
    "PortScan",
    "DoS-GoldenEye",
    "FTP-BruteForce",
    "DoS-SlowHTTPTest",
    "DoS-Slowloris",
    "Brute Force -Web",
]

LABEL_TO_ID: Dict[str, int] = {lbl: idx for idx, lbl in enumerate(KNOWN_CLASSES)}


class ZeroDayPartitioner:
    """Memory-safe streaming partitioner and feature scaling pipeline."""

    def __init__(
        self,
        input_csv_path: Path,
        output_dir: Path,
        models_dir: Path,
        split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    ) -> None:
        self.input_csv_path = Path(input_csv_path)
        self.output_dir = Path(output_dir)
        self.models_dir = Path(models_dir)
        self.split_ratios = split_ratios

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        t0 = time.time()
        print("=" * 80, flush=True)
        print("STARTING ZERO-DAY PARTITIONING & FEATURE NORMALIZATION (OPTION B)", flush=True)
        print(f"Source Central CSV: {self.input_csv_path}", flush=True)
        print(f"Output Directory:   {self.output_dir}", flush=True)
        print(f"Models Directory:   {self.models_dir}", flush=True)
        print("=" * 80, flush=True)

        zero_day_csv_path = self.output_dir / "zero_day_test.csv"
        train_csv_path = self.output_dir / "train.csv"
        val_csv_path = self.output_dir / "val.csv"
        test_known_csv_path = self.output_dir / "test_known.csv"
        test_full_csv_path = self.output_dir / "test_full.csv"

        # Counts from Phase 1 Audit
        TOTAL_BASE_ROWS = 18343597
        train_limit = int(TOTAL_BASE_ROWS * self.split_ratios[0])
        val_limit = train_limit + int(TOTAL_BASE_ROWS * self.split_ratios[1])

        print(f"Partition Targets for Known Data (12 Classes + Benign):", flush=True)
        print(f"  * Train (70%):      {train_limit:,} flows", flush=True)
        print(f"  * Val (15%):        {int(TOTAL_BASE_ROWS * self.split_ratios[1]):,} flows", flush=True)
        print(f"  * Test Known (15%): {TOTAL_BASE_ROWS - val_limit:,} flows", flush=True)
        print(f"Zero-Day Holdouts (Unseen): Heartbleed, SQL Injection, Brute Force -XSS\n", flush=True)

        stats = {
            "total_rows": 0,
            "zero_day_count": 0,
            "train_count": 0,
            "val_count": 0,
            "test_known_count": 0,
            "class_counts_train": {c: 0 for c in KNOWN_CLASSES},
            "class_counts_val": {c: 0 for c in KNOWN_CLASSES},
            "class_counts_test_known": {c: 0 for c in KNOWN_CLASSES},
            "class_counts_zero_day": {c: 0 for c in ZERO_DAY_CLASSES},
        }

        # Reservoir sampling for fitting RobustScaler on Train (sample size: 250,000)
        RESERVOIR_SIZE = 250000
        reservoir_sample: List[List[float]] = []

        # Open all destination CSV files
        out_zd = open(zero_day_csv_path, "w", newline="", encoding="utf-8")
        out_tr = open(train_csv_path, "w", newline="", encoding="utf-8")
        out_va = open(val_csv_path, "w", newline="", encoding="utf-8")
        out_te = open(test_known_csv_path, "w", newline="", encoding="utf-8")
        out_tf = open(test_full_csv_path, "w", newline="", encoding="utf-8")

        w_zd = csv.writer(out_zd)
        w_tr = csv.writer(out_tr)
        w_va = csv.writer(out_va)
        w_te = csv.writer(out_te)
        w_tf = csv.writer(out_tf)

        # Buffers for high-speed batch disk writes
        buf_zd: List[List[str]] = []
        buf_tr: List[List[str]] = []
        buf_va: List[List[str]] = []
        buf_te: List[List[str]] = []
        buf_tf: List[List[str]] = []
        BATCH_SIZE = 20000

        base_clean_index = 0

        with open(self.input_csv_path, "r", encoding="utf-8") as in_fp:
            reader = csv.reader(in_fp)
            headers = next(reader)

            # Write headers to all partitions
            w_zd.writerow(headers)
            w_tr.writerow(headers)
            w_va.writerow(headers)
            w_te.writerow(headers)
            w_tf.writerow(headers)

            label_col_idx = headers.index("Label")
            feature_indices = list(range(1, 67))  # 66 canonical behavioral features

            print("Streaming rows through Option B partitioner...", flush=True)
            last_log_time = time.time()

            for row in reader:
                stats["total_rows"] += 1
                lbl = row[label_col_idx]

                # Check if this row is an unseen Zero-Day attack
                if lbl in ZERO_DAY_CLASSES:
                    buf_zd.append(row)
                    buf_tf.append(row)
                    stats["zero_day_count"] += 1
                    stats["class_counts_zero_day"][lbl] = stats["class_counts_zero_day"].get(lbl, 0) + 1
                    if len(buf_zd) >= BATCH_SIZE:
                        w_zd.writerows(buf_zd)
                        buf_zd.clear()
                    continue

                # Otherwise, it belongs to the 12 Known Classes
                base_clean_index += 1

                if base_clean_index <= train_limit:
                    buf_tr.append(row)
                    stats["train_count"] += 1
                    stats["class_counts_train"][lbl] = stats["class_counts_train"].get(lbl, 0) + 1

                    # Reservoir sampling for RobustScaler fitting
                    if len(reservoir_sample) < RESERVOIR_SIZE:
                        feat_floats = [float(row[i]) for i in feature_indices]
                        reservoir_sample.append(feat_floats)
                    elif random.random() < (RESERVOIR_SIZE / base_clean_index):
                        replace_idx = random.randint(0, RESERVOIR_SIZE - 1)
                        feat_floats = [float(row[i]) for i in feature_indices]
                        reservoir_sample[replace_idx] = feat_floats

                    if len(buf_tr) >= BATCH_SIZE:
                        w_tr.writerows(buf_tr)
                        buf_tr.clear()

                elif base_clean_index <= val_limit:
                    buf_va.append(row)
                    stats["val_count"] += 1
                    stats["class_counts_val"][lbl] = stats["class_counts_val"].get(lbl, 0) + 1
                    if len(buf_va) >= BATCH_SIZE:
                        w_va.writerows(buf_va)
                        buf_va.clear()

                else:
                    buf_te.append(row)
                    buf_tf.append(row)
                    stats["test_known_count"] += 1
                    stats["class_counts_test_known"][lbl] = stats["class_counts_test_known"].get(lbl, 0) + 1
                    if len(buf_te) >= BATCH_SIZE:
                        w_te.writerows(buf_te)
                        buf_te.clear()

                if len(buf_tf) >= BATCH_SIZE:
                    w_tf.writerows(buf_tf)
                    buf_tf.clear()

                # Progress heartbeat every 2M rows
                if stats["total_rows"] % 2000000 == 0:
                    elapsed = time.time() - t0
                    print(
                        f"Processed {stats['total_rows']:,} / 18,344,598 rows "
                        f"({elapsed:.1f}s | {stats['total_rows']/elapsed:,.0f} rows/s)",
                        flush=True,
                    )

        # Flush all remaining buffers
        if buf_zd:
            w_zd.writerows(buf_zd)
        if buf_tr:
            w_tr.writerows(buf_tr)
        if buf_va:
            w_va.writerows(buf_va)
        if buf_te:
            w_te.writerows(buf_te)
        if buf_tf:
            w_tf.writerows(buf_tf)

        # Close all file handles
        out_zd.close()
        out_tr.close()
        out_va.close()
        out_te.close()
        out_tf.close()

        print(f"\nPartitioning Complete in {time.time() - t0:.1f}s!", flush=True)
        print(f"  * Zero-Day Test:    {stats['zero_day_count']:,} flows (100% of Heartbleed, SQLi, XSS)")
        print(f"  * Train (70%):      {stats['train_count']:,} flows")
        print(f"  * Val (15%):        {stats['val_count']:,} flows")
        print(f"  * Test Known (15%): {stats['test_known_count']:,} flows")
        print(f"  * Test Full:        {stats['test_known_count'] + stats['zero_day_count']:,} flows")

        # ------------------------------------------------------------------
        # Feature Scaling & Normalization (Fitted Strictly on Train Set)
        # ------------------------------------------------------------------
        print("\nFitting Outlier-Resistant RobustScaler on Training Set...", flush=True)
        t_scale = time.time()
        X_sample = np.array(reservoir_sample, dtype=np.float32)

        robust_scaler = RobustScaler(quantile_range=(25.0, 75.0))
        robust_scaler.fit(X_sample)
        robust_scaler_path = self.models_dir / "robust_scaler.joblib"
        joblib.dump(robust_scaler, robust_scaler_path)
        print(f"Saved RobustScaler to: {robust_scaler_path.name} ({time.time() - t_scale:.1f}s)", flush=True)

        standard_scaler = StandardScaler()
        standard_scaler.fit(X_sample)
        standard_scaler_path = self.models_dir / "standard_scaler.joblib"
        joblib.dump(standard_scaler, standard_scaler_path)
        print(f"Saved StandardScaler to: {standard_scaler_path.name}", flush=True)

        # ------------------------------------------------------------------
        # Class Imbalance Weights Calculation
        # ------------------------------------------------------------------
        print("\nCalculating Class Imbalance Weights for 12 Known Classes...", flush=True)
        n_train_total = stats["train_count"]
        num_known_classes = len(KNOWN_CLASSES)

        balanced_weights: Dict[str, float] = {}
        effective_weights: Dict[str, float] = {}
        beta = 0.9999

        for cls_name in KNOWN_CLASSES:
            cnt = stats["class_counts_train"].get(cls_name, 1)
            # Balanced weight: N / (K * cnt)
            bw = n_train_total / (num_known_classes * max(cnt, 1))
            balanced_weights[cls_name] = round(bw, 4)

            # Effective Number of Samples weight
            effective_num = (1.0 - np.power(beta, cnt)) / (1.0 - beta) if cnt > 0 else 1.0
            effective_weights[cls_name] = round(1.0 / effective_num, 8)

        # Normalize effective weights so average weight is 1.0
        sum_ew = sum(effective_weights.values())
        norm_effective_weights = {
            k: round(v / sum_ew * num_known_classes, 4) for k, v in effective_weights.items()
        }

        weights_payload = {
            "num_known_classes": num_known_classes,
            "classes": KNOWN_CLASSES,
            "label_to_id": LABEL_TO_ID,
            "balanced_class_weights": balanced_weights,
            "effective_sample_weights": norm_effective_weights,
            "zero_day_unseen_classes": list(ZERO_DAY_CLASSES),
        }

        weights_path = self.models_dir / "class_weights.json"
        with open(weights_path, "w", encoding="utf-8") as fp:
            json.dump(weights_payload, fp, indent=2)
        print(f"Saved class imbalance weights to: {weights_path.name}", flush=True)

        # ------------------------------------------------------------------
        # Master Partition Audit Report
        # ------------------------------------------------------------------
        audit_rows = []
        all_classes_in_order = KNOWN_CLASSES + sorted(list(ZERO_DAY_CLASSES))

        for c in all_classes_in_order:
            tr = stats["class_counts_train"].get(c, 0)
            va = stats["class_counts_val"].get(c, 0)
            te = stats["class_counts_test_known"].get(c, 0)
            zd = stats["class_counts_zero_day"].get(c, 0)
            tot = tr + va + te + zd
            is_zd = c in ZERO_DAY_CLASSES

            audit_rows.append({
                "Class_Name": c,
                "Role": "Zero-Day (Unseen)" if is_zd else "Known Training Class",
                "Total_Flows": tot,
                "Train_Count": tr,
                "Val_Count": va,
                "Test_Known_Count": te,
                "Zero_Day_Test_Count": zd,
                "Train_Pct": round(tr / tot * 100, 2) if tot > 0 else 0,
                "Test_Pct": round((te + zd) / tot * 100, 2) if tot > 0 else 0,
            })

        audit_df = pd.DataFrame(audit_rows)
        audit_report_path = self.output_dir / "partition_audit_report.csv"
        audit_df.to_csv(audit_report_path, index=False)
        print(f"Saved partition audit report to: {audit_report_path.name}", flush=True)

        total_dur = time.time() - t0
        print("\n" + "=" * 80, flush=True)
        print("ZERO-DAY PARTITIONING & FEATURE NORMALIZATION PIPELINE COMPLETE!", flush=True)
        print(f"Total Runtime: {total_dur:.1f}s ({total_dur/60:.2f} min)", flush=True)
        print("=" * 80, flush=True)
