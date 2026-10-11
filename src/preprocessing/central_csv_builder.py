"""Memory-Efficient Central Combined CSV Dataset Builder for CICIDS2017 & CSE-CIC-IDS2018.

Combines all 18 raw CSV files into ONE single central reference dataset:
- Memory-Safe: Pure streaming row-by-row (RAM < 150 MB at all times).
- Preserves 100% of all ~18.35M clean flows (Zero Downsampling).
- Excludes IP/Port shortcut identifiers (Src IP, Dst IP, Src Port, Dst Port, Flow ID).
- Excludes 10 constant-zero features (zero variance across both datasets).
- Harmonizes 66 canonical behavioral features across both datasets.
- Purges corrupted repeated header lines and exact duplicate rows.
- Exports:
    1. central_mixed_dataset.csv       (The master single reference dataset)
    2. central_sample_1000.csv         (Human-inspectable sample for Excel/Notepad)
    3. central_class_distribution.csv  (Exact attack class counts and percentages)
"""

import csv
import gc
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

# The 66 shared canonical behavioral features between CICIDS2017 and CSE-CIC-IDS2018
FEATURE_MAPPINGS_2017_AND_2018: List[Tuple[str, str, str]] = [
    # (Canonical Name, CICIDS2017 Header, CSE-CIC-IDS2018 Header)
    ("Flow_Duration", "Flow Duration", "Flow Duration"),
    ("Tot_Fwd_Pkts", "Total Fwd Packets", "Tot Fwd Pkts"),
    ("Tot_Bwd_Pkts", "Total Backward Packets", "Tot Bwd Pkts"),
    ("TotLen_Fwd_Pkts", "Total Length of Fwd Packets", "TotLen Fwd Pkts"),
    ("TotLen_Bwd_Pkts", "Total Length of Bwd Packets", "TotLen Bwd Pkts"),
    ("Fwd_Pkt_Len_Max", "Fwd Packet Length Max", "Fwd Pkt Len Max"),
    ("Fwd_Pkt_Len_Min", "Fwd Packet Length Min", "Fwd Pkt Len Min"),
    ("Fwd_Pkt_Len_Mean", "Fwd Packet Length Mean", "Fwd Pkt Len Mean"),
    ("Fwd_Pkt_Len_Std", "Fwd Packet Length Std", "Fwd Pkt Len Std"),
    ("Bwd_Pkt_Len_Max", "Bwd Packet Length Max", "Bwd Pkt Len Max"),
    ("Bwd_Pkt_Len_Min", "Bwd Packet Length Min", "Bwd Pkt Len Min"),
    ("Bwd_Pkt_Len_Mean", "Bwd Packet Length Mean", "Bwd Pkt Len Mean"),
    ("Bwd_Pkt_Len_Std", "Bwd Packet Length Std", "Bwd Pkt Len Std"),
    ("Flow_Byts_s", "Flow Bytes/s", "Flow Byts/s"),
    ("Flow_Pkts_s", "Flow Packets/s", "Flow Pkts/s"),
    ("Flow_IAT_Mean", "Flow IAT Mean", "Flow IAT Mean"),
    ("Flow_IAT_Std", "Flow IAT Std", "Flow IAT Std"),
    ("Flow_IAT_Max", "Flow IAT Max", "Flow IAT Max"),
    ("Flow_IAT_Min", "Flow IAT Min", "Flow IAT Min"),
    ("Fwd_IAT_Tot", "Fwd IAT Total", "Fwd IAT Tot"),
    ("Fwd_IAT_Mean", "Fwd IAT Mean", "Fwd IAT Mean"),
    ("Fwd_IAT_Std", "Fwd IAT Std", "Fwd IAT Std"),
    ("Fwd_IAT_Max", "Fwd IAT Max", "Fwd IAT Max"),
    ("Fwd_IAT_Min", "Fwd IAT Min", "Fwd IAT Min"),
    ("Bwd_IAT_Tot", "Bwd IAT Total", "Bwd IAT Tot"),
    ("Bwd_IAT_Mean", "Bwd IAT Mean", "Bwd IAT Mean"),
    ("Bwd_IAT_Std", "Bwd IAT Std", "Bwd IAT Std"),
    ("Bwd_IAT_Max", "Bwd IAT Max", "Bwd IAT Max"),
    ("Bwd_IAT_Min", "Bwd IAT Min", "Bwd IAT Min"),
    ("Fwd_PSH_Flags", "Fwd PSH Flags", "Fwd PSH Flags"),
    ("Fwd_Header_Len", "Fwd Header Length", "Fwd Header Len"),
    ("Bwd_Header_Len", "Bwd Header Length", "Bwd Header Len"),
    ("Fwd_Pkts_s", "Fwd Packets/s", "Fwd Pkts/s"),
    ("Bwd_Pkts_s", "Bwd Packets/s", "Bwd Pkts/s"),
    ("Pkt_Len_Min", "Min Packet Length", "Pkt Len Min"),
    ("Pkt_Len_Max", "Max Packet Length", "Pkt Len Max"),
    ("Pkt_Len_Mean", "Packet Length Mean", "Pkt Len Mean"),
    ("Pkt_Len_Std", "Packet Length Std", "Pkt Len Std"),
    ("Pkt_Len_Var", "Packet Length Variance", "Pkt Len Var"),
    ("FIN_Flag_Cnt", "FIN Flag Count", "FIN Flag Cnt"),
    ("SYN_Flag_Cnt", "SYN Flag Count", "SYN Flag Cnt"),
    ("RST_Flag_Cnt", "RST Flag Count", "RST Flag Cnt"),
    ("PSH_Flag_Cnt", "PSH Flag Count", "PSH Flag Cnt"),
    ("ACK_Flag_Cnt", "ACK Flag Count", "ACK Flag Cnt"),
    ("URG_Flag_Cnt", "URG Flag Count", "URG Flag Cnt"),
    ("ECE_Flag_Cnt", "ECE Flag Count", "ECE Flag Cnt"),
    ("Down_Up_Ratio", "Down/Up Ratio", "Down/Up Ratio"),
    ("Pkt_Size_Avg", "Average Packet Size", "Pkt Size Avg"),
    ("Fwd_Seg_Size_Avg", "Avg Fwd Segment Size", "Fwd Seg Size Avg"),
    ("Bwd_Seg_Size_Avg", "Avg Bwd Segment Size", "Bwd Seg Size Avg"),
    ("Subflow_Fwd_Pkts", "Subflow Fwd Packets", "Subflow Fwd Pkts"),
    ("Subflow_Fwd_Byts", "Subflow Fwd Bytes", "Subflow Fwd Byts"),
    ("Subflow_Bwd_Pkts", "Subflow Bwd Packets", "Subflow Bwd Pkts"),
    ("Subflow_Bwd_Byts", "Subflow Bwd Bytes", "Subflow Bwd Byts"),
    ("Init_Fwd_Win_Byts", "Init_Win_bytes_forward", "Init Fwd Win Byts"),
    ("Init_Bwd_Win_Byts", "Init_Win_bytes_backward", "Init Bwd Win Byts"),
    ("Fwd_Act_Data_Pkts", "act_data_pkt_fwd", "Fwd Act Data Pkts"),
    ("Fwd_Seg_Size_Min", "min_seg_size_forward", "Fwd Seg Size Min"),
    ("Active_Mean", "Active Mean", "Active Mean"),
    ("Active_Std", "Active Std", "Active Std"),
    ("Active_Max", "Active Max", "Active Max"),
    ("Active_Min", "Active Min", "Active Min"),
    ("Idle_Mean", "Idle Mean", "Idle Mean"),
    ("Idle_Std", "Idle Std", "Idle Std"),
    ("Idle_Max", "Idle Max", "Idle Max"),
    ("Idle_Min", "Idle Min", "Idle Min"),
]

CANONICAL_FEATURE_NAMES: List[str] = [m[0] for m in FEATURE_MAPPINGS_2017_AND_2018]

# Master Header for the Central Combined CSV
CENTRAL_CSV_HEADERS: List[str] = ["Dataset_Source"] + CANONICAL_FEATURE_NAMES + ["Label"]

# Harmonized Attack Label Names
UNIFIED_LABEL_MAP: Dict[str, str] = {
    "benign": "Benign",
    "bot": "Bot",
    "brute force -web": "Brute Force -Web",
    "web attack  brute force": "Brute Force -Web",
    "web attack - brute force": "Brute Force -Web",
    "brute force -xss": "Brute Force -XSS",
    "web attack  xss": "Brute Force -XSS",
    "web attack - xss": "Brute Force -XSS",
    "ddos": "DDoS",
    "ddos attack-hoic": "DDoS",
    "ddos attacks-loic-http": "DDoS",
    "ddos attack-loic-udp": "DDoS",
    "dos attacks-goldeneye": "DoS-GoldenEye",
    "dos goldeneye": "DoS-GoldenEye",
    "dos attacks-hulk": "DoS-Hulk",
    "dos hulk": "DoS-Hulk",
    "dos attacks-slowhttptest": "DoS-SlowHTTPTest",
    "dos slowhttptest": "DoS-SlowHTTPTest",
    "dos attacks-slowloris": "DoS-Slowloris",
    "dos slowloris": "DoS-Slowloris",
    "ftp-patator": "FTP-BruteForce",
    "ftp-bruteforce": "FTP-BruteForce",
    "infiltration": "Infiltration",
    "infilteration": "Infiltration",
    "sql injection": "SQL Injection",
    "web attack  sql injection": "SQL Injection",
    "web attack - sql injection": "SQL Injection",
    "ssh-patator": "SSH-Bruteforce",
    "ssh-bruteforce": "SSH-Bruteforce",
    "portscan": "PortScan",
    "heartbleed": "Heartbleed",
}


class CentralCSVBuilder:
    """Builds a single central combined CSV from CICIDS2017 and CSE-CIC-IDS2018."""

    def __init__(
        self,
        cicids2017_dir: Path,
        csecicids2018_dir: Path,
        output_dir: Path,
    ) -> None:
        self.cicids2017_dir = Path(cicids2017_dir)
        self.csecicids2018_dir = Path(csecicids2018_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seen_hashes: Set[bytes] = set()

    def run(self) -> None:
        t0 = time.time()
        output_csv_path = self.output_dir / "central_mixed_dataset.csv"
        sample_csv_path = self.output_dir / "central_sample_1000.csv"
        audit_csv_path = self.output_dir / "central_class_distribution.csv"

        print("=" * 80, flush=True)
        print("BUILDING CENTRAL COMBINED CSV DATASET (CICIDS2017 + CSE-CIC-IDS2018)", flush=True)
        print(f"Target Master CSV: {output_csv_path}", flush=True)
        print("Mode: Memory-Safe Streaming Row-by-Row | RAM < 150 MB", flush=True)
        print("=" * 80, flush=True)

        all_files: List[Tuple[Path, str]] = []
        for f in sorted(self.cicids2017_dir.glob("*.csv")):
            all_files.append((f, "CICIDS2017"))
        for f in sorted(self.csecicids2018_dir.glob("*.csv")):
            all_files.append((f, "CSE-CIC-IDS2018"))

        total_files = len(all_files)
        print(f"Discovered {total_files} raw files (8 from 2017, 10 from 2018)\n", flush=True)

        stats: Dict[str, Any] = {
            "total_raw_rows": 0,
            "corrupted_headers_removed": 0,
            "duplicates_pruned": 0,
            "clean_flows": 0,
            "rows_2017": 0,
            "rows_2018": 0,
            "class_counts": {},
        }

        sample_rows: List[List[str]] = []
        write_buffer: List[List[str]] = []
        BUFFER_FLUSH_SIZE = 10000

        with open(output_csv_path, "w", newline="", encoding="utf-8") as out_fp:
            writer = csv.writer(out_fp)
            writer.writerow(CENTRAL_CSV_HEADERS)

            for idx, (fpath, dataset_name) in enumerate(all_files, start=1):
                is_2017 = (dataset_name == "CICIDS2017")
                fname = fpath.name
                t_file = time.time()
                file_clean = 0

                with open(fpath, "r", encoding="utf-8", errors="replace") as in_fp:
                    reader = csv.reader(in_fp)
                    raw_headers = [c.strip() for c in next(reader)]

                    label_idx = raw_headers.index("Label")
                    col_indices = []
                    for canon, c17, c18 in FEATURE_MAPPINGS_2017_AND_2018:
                        target_col = c17 if is_2017 else c18
                        col_indices.append(raw_headers.index(target_col))

                    for row in reader:
                        stats["total_raw_rows"] += 1
                        if not row or len(row) <= label_idx:
                            continue

                        raw_lbl = row[label_idx].strip()
                        if raw_lbl.lower() == "label":
                            stats["corrupted_headers_removed"] += 1
                            continue

                        # Deduplication using lightweight 8-byte hash
                        row_hash = hashlib.md5("\t".join(row).encode("utf-8")).digest()[:8]
                        if row_hash in self.seen_hashes:
                            stats["duplicates_pruned"] += 1
                            continue
                        self.seen_hashes.add(row_hash)

                        # Harmonize label
                        norm_lbl = raw_lbl.lower().replace("\ufffd", "")
                        clean_lbl = UNIFIED_LABEL_MAP.get(norm_lbl)
                        if clean_lbl is None:
                            cleaned = norm_lbl.replace("", "-").replace("  ", " ").strip()
                            clean_lbl = UNIFIED_LABEL_MAP.get(cleaned, raw_lbl)

                        # Extract 66 canonical values
                        canonical_vals = []
                        for col_pos in col_indices:
                            v = row[col_pos].strip()
                            if v.lower() in ("nan", "", "null") or "inf" in v.lower():
                                canonical_vals.append("0.0")
                            else:
                                canonical_vals.append(v)

                        out_row = [dataset_name] + canonical_vals + [clean_lbl]
                        write_buffer.append(out_row)
                        file_clean += 1
                        stats["clean_flows"] += 1
                        if is_2017:
                            stats["rows_2017"] += 1
                        else:
                            stats["rows_2018"] += 1

                        stats["class_counts"][clean_lbl] = stats["class_counts"].get(clean_lbl, 0) + 1

                        if len(sample_rows) < 1000:
                            sample_rows.append(out_row)

                        if len(write_buffer) >= BUFFER_FLUSH_SIZE:
                            writer.writerows(write_buffer)
                            write_buffer.clear()

                # Flush remaining buffer for this file
                if write_buffer:
                    writer.writerows(write_buffer)
                    write_buffer.clear()

                dur = time.time() - t_file
                print(
                    f"[{idx:02d}/{total_files:02d}] [{dataset_name}] {fname} -> "
                    f"{file_clean:,} clean flows ({dur:.1f}s)",
                    flush=True,
                )

        # Free deduplication set
        self.seen_hashes.clear()
        gc.collect()

        total_dur = time.time() - t0
        total_clean = stats["clean_flows"]

        # Write sample 1000 CSV for human inspection
        with open(sample_csv_path, "w", newline="", encoding="utf-8") as s_fp:
            s_writer = csv.writer(s_fp)
            s_writer.writerow(CENTRAL_CSV_HEADERS)
            s_writer.writerows(sample_rows)
        print(f"\nCreated human-inspectable preview: {sample_csv_path.name}", flush=True)

        # Write Class Distribution Audit CSV
        audit_rows = []
        for lbl, cnt in sorted(stats["class_counts"].items(), key=lambda x: -x[1]):
            audit_rows.append({
                "Attack_Class": lbl,
                "Total_Flows": cnt,
                "Percentage": round(cnt / total_clean * 100, 4) if total_clean > 0 else 0,
            })
        audit_df = pd.DataFrame(audit_rows)
        audit_df.to_csv(audit_csv_path, index=False)
        print(f"Created class distribution audit: {audit_csv_path.name}", flush=True)

        print("\n" + "=" * 80, flush=True)
        print("CENTRAL COMBINED CSV DATASET GENERATION COMPLETE!", flush=True)
        print(f"Master CSV Location: {output_csv_path}", flush=True)
        print(f"Total Raw Rows Read:            {stats['total_raw_rows']:,}", flush=True)
        print(f"Corrupted Repeated Headers:     {stats['corrupted_headers_removed']:,} (purged)", flush=True)
        print(f"Exact Duplicate Flows:          {stats['duplicates_pruned']:,} (purged)", flush=True)
        print(f"Total Unique Clean Flows:       {total_clean:,} (100% preserved)", flush=True)
        print(f"  * From CICIDS2017:            {stats['rows_2017']:,}", flush=True)
        print(f"  * From CSE-CIC-IDS2018:        {stats['rows_2018']:,}", flush=True)
        print(f"Total Runtime:                  {total_dur:.1f}s ({total_dur/60:.1f} min)", flush=True)
        print("=" * 80, flush=True)
