# CNN–LSTM Based Anomaly Detection for Zero-Day and Low-Frequency Network Intrusions Using CICIDS2017

## Master's Thesis Research Implementation

### 1. Research Overview
This research develops and experimentally evaluates a deep learning framework combining **Convolutional Neural Networks (CNN)** and **Long Short-Term Memory Networks (LSTM)** for network intrusion/anomaly detection.

The core research objectives are:
1. **Network Intrusion/Anomaly Detection**: Extract spatial/local flow feature correlations using CNN and temporal sequence transitions using LSTM.
2. **Low-Frequency Attack Detection**: Evaluate model detection capability on rare intrusion attacks (e.g. Bot, Web Attacks).
3. **Unseen / Zero-Day-Like Attack Detection**: Scientifically evaluate model generalization against attack categories held out completely during training (e.g. Heartbleed, Infiltration).
4. **Temporal / Sequence Learning**: Investigate meaningful flow sequence construction.
5. **Cross-Dataset Generalization**: Evaluate model transferability and domain shift across intrusion detection datasets (CICIDS2017 and CSE-CIC-IDS2018).

---

### 2. Workspace Directory Structure
```
D:\NID Project\
│
├── configs\
│   └── project_config.py          # Central paths, configuration, and environment flags
│
├── src\
│   ├── data\
│   │   ├── auditor.py             # Read-only dataset discovery & audit
│   │   └── sequence_loader.py     # Bounded streaming DataLoader (RAM < 100 MB)
│   ├── preprocessing\
│   │   ├── central_csv_builder.py # Zero-loss single CSV builder (18.34M flows)
│   │   └── zero_day_partitioner.py# Option B zero-day isolator & 70/15/15 splitter
│   ├── models\
│   │   └── cnn_lstm_model.py      # DualHeadCNNLSTM (Focal Loss + Autoencoder heads)
│   ├── training\
│   │   └── trainer.py             # Cosine annealing, early stopping & heartbeat logs
│   └── evaluation\
│       └── zero_day_evaluator.py  # Anomaly calibration & zero-day detection metrics
│
├── scripts\
│   ├── generate_audit_artifacts.py # Auditing runner generating all CSV/TXT audit reports
│   ├── generate_pdf_report.py      # PDF audit report generator
│   ├── generate_technical_report_pdf.py # Official Technical Documentation PDF generator
│   ├── build_central_csv_dataset.py# CLI for unified 18.34M dataset compilation
│   ├── run_zero_day_partitioning.py# CLI for Option B zero-day test & train/val split
│   ├── train_cnn_lstm.py           # CLI trainer with streaming sequence loader
│   └── evaluate_zero_day.py        # CLI zero-day evaluator & metrics generator
│
├── notebooks\
│   └── Master_Thesis_CNN_LSTM_Colab_Kaggle.ipynb # 1-click cloud GPU training notebook
│
├── data\
│   ├── audit\                     # Forensic audit CSVs, TXT, and PDF report
│   └── processed\                 # Central dataset, Option B partitions, scaler, audit reports
│
├── models\                        # Trained checkpoints (.pt) and scaler joblib artifacts
├── results\                       # Classification reports and zero-day evaluation metrics
├── reports\
│   ├── Dataset_Audit_Report.pdf            # Phase 1 Forensic Audit PDF Report
│   ├── Technical_Implementation_Report.pdf # Official Architecture & Technical Specification PDF
│   └── Technical_Implementation_Report.md  # Markdown Source Specification
└── README.md                      # Project documentation and guidelines
```

---

### 3. Dataset Roles and Allocation
The raw datasets located in this workspace are strictly **READ-ONLY**:

* **Primary Dataset: CSE-CIC-IDS2018** (`D:\NID Project\CSE-CIC-IDS2018`)
  * **Role**: Primary dataset for core CNN-LSTM model development, temporal sequence learning, and controlled low-frequency intrusion experiments.
  * **Volume**: 10 CSV files, 16,233,002 rows (16,232,943 data flows + 59 repeated headers), ~6.58 GB.
  * **Key Properties**: Contains native wall-clock `Timestamp` column (`DD/MM/YYYY HH:MM:SS`), making it optimal for rigorous temporal windowing and chronological sequence modeling. Contains controlled low-frequency attacks (`SQL Injection`: 87, `Brute Force -XSS`: 230, `Brute Force -Web`: 611, `DDOS attack-LOIC-UDP`: 1,730).

* **Secondary / Evaluation Dataset: CICIDS2017** (`D:\NID Project\CICIDS2017 Dataset`)
  * **Role**: Reserved primarily for cross-dataset generalization testing and carefully designed unseen-attack / zero-day evaluation.
  * **Volume**: 8 CSV files, 2,830,743 flows, 884.66 MB.
  * **Key Properties**: Pristine benchmark testbed containing attacks completely absent from 2018 (e.g. `Heartbleed`: 11 flows; `PortScan`: 158,930 flows), enabling rigorous, contamination-free unseen attack evaluation.

---

### 4. Approved Research Architectural Directives
1. **Strict Deduplication (Zero Tolerance)**:
   * All exact duplicate flows (10.89% in CICIDS2017; 2.53% in CSE-CIC-IDS2018) will be pruned prior to train/test partitioning to prevent data leakage and artificial score inflation.
2. **Exclusion of Destination Port**:
   * `Destination Port` / `Dst Port` is removed from the feature set to prevent models from learning trivial shortcuts based on static target service ports (e.g., port 21 for FTP attacks).
3. **Isolated Dataset Processing (No Mixing)**:
   * Datasets will remain separate to avoid cloud vs. on-premise domain-shift conflation and maintain clean scientific transfer evaluation.
4. **Temporal Sequence Formulation**:
   * Leveraging CSE-CIC-IDS2018's explicit timestamps for chronological sequence construction in the primary CNN-LSTM pipeline.

---

### 5. Audit Artifacts
All empirical findings from the audit stage are cataloged under `data/audit/`:
* `dataset_summary.csv`: Overall summary of file count, rows, disk size, and class ratios.
* `file_statistics.csv`: Per-file breakdown of sizes, rows, columns, and in-file duplicate rates.
* `class_distribution.csv`: Overall attack label counts and rarity classifications.
* `per_file_class_distribution.csv`: Day/file-level label distributions.
* `feature_summary.csv`: In-depth analysis of all 79/80 flow features.
* `schema_report.txt`: Schema consistency and 78-feature cross-dataset mapping.
* `temporal_audit.txt`: Investigation of Timestamp support and sequence feasibility.
* `data_quality_report.txt`: Non-finite values, negatives, sentinels (-1), and duplicates.
* `dataset_comparison.csv`: Side-by-side comparison matrix.
* `potential_leakage_features.csv`: Categorization and risk assessment of candidate leakage features.
* `audit_report.txt`: Master human-readable scientific audit report.

---

### 6. Reproducibility & Environment
* **Platform**: Windows 11 Pro 64-bit
* **Python**: 3.11.0
* **PyTorch**: 2.2.2 (CPU execution; CUDA GPU not available on local hardware)
* **Pandas**: 3.0.0
* **NumPy**: 1.26.4
* **Hardware**: Intel Core i7-7500U @ 2.70GHz (2 cores, 4 threads), 8.0 GB Physical RAM
