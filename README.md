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
│   │   └── sequence_loader.py     # Bounded streaming DataLoader 
│   ├── preprocessing\
│   │   ├── central_csv_builder.py # Zero-loss single CSV builder 
│   │   └── zero_day_partitioner.py# zero-day isolator & 70/15/15 splitter
│   ├── models\
│   │   └── cnn_lstm_model.py      # DualHeadCNNLSTM (Focal Loss + Autoencoder heads)
│   ├── training\
│   │   └── trainer.py             # Cosine annealing, early stopping & heartbeat logs
│   └── evaluation\
│       └── zero_day_evaluator.py  # Anomaly calibration & zero-day detection metrics
│
├── scripts\
│   ├── build_central_csv_dataset.py# CLI for unified 18.34M dataset compilation
│   ├── run_zero_day_partitioning.py# CLI for Option B zero-day test & train/val split
│   ├── train_cnn_lstm.py           # CLI trainer with streaming sequence loader
│   └── evaluate_zero_day.py        # CLI zero-day evaluator & metrics generator
│
├── data\
│   ├── audit\                     # Forensic audit CSVs, TXT, and PDF report
│   └── processed\                 # Central dataset, partitions, scaler, audit reports
│
├── models\                        # Trained checkpoints (.pt) and scaler joblib artifacts
├── results\                       # Classification reports and zero-day evaluation metrics
├── reports\
│   ├── Dataset_Audit_Report.pdf            # Phase 1 Forensic Audit PDF Report
│   ├── Technical_Implementation_Report.pdf # Official Architecture & Technical Specification PDF
│   └── Technical_Implementation_Report.md  # Markdown Source Specification
└── README.md                      # Project documentation and guidelines
