"""CLI runner for evaluating Multi-Class Known Attacks and Unseen Zero-Day Intrusions."""

import argparse
import json
import sys
from pathlib import Path

import torch

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from configs.project_config import MODELS_DIR, PROCESSED_DATA_DIR, RESULTS_DIR
from src.data.sequence_loader import create_streaming_dataloader
from src.evaluation.zero_day_evaluator import ZeroDayEvaluator
from src.models.cnn_lstm_model import DualHeadCNNLSTM


def main():
    parser = argparse.ArgumentParser(description="Evaluate Zero-Day & Known Attack Detection")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=10, help="Temporal sequence length")
    parser.add_argument("--target_fpr", type=float, default=0.01, help="Target FPR on Benign for threshold calibration")
    parser.add_argument("--max_eval_rows", type=int, default=25000, help="Max flows for val calibration and test known evaluation")
    parser.add_argument("--val_sample", type=int, default=None, help="If set, evaluate only on N validation samples")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda)")
    args = parser.parse_args()

    checkpoint_path = MODELS_DIR / "best_cnn_lstm.pt"
    scaler_path = MODELS_DIR / "robust_scaler.joblib"
    val_csv = PROCESSED_DATA_DIR / "val.csv"
    test_known_csv = PROCESSED_DATA_DIR / "test_known.csv"
    zero_day_csv = PROCESSED_DATA_DIR / "zero_day_test.csv"
    class_weights_path = MODELS_DIR / "class_weights.json"

    with open(class_weights_path, "r", encoding="utf-8") as fp:
        meta = json.load(fp)
        class_names = meta["classes"]

    print("=" * 80, flush=True)
    print("ZERO-DAY & LOW-FREQUENCY INTRUSION DETECTION BENCHMARK EVALUATION", flush=True)
    print(f"Checkpoint:       {checkpoint_path}", flush=True)
    print(f"Known Test:       {test_known_csv} (Eval Rows: {args.max_eval_rows})", flush=True)
    print(f"Zero-Day Test:    {zero_day_csv} (1,001 flows: Heartbleed, SQLi, XSS)", flush=True)
    print("=" * 80, flush=True)

    val_loader = create_streaming_dataloader(
        csv_file_path=val_csv,
        scaler_path=scaler_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_rows=args.max_eval_rows,
    )

    test_csv_to_use = test_known_csv
    try:
        with open(test_known_csv, "r", encoding="utf-8") as f:
            f.readline()
    except (PermissionError, FileNotFoundError):
        print(f"Notice: Using {PROCESSED_DATA_DIR / 'test_full.csv'} for evaluation.", flush=True)
        test_csv_to_use = PROCESSED_DATA_DIR / "test_full.csv"

    test_known_loader = create_streaming_dataloader(
        csv_file_path=test_csv_to_use,
        scaler_path=scaler_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_rows=args.max_eval_rows,
    )

    zero_day_loader = create_streaming_dataloader(
        csv_file_path=zero_day_csv,
        scaler_path=scaler_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )

    model = DualHeadCNNLSTM(
        num_features=66,
        seq_len=args.seq_len,
        num_known_classes=len(class_names),
        cnn_channels=128,
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.25,
    )

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded trained model checkpoint from {checkpoint_path.name}")
    else:
        print("Note: Running evaluation on initialized model (train model first for trained metrics).")

    evaluator = ZeroDayEvaluator(
        model=model,
        val_loader=val_loader,
        test_known_loader=test_known_loader,
        zero_day_loader=zero_day_loader,
        class_names=class_names,
        results_dir=RESULTS_DIR,
        device=args.device,
    )

    if args.val_sample:
        evaluator.evaluate_validation_sample(num_samples=args.val_sample)
        return

    print("\n1. Calibrating Anomaly Threshold tau on Benign Validation Traffic...", flush=True)
    tau = evaluator.calibrate_anomaly_threshold(target_fpr=args.target_fpr)
    print(f"   Calibrated Threshold (tau at {args.target_fpr*100:.1f}% FPR): {tau:.6f}", flush=True)

    print("\n2. Evaluating Multi-Class Known Attack Classification...", flush=True)
    known_metrics = evaluator.evaluate_known_classes()
    print(f"   Overall Known Macro-F1:    {known_metrics['macro_f1']:.4f} ({known_metrics['macro_f1']*100:.1f}%)", flush=True)
    print(f"   Overall Known Weighted-F1: {known_metrics['weighted_f1']:.4f} ({known_metrics['weighted_f1']*100:.1f}%)", flush=True)

    # Print dedicated Low-Frequency Attack Table
    lf_summary = known_metrics.get("low_frequency_summary", {})
    lf_classes = lf_summary.get("classes_evaluated", {})
    print("\n" + "=" * 85, flush=True)
    print("LOW-FREQUENCY ATTACK DETECTION PERFORMANCE (STEALTH ATTACKS BENCHMARK)", flush=True)
    print("=" * 85, flush=True)
    print(f"{'Attack Class':<22} {'Loss Weight':<12} {'Test Support':<14} {'Precision':<12} {'Recall':<12} {'F1-Score':<10}", flush=True)
    print("-" * 85, flush=True)
    for c_name, c_info in lf_classes.items():
        w = c_info.get("loss_weight", 1.0)
        s = c_info.get("support", 0)
        p = c_info.get("precision", 0.0) * 100
        r = c_info.get("recall", 0.0) * 100
        f = c_info.get("f1_score", 0.0)
        print(f"{c_name:<22} {w:>8,.1f}x   {s:<14,d} {p:>8.2f}%    {r:>8.2f}%    {f:>8.4f}", flush=True)
    print("-" * 85, flush=True)
    print(f"Low-Frequency Macro-F1:    {lf_summary.get('low_frequency_macro_f1', 0.0):.4f}", flush=True)
    print(f"Low-Frequency Mean Recall: {lf_summary.get('low_frequency_mean_recall', 0.0)*100:.2f}%", flush=True)
    print("=" * 85, flush=True)

    print("\n3. Evaluating Unseen Zero-Day Anomaly Detection Head (Option B Benchmark)...", flush=True)
    zd_metrics = evaluator.evaluate_zero_day_detection(threshold=tau)
    family_breakdown = zd_metrics.get("zero_day_family_breakdown", {})

    print("\n" + "=" * 85, flush=True)
    print(f"UNSEEN ZERO-DAY ATTACK ANOMALY BREAKDOWN (CALIBRATED TAU = {tau:.6f})", flush=True)
    print("=" * 85, flush=True)
    print(f"{'Attack Family':<22} {'Exploit Vector':<26} {'Flows':<8} {'Flagged':<10} {'Recall':<10} {'Mean MSE':<10}", flush=True)
    print("-" * 85, flush=True)
    for fname, finfo in family_breakdown.items():
        vec = finfo.get("vector", "Network Exploit")[:24]
        flows = finfo.get("total_flows", 0)
        det = finfo.get("detected_count", 0)
        rec = finfo.get("detection_recall", 0.0) * 100
        mse = finfo.get("mean_anomaly_score", 0.0)
        print(f"{fname:<22} {vec:<26} {flows:<8,d} {det:<10,d} {rec:>6.2f}%    {mse:<10.4f}", flush=True)
    print("-" * 85, flush=True)
    zd_tested = zd_metrics['total_zero_day_flows_tested']
    zd_det = zd_metrics['zero_day_detected_count']
    zd_rec = zd_metrics['zero_day_detection_recall'] * 100
    print(f"TOTAL ZERO-DAY RECALL:     {zd_rec:.2f}% ({zd_det:,d} / {zd_tested:,d} novel exploits detected)", flush=True)
    print("=" * 85, flush=True)

    print("\n4. Generating Human-Level Scientific Interpretation Report...", flush=True)
    report_file = evaluator.generate_human_interpretation_report(
        known_metrics=known_metrics,
        zd_metrics=zd_metrics,
        calibrated_threshold=tau,
        target_fpr=args.target_fpr,
    )

    print("\n" + "=" * 85, flush=True)
    print(f"EVALUATION COMPLETE! Report saved: {report_file.name}", flush=True)
    print("=" * 85, flush=True)


if __name__ == "__main__":
    main()
