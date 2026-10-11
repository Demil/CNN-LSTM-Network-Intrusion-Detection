"""CLI runner for training the Dual-Head Hybrid CNN-LSTM."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from configs.project_config import MODELS_DIR, PROCESSED_DATA_DIR
from src.data.sequence_loader import create_streaming_dataloader
from src.models.cnn_lstm_model import DualHeadCNNLSTM
from src.training.trainer import CNNLSTMTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Dual-Head CNN-LSTM Intrusion Detection Model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=10, help="Temporal sequence length")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Max steps per epoch (useful for local laptop smoke test)")
    parser.add_argument("--max_val_steps", type=int, default=None, help="Max val steps (useful for local laptop smoke test)")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda)")
    args = parser.parse_args()

    train_csv = PROCESSED_DATA_DIR / "train.csv"
    val_csv = PROCESSED_DATA_DIR / "val.csv"
    scaler_path = MODELS_DIR / "robust_scaler.joblib"
    class_weights_path = MODELS_DIR / "class_weights.json"

    print("=" * 80)
    print("DUAL-HEAD CNN-LSTM INTRUSION DETECTION TRAINING")
    print(f"Train Dataset: {train_csv}")
    print(f"Val Dataset:   {val_csv}")
    print(f"Scaler:        {scaler_path}")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | Seq Len: {args.seq_len}")
    print("=" * 80)

    train_loader = create_streaming_dataloader(
        csv_file_path=train_csv,
        scaler_path=scaler_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )

    val_loader = create_streaming_dataloader(
        csv_file_path=val_csv,
        scaler_path=scaler_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )

    model = DualHeadCNNLSTM(
        num_features=66,
        seq_len=args.seq_len,
        num_known_classes=12,
        cnn_channels=128,
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.25,
    )

    trainer = CNNLSTMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights_path=class_weights_path,
        learning_rate=1e-3,
        weight_decay=1e-4,
        recon_weight=0.1,
        device=args.device,
        checkpoint_dir=MODELS_DIR,
    )

    max_val_steps = args.max_val_steps
    if args.max_train_steps is not None and max_val_steps is None:
        max_val_steps = max(args.max_train_steps // 5, 100)

    results = trainer.fit(
        epochs=args.epochs,
        max_train_steps_per_epoch=args.max_train_steps,
        max_val_steps=max_val_steps,
    )

    print("\nTraining completed successfully!")
    print(f"Best Validation Macro-F1: {results['best_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
