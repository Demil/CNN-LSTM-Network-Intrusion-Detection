"""Training Engine for Dual-Head Hybrid CNN-LSTM Network Intrusion Detection."""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.models.cnn_lstm_model import DualHeadCNNLSTM, FocalLoss


class CNNLSTMTrainer:
    """Trains the Dual-Head CNN-LSTM with Class-Weighted Focal Loss and Reconstruction MSE."""

    def __init__(
        self,
        model: DualHeadCNNLSTM,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights_path: Optional[Path] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        recon_weight: float = 0.1,
        device: Optional[str] = None,
        checkpoint_dir: Optional[Path] = None,
    ) -> None:
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.recon_weight = recon_weight
        self.checkpoint_dir = Path(checkpoint_dir or "models")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------------------
        # Load Class Weights for Focal Loss
        # -------------------------------------------------------------
        alpha_weights = None
        if class_weights_path and Path(class_weights_path).exists():
            with open(class_weights_path, "r", encoding="utf-8") as fp:
                meta = json.load(fp)
                b_weights = meta.get("balanced_class_weights", {})
                w_list = [b_weights.get(cls_name, 1.0) for cls_name in meta["classes"]]
                # Normalize so mean weight is 1.0
                mean_w = sum(w_list) / len(w_list)
                norm_w = [w / mean_w for w in w_list]
                alpha_weights = torch.tensor(norm_w, dtype=torch.float32).to(self.device)

        self.focal_loss_fn = FocalLoss(alpha=alpha_weights, gamma=2.0)
        self.mse_loss_fn = nn.MSELoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10, eta_min=1e-5
        )

        self.best_val_macro_f1 = 0.0

    def train_epoch(self, max_steps: Optional[int] = None) -> Dict[str, float]:
        """Trains for one epoch or up to max_steps."""
        self.model.train()
        total_loss = 0.0
        total_focal = 0.0
        total_recon = 0.0
        steps = 0

        t0 = time.time()
        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            logits, recon, _ = self.model(batch_x)

            l_focal = self.focal_loss_fn(logits, batch_y)
            l_recon = self.mse_loss_fn(recon, batch_x)
            l_total = l_focal + (self.recon_weight * l_recon)

            l_total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += l_total.item()
            total_focal += l_focal.item()
            total_recon += l_recon.item()
            steps += 1

            if steps % 100 == 0:
                elapsed_b = time.time() - t0
                speed = steps / max(elapsed_b, 0.001)
                print(
                    f"  [Batch {steps:,}] Total Loss: {l_total.item():.4f} "
                    f"(Focal: {l_focal.item():.4f}, Recon: {l_recon.item():.4f}) | {speed:.1f} batches/s",
                    flush=True,
                )

            if max_steps and steps >= max_steps:
                break

        dur = time.time() - t0
        avg_loss = total_loss / max(steps, 1)
        avg_focal = total_focal / max(steps, 1)
        avg_recon = total_recon / max(steps, 1)

        return {
            "loss": avg_loss,
            "focal_loss": avg_focal,
            "recon_loss": avg_recon,
            "steps": steps,
            "duration": dur,
        }

    def evaluate(self, max_steps: Optional[int] = None) -> Dict[str, float]:
        """Evaluates on the validation set."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        steps = 0

        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                logits, recon, _ = self.model(batch_x)
                l_focal = self.focal_loss_fn(logits, batch_y)
                l_recon = self.mse_loss_fn(recon, batch_x)
                l_total = l_focal + (self.recon_weight * l_recon)

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                targets = batch_y.cpu().numpy()

                all_preds.extend(preds)
                all_targets.extend(targets)
                total_loss += l_total.item()
                steps += 1

                if steps % 100 == 0:
                    print(f"  [Validating] Step {steps:,} | Current Val Loss: {total_loss / steps:.4f}", flush=True)

                if max_steps and steps >= max_steps:
                    break

        avg_loss = total_loss / max(steps, 1)
        macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

        return {
            "val_loss": avg_loss,
            "val_macro_f1": macro_f1,
            "val_weighted_f1": weighted_f1,
            "val_steps": steps,
        }

    def fit(
        self,
        epochs: int = 5,
        max_train_steps_per_epoch: Optional[int] = None,
        max_val_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Full training loop with model checkpointing."""
        history = []
        print(f"Training on device: {self.device}", flush=True)

        for ep in range(1, epochs + 1):
            t_ep = time.time()
            train_metrics = self.train_epoch(max_steps=max_train_steps_per_epoch)
            val_metrics = self.evaluate(max_steps=max_val_steps)
            self.scheduler.step()

            ep_dur = time.time() - t_ep
            val_f1 = val_metrics["val_macro_f1"]

            print(
                f"[Epoch {ep:02d}/{epochs:02d}] "
                f"Train Loss: {train_metrics['loss']:.4f} (Focal: {train_metrics['focal_loss']:.4f}, Recon: {train_metrics['recon_loss']:.4f}) | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | Val Macro-F1: {val_f1:.4f} ({ep_dur:.1f}s)",
                flush=True,
            )

            # Checkpoint best model
            if val_f1 > self.best_val_macro_f1:
                self.best_val_macro_f1 = val_f1
                ckpt_path = self.checkpoint_dir / "best_cnn_lstm.pt"
                torch.save(
                    {
                        "epoch": ep,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_val_macro_f1": val_f1,
                    },
                    ckpt_path,
                )
                print(f"  -> Best model saved to: {ckpt_path.name} (Macro-F1: {val_f1:.4f})", flush=True)

            history.append({
                "epoch": ep,
                "train": train_metrics,
                "val": val_metrics,
            })

        return {"best_macro_f1": self.best_val_macro_f1, "history": history}
