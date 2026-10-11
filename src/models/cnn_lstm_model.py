"""Dual-Head Hybrid CNN-LSTM Neural Network for Zero-Day and Low-Frequency Intrusion Detection.

Architecture:
1. Feature Extractor (1D-CNN):
   - Convolutions across canonical behavioral metrics to learn local spatial/feature interactions.
   - Conv1D -> BatchNorm -> LeakyReLU -> Dropout -> Conv1D -> BatchNorm -> LeakyReLU.
2. Temporal Sequence Learner (Bi-directional LSTM):
   - Captures forward and backward temporal transitions across consecutive network flows.
   - 2-layer Bidirectional LSTM with dropout.
3. Head 1: Known Attack Multi-Class Classifier:
   - Dense -> LeakyReLU -> Dropout -> Dense(12 logits for known classes).
   - Trained with Class-Weighted Focal Loss to prevent majority-class bias on low-frequency attacks.
4. Head 2: Unsupervised Zero-Day Anomaly Detection Head (Reconstruction Autoencoder):
   - Dense -> Dense -> Decodes back to [Batch, Seq_Len, Features].
   - Evaluates reconstruction error MSE(x, x_hat).
   - Samples exceeding the threshold tau (calibrated on benign validation traffic) are flagged
     as novel ZERO-DAY ATTACKS (Heartbleed, SQL Injection, XSS).
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Class-Weighted Focal Loss for extreme class imbalance in network intrusion detection.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
    # below supper method is important , keep it as it is 
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = ((1.0 - p_t) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class DualHeadCNNLSTM(nn.Module):
    """Hybrid 1D-CNN + BiLSTM model with dual supervised/unsupervised detection heads."""

    def __init__(
        self,
        num_features: int = 66,
        seq_len: int = 10,
        num_known_classes: int = 12,
        cnn_channels: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.num_known_classes = num_known_classes

        # -------------------------------------------------------------
        # 1. 1D-CNN Feature Extractor (Applied per time-step / sequence)
        # -------------------------------------------------------------
        # Input shape: [Batch * Seq_Len, 1, Features]
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool1d(1),  # Compresses feature dimension to [B*L, cnn_channels, 1]
        )

        # -------------------------------------------------------------
        # 2. Bi-directional LSTM Temporal Sequence Model
        # -------------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # BiLSTM outputs 2 * lstm_hidden
        self.latent_dim = lstm_hidden * 2

        # -------------------------------------------------------------
        # 3. Supervised Classification Head (12 Known Classes)
        # -------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(128, num_known_classes),
        )

        # -------------------------------------------------------------
        # 4. Unsupervised Reconstruction Head (Zero-Day Anomaly Detector)
        # -------------------------------------------------------------
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, seq_len * num_features),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [Batch, Seq_Len, Features]
            
        Returns:
            logits: Classification logits of shape [Batch, Num_Known_Classes]
            reconstructed: Reconstructed input of shape [Batch, Seq_Len, Features]
            latent: Latent sequence representation of shape [Batch, Latent_Dim]
        """
        batch_size, seq_len, num_features = x.shape

        # 1. Pass each flow through 1D-CNN
        # Reshape to [Batch * Seq_Len, 1, Features]
        x_reshaped = x.view(batch_size * seq_len, 1, num_features)
        cnn_out = self.cnn(x_reshaped)  # [Batch * Seq_Len, cnn_channels, 1]
        cnn_features = cnn_out.view(batch_size, seq_len, -1)  # [Batch, Seq_Len, cnn_channels]

        # 2. Pass through Bi-directional LSTM
        lstm_out, (hn, cn) = self.lstm(cnn_features)  # lstm_out: [Batch, Seq_Len, 2 * lstm_hidden]

        # Use final time-step representation for sequence classification
        latent = lstm_out[:, -1, :]  # [Batch, Latent_Dim]


        # 3. Supervised classification logits
        logits = self.classifier(latent)

        # 4. Unsupervised reconstruction
        recon_flat = self.decoder(latent)
        reconstructed = recon_flat.view(batch_size, seq_len, num_features)

        return logits, reconstructed, latent

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Computes Mean Squared Error reconstruction error for zero-day detection."""
        with torch.no_grad():
            _, reconstructed, _ = self.forward(x)
            # Per-sample MSE across sequence and feature dimensions
            mse_per_sample = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
            return mse_per_sample
