"""Residual bag-of-words regressor for the VFM target (0-99).

VFM is roughly bell-shaped on 0-99, so plain standardization is used.
The trainer computes Y_MEAN / Y_STD from the training data and persists
them alongside the weights, so inference inverts the exact normalization
used in training. The HashingVectorizer is stateless — nothing to fit.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import HashingVectorizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.items import Wine

N_FEATURES: int = 5_000
HIDDEN_SIZE: int = 512
DROPOUT_PROB: float = 0.2
BATCH_SIZE: int = 256
LEARNING_RATE: float = 1e-3
EPOCHS: int = 10
MAX_GRAD_NORM: float = 1.0
SEED: int = 42

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

vectorizer = HashingVectorizer(n_features=N_FEATURES, stop_words="english", binary=True)


class ResidualBlock(nn.Module):
    """Two-layer block with LayerNorm, dropout, and a skip connection."""

    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.relu(self.block(inputs) + inputs)
        return out


class VfmNet(nn.Module):
    """Bag-of-words regressor: projection -> 2 residual blocks -> scalar."""

    def __init__(self, input_size: int = N_FEATURES, hidden: int = HIDDEN_SIZE) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            ResidualBlock(hidden, DROPOUT_PROB),
            ResidualBlock(hidden, DROPOUT_PROB),
            nn.Linear(hidden, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.network(inputs)
        return out


class VfmTrainer:
    """Trains VfmNet on curated wines, serves predictions, saves/loads weights.

    Attributes:
        model: The network (trained after .train(), or loaded via .load()).
        y_mean: Target mean computed from the training data.
        y_std: Target std computed from the training data.
    """

    def __init__(self) -> None:
        torch.manual_seed(SEED)
        self.model = VfmNet().to(DEVICE)
        self.y_mean: float = 0.0
        self.y_std: float = 1.0

    def train(self, wines: list[Wine]) -> None:
        """Fit the network; prints loss per epoch."""
        features = vectorizer.transform([w.summary for w in wines])
        targets = np.array([w.vfm for w in wines], dtype=np.float32)
        self.y_mean, self.y_std = float(targets.mean()), float(targets.std())
        print(f"Target stats -> Y_MEAN={self.y_mean:.4f}  Y_STD={self.y_std:.4f}")
        normalized = (targets - self.y_mean) / self.y_std

        dataset = TensorDataset(
            torch.FloatTensor(features.toarray()),
            torch.FloatTensor(normalized).unsqueeze(1),
        )
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        loss_function = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        self.model.train()
        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            for batch_x, batch_y in tqdm(loader, desc=f"Epoch {epoch + 1}"):
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                loss = loss_function(self.model(batch_x), batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            print(f"Epoch {epoch + 1}: loss {epoch_loss / len(loader):.4f}")

    def predict(self, summary: str) -> float:
        """Predict VFM for one assembled summary, inverting the standardization."""
        self.model.eval()
        features = vectorizer.transform([summary]).toarray()
        with torch.no_grad():
            normalized: float = self.model(torch.FloatTensor(features).to(DEVICE)).item()
        return normalized * self.y_std + self.y_mean

    def save(self, path: str) -> None:
        """Persist weights and target-normalization stats together."""
        torch.save(
            {"state_dict": self.model.state_dict(), "y_mean": self.y_mean, "y_std": self.y_std},
            path,
        )

    def load(self, path: str) -> None:
        """Restore weights and target-normalization stats."""
        checkpoint = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.y_mean = float(checkpoint["y_mean"])
        self.y_std = float(checkpoint["y_std"])
        self.model.eval()
