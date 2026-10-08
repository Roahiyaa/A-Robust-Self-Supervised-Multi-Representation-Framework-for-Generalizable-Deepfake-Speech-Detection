import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

class TemperatureScaler(nn.Module):
    """
    Applies Temperature Scaling to calibrate model confidence probabilities.
    Learns a single scalar temperature T > 0 strictly fitted on validation data logits.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Scale logits by temperature parameter.
        """
        return logits / self.temperature

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, max_iter: int = 50, lr: int = 0.01) -> float:
        """
        Fits optimal temperature parameter T strictly on validation set.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return float(self.temperature.item())
