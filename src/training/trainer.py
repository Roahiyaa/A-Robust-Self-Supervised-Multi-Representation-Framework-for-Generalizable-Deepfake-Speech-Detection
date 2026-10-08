import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from src.training.losses import MultiTaskLoss
from src.utils.logger import setup_logger

class Trainer:
    """
    Standard Trainer handling epoch loops, validation evaluation, checkpoint saving/resuming,
    learning-rate scheduling, and early stopping.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: MultiTaskLoss,
        device: torch.device,
        save_dir: str = "checkpoints",
        scheduler: Optional[Any] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.save_dir = save_dir
        self.scheduler = scheduler
        self.logger = setup_logger("trainer")
        
        os.makedirs(save_dir, exist_ok=True)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            waveforms = batch["waveform"].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch.items() if k != "waveform" and isinstance(v, torch.Tensor)}
            
            self.optimizer.zero_grad()
            outputs = self.model(waveforms)
            loss_dict = self.loss_fn(outputs, targets)
            
            loss = loss_dict["total_loss"]
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / max(num_batches, 1)
        if self.scheduler is not None:
            self.scheduler.step()
            
        return {"train_loss": avg_loss}

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                waveforms = batch["waveform"].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch.items() if k != "waveform" and isinstance(v, torch.Tensor)}
                
                outputs = self.model(waveforms)
                loss_dict = self.loss_fn(outputs, targets)
                total_loss += loss_dict["total_loss"].item()
                num_batches += 1
                
        avg_loss = total_loss / max(num_batches, 1)
        return {"val_loss": avg_loss}

    def save_checkpoint(self, epoch: int, path: str):
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict()
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to: {path}")

    def load_checkpoint(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.logger.info(f"Loaded checkpoint from: {path} (epoch {checkpoint['epoch']})")
