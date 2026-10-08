import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class UncertaintyAndOODDetector:
    """
    Standalone evaluation module for:
    1. Epistemic Uncertainty via Monte Carlo (MC) Dropout.
    2. Out-Of-Distribution (OOD) score using Energy-based score formulation:
       E(x; T) = - T * logsumexp(logits / T)
    """
    def __init__(self, temperature: float = 1.0, num_mc_passes: int = 10):
        self.temperature = temperature
        self.num_mc_passes = num_mc_passes

    def compute_energy_score(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Computes free-energy score. Higher energy indicates lower in-distribution likelihood (OOD).
        """
        energy = - self.temperature * torch.logsumexp(logits / self.temperature, dim=-1)
        return energy

    def compute_mc_dropout_uncertainty(
        self,
        model: nn.Module,
        waveform: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs N Monte Carlo forward passes with active dropout to measure predictive mean & variance.
        Returns:
            mean_probs: (B, num_classes)
            uncertainty_variance: (B,) predictive variance across forward passes
        """
        model.train() # Enable dropout during evaluation
        mc_probs = []
        
        with torch.no_grad():
            for _ in range(self.num_mc_passes):
                outputs = model(waveform)
                logits = outputs["binary"]
                probs = F.softmax(logits, dim=-1)
                mc_probs.append(probs)
                
        stacked_probs = torch.stack(mc_probs, dim=0) # (N, B, num_classes)
        mean_probs = stacked_probs.mean(dim=0)
        variance = stacked_probs.var(dim=0).mean(dim=-1) # (B,)
        
        return mean_probs, variance
