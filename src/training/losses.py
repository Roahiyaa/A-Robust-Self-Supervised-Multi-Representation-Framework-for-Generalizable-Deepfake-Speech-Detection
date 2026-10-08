import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon).
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: (B, D) normalized feature embeddings
        labels: (B,) class labels
        """
        device = features.device
        batch_size = features.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=device)
            
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # mask-out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        loss = - mean_log_prob_pos.mean()
        return loss

class MultiTaskLoss(nn.Module):
    """
    Combines weighted losses across active task heads:
    L_total = lambda_binary * L_binary + I_attack * lambda_attack * L_attack + I_segment * lambda_segment * L_segment + I_supcon * lambda_supcon * L_supcon
    """
    def __init__(
        self,
        lambda_binary: float = 1.0,
        lambda_attack: float = 0.5,
        lambda_segment: float = 0.5,
        lambda_supcon: float = 0.1,
        use_supcon: bool = False
    ):
        super().__init__()
        self.lambda_binary = lambda_binary
        self.lambda_attack = lambda_attack
        self.lambda_segment = lambda_segment
        self.lambda_supcon = lambda_supcon
        self.use_supcon = use_supcon
        
        self.bce_loss = nn.CrossEntropyLoss()
        self.attack_loss_fn = nn.CrossEntropyLoss()
        self.segment_loss_fn = nn.BCEWithLogitsLoss()
        self.supcon_loss_fn = SupervisedContrastiveLoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        embeddings: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        loss_dict = {}
        total_loss = 0.0
        
        # 1. Binary Loss
        binary_loss = self.bce_loss(predictions["binary"], targets["label"])
        loss_dict["binary_loss"] = binary_loss
        total_loss += self.lambda_binary * binary_loss
        
        # 2. Conditional Attack Loss
        if "attack" in predictions and "attack_label" in targets:
            attack_loss = self.attack_loss_fn(predictions["attack"], targets["attack_label"])
            loss_dict["attack_loss"] = attack_loss
            total_loss += self.lambda_attack * attack_loss
            
        # 3. Conditional Segment Loss
        if "segment" in predictions and "segment_mask" in targets:
            segment_loss = self.segment_loss_fn(predictions["segment"], targets["segment_mask"])
            loss_dict["segment_loss"] = segment_loss
            total_loss += self.lambda_segment * segment_loss
            
        # 4. Optional SupCon Loss
        if self.use_supcon and embeddings is not None:
            norm_embeds = F.normalize(embeddings, p=2, dim=1)
            supcon_loss = self.supcon_loss_fn(norm_embeds, targets["label"])
            loss_dict["supcon_loss"] = supcon_loss
            total_loss += self.lambda_supcon * supcon_loss
            
        loss_dict["total_loss"] = total_loss
        return loss_dict
