import torch
import torch.nn as nn
from typing import Dict, Optional

class MultiTaskHeads(nn.Module):
    """
    Multi-Task Output Heads:
    1. Binary Classifier Head (REAL vs FAKE): Always active.
    2. Conditional Attack Classifier Head: Active only when attack labels exist.
    3. Conditional Segment Localization Head: Active strictly when segment-level annotations exist.
    """
    def __init__(
        self,
        embed_dim: int = 256,
        num_attack_classes: int = 10,
        enable_attack_head: bool = False,
        enable_segment_head: bool = False
    ):
        super().__init__()
        self.enable_attack_head = enable_attack_head
        self.enable_segment_head = enable_segment_head

        # Binary Head
        self.binary_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 2)
        )

        # Conditional Attack Head
        if enable_attack_head:
            self.attack_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, num_attack_classes)
            )

        # Conditional Segment Localization Head
        if enable_segment_head:
            self.segment_head = nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(embed_dim // 2, 1, kernel_size=1)
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Input: x of shape (B, T_frames, embed_dim)
        Returns dictionary containing logits:
            - 'binary': (B, 2) utterance-level prediction
            - 'attack': (B, num_attack_classes) [if active]
            - 'segment': (B, T_frames) frame-level anomaly probabilities [if active]
        """
        # Utterance pooling (mean pooling across sequence dimension)
        pooled = x.mean(dim=1) # (B, embed_dim)
        
        outputs = {
            "binary": self.binary_head(pooled)
        }
        
        if self.enable_attack_head:
            outputs["attack"] = self.attack_head(pooled)
            
        if self.enable_segment_head:
            # Transpose for Conv1d: (B, embed_dim, T_frames)
            x_trans = x.transpose(1, 2)
            seg_logits = self.segment_head(x_trans).squeeze(1) # (B, T_frames)
            outputs["segment"] = seg_logits
            
        return outputs
