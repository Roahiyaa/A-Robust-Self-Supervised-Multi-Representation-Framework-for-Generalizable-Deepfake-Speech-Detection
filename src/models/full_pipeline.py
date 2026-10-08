import torch
import torch.nn as nn
from typing import Dict, Any

from src.features.ssl_extractor import SSLExtractor
from src.features.spectral_extractor import SpectralExtractor
from src.features.acoustic_extractor import AcousticExtractor
from src.models.fusion.gated_fusion import GatedFeatureFusion
from src.models.fusion.cross_attention import BidirectionalCrossAttention
from src.models.temporal.temporal_transformer import TemporalTransformer
from src.models.heads.multi_task_heads import MultiTaskHeads

class DeepfakeSpeechDetectionPipeline(nn.Module):
    """
    Proposed Multi-Representation Framework for Generalizable Deepfake Speech Detection.
    Combines Branch A (SSL WavLM), Branch B (Spectral LFCC/LogMel), and Branch C (Acoustic with Validity Masks).
    Uses Gated Feature Fusion, Cross-Attention, Temporal Transformer, and Conditional Multi-Task Heads.
    """
    def __init__(
        self,
        embed_dim: int = 256,
        ssl_model_name: str = "microsoft/wavlm-base-plus",
        ssl_strategy: str = "frozen",
        spectral_type: str = "lfcc",
        enable_attack_head: bool = False,
        enable_segment_head: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 1. Feature Branches
        self.branch_a_ssl = SSLExtractor(model_name=ssl_model_name, finetune_strategy=ssl_strategy)
        self.branch_b_spectral = SpectralExtractor(spectral_type=spectral_type)
        self.branch_c_acoustic = AcousticExtractor()
        
        # 2. Linear Projections into D_embed
        # Branch A SSL output dim (e.g. 768 for WavLM base)
        ssl_dim = self.branch_a_ssl.hidden_dim
        self.proj_a = nn.Linear(ssl_dim, embed_dim)
        
        # Branch B Spectral Conv Projection
        self.conv_b = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((embed_dim, None))
        )
        self.proj_b = nn.Linear(16, 1) # Reduce channels
        
        # Branch C Acoustic Projection (9 numeric + 3 masks = 12 dim)
        self.proj_c = nn.Linear(12, embed_dim)
        
        # 3. Fusion & Temporal Modeling
        self.gated_fusion = GatedFeatureFusion(embed_dim=embed_dim)
        self.cross_attention = BidirectionalCrossAttention(embed_dim=embed_dim)
        self.temporal_transformer = TemporalTransformer(embed_dim=embed_dim)
        
        # 4. Multi-Task Heads
        self.heads = MultiTaskHeads(
            embed_dim=embed_dim,
            enable_attack_head=enable_attack_head,
            enable_segment_head=enable_segment_head
        )

    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Input: waveform of shape (B, T_samples)
        Returns dictionary of task logits.
        """
        # --- Branch A: SSL ---
        h_a = self.branch_a_ssl(waveform) # (B, T_a, ssl_dim)
        z_a = self.proj_a(h_a) # (B, T_a, embed_dim)
        
        # --- Branch B: Spectral ---
        h_b = self.branch_b_spectral(waveform) # (B, 1, F, T_b)
        h_b_conv = self.conv_b(h_b) # (B, 16, embed_dim, T_b)
        # Permute & project to (B, T_b, embed_dim)
        h_b_perm = h_b_conv.permute(0, 3, 2, 1) # (B, T_b, embed_dim, 16)
        z_b = self.proj_b(h_b_perm).squeeze(-1) # (B, T_b, embed_dim)
        
        # --- Branch C: Acoustic ---
        feats_c, masks_c = self.branch_c_acoustic(waveform) # (B, T_c, 9), (B, T_c, 3)
        h_c = torch.cat([feats_c, masks_c], dim=-1) # (B, T_c, 12)
        z_c = self.proj_c(h_c) # (B, T_c, embed_dim)
        
        # Interpolate sequence lengths to align with z_a sequence length T_target
        T_target = z_a.shape[1]
        
        if z_b.shape[1] != T_target:
            z_b = nn.functional.interpolate(z_b.transpose(1, 2), size=T_target, mode='linear', align_corners=False).transpose(1, 2)
        if z_c.shape[1] != T_target:
            z_c = nn.functional.interpolate(z_c.transpose(1, 2), size=T_target, mode='linear', align_corners=False).transpose(1, 2)
            
        # --- Fusion & Cross Attention ---
        z_cross = self.cross_attention(z_a, z_b)
        z_fused = self.gated_fusion(z_cross, z_b, z_c)
        
        # --- Temporal Transformer ---
        z_temporal = self.temporal_transformer(z_fused)
        
        # --- Prediction Heads ---
        outputs = self.heads(z_temporal)
        return outputs
