import torch
import torch.nn as nn

class GatedFeatureFusion(nn.Module):
    """
    Learns adaptive dynamic gating weights (g_A, g_B, g_C) for Branch A (SSL),
    Branch B (Spectral), and Branch C (Acoustic) features projected into d_embed.
    """
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        # Gate network takes concatenated features (3 * embed_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 3),
            nn.Sigmoid()
        )

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor, z_c: torch.Tensor) -> torch.Tensor:
        """
        Inputs shape: (B, T_frames, embed_dim)
        Output shape: (B, T_frames, embed_dim)
        """
        concat_feats = torch.cat([z_a, z_b, z_c], dim=-1) # (B, T, 3 * embed_dim)
        gates = self.gate_net(concat_feats) # (B, T, 3)
        
        g_a = gates[..., 0:1] # (B, T, 1)
        g_b = gates[..., 1:2]
        g_c = gates[..., 2:3]
        
        fused = g_a * z_a + g_b * z_b + g_c * z_c
        return fused
