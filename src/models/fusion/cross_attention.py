import torch
import torch.nn as nn

class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional Cross-Attention Bridge:
    - z_a (SSL) queries z_b (Spectral)
    - z_b (Spectral) queries z_a (SSL)
    Exposes cross-modal feature discrepancies typical of deepfake artifacts.
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.attn_a_to_b = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.attn_b_to_a = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm_a = nn.LayerNorm(embed_dim)
        self.norm_b = nn.LayerNorm(embed_dim)

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """
        Inputs: z_a, z_b of shape (B, T_frames, embed_dim)
        Returns: Fused cross-attended feature map of shape (B, T_frames, embed_dim)
        """
        # z_a querying z_b
        attn_out_a, _ = self.attn_a_to_b(query=z_a, key=z_b, value=z_b)
        out_a = self.norm_a(z_a + attn_out_a)
        
        # z_b querying z_a
        attn_out_b, _ = self.attn_b_to_a(query=z_b, key=z_a, value=z_a)
        out_b = self.norm_b(z_b + attn_out_b)
        
        return (out_a + out_b) / 2.0
