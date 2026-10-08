import torch
import torch.nn as nn

class TemporalTransformer(nn.Module):
    """
    Temporal Transformer Encoder modeling frame-wise and chunk-sequence dynamics.
    """
    def __init__(self, embed_dim: int = 256, num_layers: int = 2, nhead: int = 4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x of shape (B, T_sequence, embed_dim)
        Output: Contextual temporal embeddings of shape (B, T_sequence, embed_dim)
        """
        return self.transformer(x)
