import torch
import torch.nn as nn
from transformers import AutoModel, AutoFeatureExtractor
from typing import Optional, Dict, Any

class SSLExtractor(nn.Module):
    """
    Branch A: Self-Supervised Speech Representation.
    Supports WavLM Base+ (primary) or wav2vec2 (comparison).
    Provides frozen, partially fine-tuned, or fully fine-tuned configuration.
    """
    def __init__(
        self,
        model_name: str = "microsoft/wavlm-base-plus",
        finetune_strategy: str = "frozen",
        unfreeze_top_k: int = 4
    ):
        super().__init__()
        self.model_name = model_name
        self.finetune_strategy = finetune_strategy
        self.unfreeze_top_k = unfreeze_top_k
        
        # Load transformer SSL encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.encoder.config.hidden_size
        
        self._configure_fine_tuning()

    def _configure_fine_tuning(self):
        if self.finetune_strategy == "frozen":
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif self.finetune_strategy == "partial":
            for param in self.encoder.parameters():
                param.requires_grad = False
            # Unfreeze last K encoder layers
            if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layers"):
                layers = self.encoder.encoder.layers
                for layer in layers[-self.unfreeze_top_k:]:
                    for param in layer.parameters():
                        param.requires_grad = True
        elif self.finetune_strategy == "full":
            for param in self.encoder.parameters():
                param.requires_grad = True

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Input: input_values of shape (B, T_samples)
        Output: contextual representation of shape (B, T_frames, hidden_dim)
        """
        outputs = self.encoder(input_values)
        last_hidden_state = outputs.last_hidden_state
        return last_hidden_state
