import pytest
import torch
from src.models.full_pipeline import DeepfakeSpeechDetectionPipeline

def test_full_pipeline_forward():
    # Construct lightweight dummy pipeline for testing
    model = DeepfakeSpeechDetectionPipeline(
        embed_dim=64,
        ssl_model_name="microsoft/wavlm-base-plus",
        ssl_strategy="frozen",
        spectral_type="lfcc",
        enable_attack_head=True,
        enable_segment_head=True
    )
    
    # Batch size 2, 16000 samples (1 second audio at 16kHz)
    dummy_audio = torch.randn(2, 16000)
    outputs = model(dummy_audio)
    
    assert "binary" in outputs
    assert outputs["binary"].shape == (2, 2)
    
    assert "attack" in outputs
    assert outputs["attack"].shape[0] == 2
    
    assert "segment" in outputs
    assert outputs["segment"].shape[0] == 2
