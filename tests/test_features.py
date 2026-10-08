import pytest
import torch
from src.features.spectral_extractor import SpectralExtractor
from src.features.acoustic_extractor import AcousticExtractor

def test_spectral_extractor_lfcc():
    extractor = SpectralExtractor(spectral_type="lfcc", sample_rate=16000)
    dummy_wav = torch.randn(2, 16000) # Batch size 2, 1 sec audio
    out = extractor(dummy_wav)
    assert out.dim() == 4
    assert out.shape[0] == 2
    assert out.shape[1] == 1

def test_spectral_extractor_logmel():
    extractor = SpectralExtractor(spectral_type="logmel", sample_rate=16000, n_mels=80)
    dummy_wav = torch.randn(2, 16000)
    out = extractor(dummy_wav)
    assert out.dim() == 4
    assert out.shape[0] == 2
    assert out.shape[2] == 80

def test_acoustic_extractor_masks():
    extractor = AcousticExtractor(sample_rate=16000)
    dummy_wav = torch.randn(2, 16000)
    feats, masks = extractor(dummy_wav)
    assert feats.dim() == 3
    assert masks.dim() == 3
    assert feats.shape[0] == 2
    assert masks.shape[0] == 2
    assert masks.shape[2] == 3 # [M_F0, M_jitter, M_shimmer]
