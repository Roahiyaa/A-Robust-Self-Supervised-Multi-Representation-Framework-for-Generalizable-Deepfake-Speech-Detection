import pytest
import numpy as np
from src.data.synthetic_test_harness import create_synthetic_audio_signal, generate_synthetic_dataset_manifest

def test_synthetic_audio_signal():
    sig = create_synthetic_audio_signal(duration_sec=2.0, sample_rate=16000)
    assert isinstance(sig, np.ndarray)
    assert len(sig) == 32000

def test_synthetic_dataset_manifest():
    manifest = generate_synthetic_dataset_manifest(num_samples=10)
    assert len(manifest) == 10
    assert "file_id" in manifest[0]
    assert "speaker_id" in manifest[0]
    assert "generator_id" in manifest[0]
