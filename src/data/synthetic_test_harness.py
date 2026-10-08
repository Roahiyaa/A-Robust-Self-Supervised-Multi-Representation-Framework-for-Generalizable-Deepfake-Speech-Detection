import numpy as np
import torch
from typing import Dict, List, Tuple

def create_synthetic_audio_signal(
    duration_sec: float = 4.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
    is_fake: bool = False
) -> np.ndarray:
    """
    Generates a synthetic audio signal (sine wave + noise) for unit testing pipeline execution.
    NOTE: Used ONLY for system testing and integration checks, NEVER for research results.
    """
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    if is_fake:
        # Add high-frequency artifact to simulate synthetic voice anomaly
        signal += 0.1 * np.sin(2 * np.pi * 4000.0 * t)
        
    noise = np.random.normal(0, 0.01, signal.shape)
    return (signal + noise).astype(np.float32)

def generate_synthetic_dataset_manifest(num_samples: int = 20) -> List[Dict[str, str]]:
    """
    Generates a metadata manifest for synthetic unit tests.
    """
    manifest = []
    generators = ["real", "tts_v1", "tts_v2", "vc_v1"]
    
    for i in range(num_samples):
        is_fake = (i % 2 == 1)
        gen_id = "real" if not is_fake else generators[(i % 3) + 1]
        speaker_id = f"spk_{i // 2:03d}"
        file_id = f"synth_audio_{i:04d}"
        
        manifest.append({
            "file_id": file_id,
            "speaker_id": speaker_id,
            "generator_id": gen_id,
            "label": "fake" if is_fake else "real",
            "label_id": 1 if is_fake else 0
        })
        
    return manifest
