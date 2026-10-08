import torch
import torch.nn as nn
import numpy as np
import librosa
from typing import Dict, Tuple

class AcousticExtractor(nn.Module):
    """
    Branch C: Acoustic and Forensic Features Extractor.
    Extracts spectral statistics (centroid, bandwidth, roll-off, flatness),
    temporal dynamics (ZCR, short-time energy), and voice quality metrics (F0, jitter, shimmer).
    Includes explicit validity mask flags (M_F0, M_jitter, M_shimmer) to explicitly signal unvoiced/unreliable frames.
    """
    def __init__(self, sample_rate: int = 16000, frame_length: int = 400, hop_length: int = 160):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input: Waveform tensor of shape (B, T_samples)
        Returns:
            features: Tensor of shape (B, T_frames, D_acoustic)
            masks: Validity mask tensor of shape (B, T_frames, 3) representing [M_F0, M_jitter, M_shimmer]
        """
        device = x.device
        x_np = x.detach().cpu().numpy()
        batch_feats = []
        batch_masks = []
        
        for wav in x_np:
            # 1. Spectral features
            cent = librosa.feature.spectral_centroid(y=wav, sr=self.sample_rate, hop_length=self.hop_length)[0]
            bw = librosa.feature.spectral_bandwidth(y=wav, sr=self.sample_rate, hop_length=self.hop_length)[0]
            rolloff = librosa.feature.spectral_rolloff(y=wav, sr=self.sample_rate, hop_length=self.hop_length)[0]
            flatness = librosa.feature.spectral_flatness(y=wav, hop_length=self.hop_length)[0]
            
            # 2. Temporal features
            zcr = librosa.feature.zero_crossing_rate(y=wav, frame_length=self.frame_length, hop_length=self.hop_length)[0]
            ste = librosa.feature.rms(y=wav, frame_length=self.frame_length, hop_length=self.hop_length)[0]
            
            # 3. Voice Quality & Pitch (F0, Jitter, Shimmer)
            f0, voiced_flag, _ = librosa.pyin(
                y=wav, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate, frame_length=self.frame_length, hop_length=self.hop_length
            )
            f0_clean = np.nan_to_num(f0, nan=0.0)
            m_f0 = np.nan_to_num(voiced_flag, nan=0.0).astype(np.float32)
            
            # Approximate Jitter & Shimmer per frame
            f0_diff = np.abs(np.diff(f0_clean, prepend=f0_clean[0]))
            jitter = np.where(f0_clean > 0, f0_diff / (f0_clean + 1e-6), 0.0)
            m_jitter = (f0_clean > 0).astype(np.float32)
            
            ste_diff = np.abs(np.diff(ste, prepend=ste[0]))
            shimmer = np.where(ste > 0, ste_diff / (ste + 1e-6), 0.0)
            m_shimmer = (ste > 0).astype(np.float32)
            
            # Stack numeric features
            # Shape of each array is (T_frames,)
            min_len = min(len(cent), len(bw), len(rolloff), len(flatness), len(zcr), len(ste), len(f0_clean))
            
            feats = np.stack([
                cent[:min_len], bw[:min_len], rolloff[:min_len], flatness[:min_len],
                zcr[:min_len], ste[:min_len], f0_clean[:min_len], jitter[:min_len], shimmer[:min_len]
            ], axis=-1) # (T_frames, 9)
            
            masks = np.stack([
                m_f0[:min_len], m_jitter[:min_len], m_shimmer[:min_len]
            ], axis=-1) # (T_frames, 3)
            
            batch_feats.append(feats)
            batch_masks.append(masks)
            
        feats_tensor = torch.tensor(np.stack(batch_feats), dtype=torch.float32, device=device)
        masks_tensor = torch.tensor(np.stack(batch_masks), dtype=torch.float32, device=device)
        return feats_tensor, masks_tensor
