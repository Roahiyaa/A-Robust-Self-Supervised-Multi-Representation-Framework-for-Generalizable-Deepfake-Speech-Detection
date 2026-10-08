import torch
import torch.nn as nn
import numpy as np
import librosa
from typing import Optional, Dict, Any

class SpectralExtractor(nn.Module):
    """
    Branch B: Configurable Spectral Representation Extractor.
    Supports LFCC, Log-Mel Spectrogram, and CQT as configurable alternatives.
    Uses robust librosa and scipy computations for cross-platform stability.
    """
    def __init__(
        self,
        spectral_type: str = "lfcc",
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        n_lfcc: int = 20
    ):
        super().__init__()
        self.spectral_type = spectral_type.lower()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_lfcc = n_lfcc

    def _linear_filter_bank(self, sr: int, n_fft: int, n_filters: int) -> np.ndarray:
        # Create linearly spaced triangular filterbank matrix
        fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
        filter_freqs = np.linspace(0, sr / 2, n_filters + 2)
        weights = np.zeros((n_filters, n_fft // 2 + 1))
        for i in range(n_filters):
            f_m1, f_m, f_p1 = filter_freqs[i], filter_freqs[i + 1], filter_freqs[i + 2]
            down_slope = (fft_freqs - f_m1) / (f_m - f_m1 + 1e-6)
            up_slope = (f_p1 - fft_freqs) / (f_p1 - f_m + 1e-6)
            weights[i] = np.maximum(0, np.minimum(down_slope, up_slope))
        return weights

    def _compute_deltas(self, feat: np.ndarray) -> np.ndarray:
        delta = librosa.feature.delta(feat)
        delta2 = librosa.feature.delta(feat, order=2)
        return np.concatenate([feat, delta, delta2], axis=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: Raw waveform x of shape (B, T_samples)
        Output: Feature map of shape (B, 1, F, T_frames)
        """
        device = x.device
        x_np = x.detach().cpu().numpy()
        batch_out = []
        
        for wav in x_np:
            if self.spectral_type == "logmel":
                mel = librosa.feature.melspectrogram(
                    y=wav, sr=self.sample_rate, n_fft=self.n_fft,
                    win_length=self.win_length, hop_length=self.hop_length, n_mels=self.n_mels
                )
                feat = np.log(mel + 1e-6)
            elif self.spectral_type == "lfcc":
                # Linear-frequency filter bank
                S = np.abs(librosa.stft(y=wav, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length))
                lin_fb = self._linear_filter_bank(sr=self.sample_rate, n_fft=self.n_fft, n_filters=self.n_mels)
                lin_spec = np.dot(lin_fb, S)
                log_lin_spec = np.log(lin_spec + 1e-6)
                lfcc_static = librosa.feature.mfcc(S=log_lin_spec, n_mfcc=self.n_lfcc)
                feat = self._compute_deltas(lfcc_static)


            elif self.spectral_type == "cqt":
                cqt = np.abs(librosa.cqt(y=wav, sr=self.sample_rate, hop_length=self.hop_length, n_bins=84))
                feat = np.log(cqt + 1e-6)
            else:
                raise ValueError(f"Unsupported spectral_type: {self.spectral_type}")
                
            batch_out.append(feat)
            
        stacked = np.stack(batch_out) # (B, F, T)
        tensor_out = torch.tensor(stacked, dtype=torch.float32, device=device)
        return tensor_out.unsqueeze(1) # (B, 1, F, T)

