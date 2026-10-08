import numpy as np
import scipy.signal as signal
from typing import Callable, Dict

class RobustnessTransformSuite:
    """
    Simulates real-world audio distortions:
    1. Additive Gaussian Noise (SNR-controlled)
    2. Resampling & Bandwidth limitation (300Hz - 3400Hz telephone filter)
    3. Simulated Room Impulse Response (Reverberation)
    4. Clipping & Quantization Distortion
    """
    @staticmethod
    def add_noise(waveform: np.ndarray, snr_db: float = 15.0) -> np.ndarray:
        signal_power = np.mean(waveform ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10.0))
        noise = np.random.normal(0, np.sqrt(noise_power + 1e-8), waveform.shape)
        return (waveform + noise).astype(np.float32)

    @staticmethod
    def telephone_bandpass(waveform: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        # Bandpass 300Hz - 3400Hz (Standard G.711 Telephony channel)
        nyquist = 0.5 * sample_rate
        low = 300.0 / nyquist
        high = 3400.0 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.lfilter(b, a, waveform)
        return filtered.astype(np.float32)

    @staticmethod
    def simulate_reverb(waveform: np.ndarray, decay_factor: float = 0.4) -> np.ndarray:
        # Simple FIR impulse response simulation
        impulse = np.array([1.0, 0.0, decay_factor, 0.0, decay_factor**2])
        reverberated = signal.fftconvolve(waveform, impulse, mode='same')
        return reverberated.astype(np.float32)
