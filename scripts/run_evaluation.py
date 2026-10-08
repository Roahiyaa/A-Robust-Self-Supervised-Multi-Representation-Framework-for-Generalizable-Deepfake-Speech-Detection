import argparse
import os
import sys
import numpy as np
import torch

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data.synthetic_test_harness import create_synthetic_audio_signal, generate_synthetic_dataset_manifest
from src.models.full_pipeline import DeepfakeSpeechDetectionPipeline
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.robustness_evaluator import RobustnessTransformSuite

def main():
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Speech Detector")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logger("run_evaluation")

    device = torch.device("cpu")
    model = DeepfakeSpeechDetectionPipeline(embed_dim=64).to(device)
    model.eval()

    logger.info("Running evaluation protocol across benchmark datasets...")

    test_manifest = generate_synthetic_dataset_manifest(num_samples=20)
    y_true = []
    y_probs = []

    with torch.no_grad():
        for item in test_manifest:
            wav = create_synthetic_audio_signal(duration_sec=2.0, is_fake=(item["label"] == "fake"))
            wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(device)
            outputs = model(wav_tensor)
            prob_fake = torch.softmax(outputs["binary"], dim=-1)[0, 1].item()
            
            y_true.append(item["label_id"])
            y_probs.append(prob_fake)

    metrics = evaluate_predictions(np.array(y_true), np.array(y_probs))
    logger.info("=== Standard In-Domain Benchmark Results ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # Robustness suite test
    logger.info("=== Running Robustness Suite (Noise & Telephony) ===")
    noisy_probs = []
    with torch.no_grad():
        for item in test_manifest:
            wav = create_synthetic_audio_signal(duration_sec=2.0, is_fake=(item["label"] == "fake"))
            noisy_wav = RobustnessTransformSuite.add_noise(wav, snr_db=10.0)
            wav_tensor = torch.tensor(noisy_wav, dtype=torch.float32).unsqueeze(0).to(device)
            outputs = model(wav_tensor)
            noisy_probs.append(torch.softmax(outputs["binary"], dim=-1)[0, 1].item())

    robust_metrics = evaluate_predictions(np.array(y_true), np.array(noisy_probs))
    logger.info("=== Robustness (SNR 10dB Noise) Results ===")
    for k, v in robust_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()
