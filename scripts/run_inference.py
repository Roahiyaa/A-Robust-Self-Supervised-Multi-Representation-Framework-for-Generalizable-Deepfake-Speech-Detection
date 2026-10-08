import argparse
import os
import sys
import torch
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config

from src.utils.logger import setup_logger
from src.data.synthetic_test_harness import create_synthetic_audio_signal
from src.models.full_pipeline import DeepfakeSpeechDetectionPipeline
from src.evaluation.uncertainty_ood import UncertaintyAndOODDetector
from src.inference.forensic_reporter import ForensicReporter

def main():
    parser = argparse.ArgumentParser(description="Run Deepfake Speech Detector Inference & Generate Forensic Report")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--audio_id", type=str, default="sample_audio_001")
    parser.add_argument("--out_dir", type=str, default="reports")
    args = parser.parse_args()

    logger = setup_logger("run_inference")
    device = torch.device("cpu")

    model = DeepfakeSpeechDetectionPipeline(embed_dim=64).to(device)
    model.eval()

    detector = UncertaintyAndOODDetector(num_mc_passes=5)
    reporter = ForensicReporter()

    wav = create_synthetic_audio_signal(duration_sec=3.0, is_fake=True)
    wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(wav_tensor)
        logits = outputs["binary"]
        raw_score = logits[0, 1].item() - logits[0, 0].item()
        prob_fake = torch.softmax(logits, dim=-1)[0, 1].item()
        energy_score = detector.compute_energy_score(logits)[0].item()

    mean_probs, uncertainty_var = detector.compute_mc_dropout_uncertainty(model, wav_tensor)
    var_val = uncertainty_var[0].item()

    report = reporter.generate_report(
        file_id=args.audio_id,
        raw_score=raw_score,
        calibrated_prob=prob_fake,
        uncertainty_var=var_val,
        energy_ood_score=energy_score
    )

    out_file = os.path.join(args.out_dir, f"{args.audio_id}_report.md")
    reporter.save_report_markdown(report, out_file)
    logger.info(f"Forensic Report generated successfully at: {out_file}")

if __name__ == "__main__":
    main()
