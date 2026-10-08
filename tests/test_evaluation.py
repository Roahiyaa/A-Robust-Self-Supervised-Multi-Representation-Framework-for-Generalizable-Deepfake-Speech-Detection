import pytest
import torch
import numpy as np
from src.evaluation.calibration import TemperatureScaler
from src.evaluation.metrics import evaluate_predictions, compute_eer, compute_ece
from src.evaluation.uncertainty_ood import UncertaintyAndOODDetector
from src.inference.forensic_reporter import ForensicReporter

def test_temperature_scaler():
    scaler = TemperatureScaler()
    logits = torch.tensor([[2.0, -1.0], [-1.0, 3.0]])
    labels = torch.tensor([0, 1])
    temp = scaler.fit(logits, labels, max_iter=5)
    assert temp > 0
    scaled = scaler(logits)
    assert scaled.shape == logits.shape

def test_metrics_evaluation():
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_probs = np.array([0.1, 0.2, 0.8, 0.9, 0.75, 0.3])
    metrics = evaluate_predictions(y_true, y_probs)
    assert "accuracy" in metrics
    assert "eer" in metrics
    assert "ece" in metrics
    assert "brier_score" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

def test_forensic_reporter(tmp_path):
    reporter = ForensicReporter()
    report = reporter.generate_report(
        file_id="test_audio_001",
        raw_score=1.5,
        calibrated_prob=0.85,
        uncertainty_var=0.002,
        energy_ood_score=1.2
    )
    assert report["statistical_predictions"]["predicted_class"] == "FAKE"
    
    md_file = tmp_path / "report.md"
    reporter.save_report_markdown(report, str(md_file))
    assert md_file.exists()
