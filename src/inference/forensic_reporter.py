import json
import os
from typing import Dict, Any, Optional

class ForensicReporter:
    """
    Generates structured forensic reports for input audio files.
    Strictly distinguishes model statistical predictions from verified forensic facts.
    """
    def __init__(self, model_version: str = "1.0.0", threshold: float = 0.5):
        self.model_version = model_version
        self.threshold = threshold

    def generate_report(
        self,
        file_id: str,
        raw_score: float,
        calibrated_prob: float,
        uncertainty_var: float,
        energy_ood_score: float,
        attack_class: Optional[str] = None,
        segment_intervals: Optional[list] = None,
        audio_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        
        predicted_class = "FAKE" if calibrated_prob >= self.threshold else "REAL"
        is_ood = energy_ood_score > 5.0 # Energy threshold for OOD warning
        
        report = {
            "metadata": {
                "file_id": file_id,
                "model_version": self.model_version,
                "decision_threshold": self.threshold,
                "audio_info": audio_metadata or {}
            },
            "statistical_predictions": {
                "predicted_class": predicted_class,
                "raw_logits_score": float(raw_score),
                "calibrated_probability_fake": float(calibrated_prob),
                "mc_dropout_uncertainty_variance": float(uncertainty_var),
                "energy_ood_score": float(energy_ood_score),
                "out_of_distribution_warning": is_ood,
                "conditional_attack_category": attack_class,
                "suspicious_time_intervals": segment_intervals or []
            },
            "forensic_disclaimer": (
                "DISCLAIMER: The outputs above represent statistical inference from a deep learning model. "
                "Confidence scores and probabilistic measures do NOT constitute physical forensic proof of source authorship "
                "or specific speech generator identity unless verified by ground-truth acoustic signals."
            )
        }
        return report

    def save_report_markdown(self, report_dict: Dict[str, Any], output_path: str):
        meta = report_dict["metadata"]
        preds = report_dict["statistical_predictions"]
        
        md_content = f"""# Forensic Deepfake Audio Analysis Report

## Metadata
* **File ID**: `{meta['file_id']}`
* **Model Version**: `{meta['model_version']}`
* **Decision Threshold**: `{meta['decision_threshold']}`

## Model Inference & Predictions
* **Predicted Class**: **{preds['predicted_class']}**
* **Calibrated Probability (Fake)**: `{preds['calibrated_probability_fake']:.4f}`
* **Epistemic Uncertainty Variance**: `{preds['mc_dropout_uncertainty_variance']:.6f}`
* **OOD Energy Score**: `{preds['energy_ood_score']:.4f}`
* **OOD Warning Flag**: `{preds['out_of_distribution_warning']}`

---

> [!NOTE]
> {report_dict['forensic_disclaimer']}
"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)
