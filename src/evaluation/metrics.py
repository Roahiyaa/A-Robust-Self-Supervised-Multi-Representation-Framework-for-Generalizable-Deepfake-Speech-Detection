import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc, brier_score_loss, accuracy_score, f1_score, precision_score, recall_score
from typing import Dict, Tuple

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Computes Equal Error Rate (EER) and the optimal decision threshold.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])
    return eer, eer_threshold

def compute_ece(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE).
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (y_probs >= bin_lower) & (y_probs < bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin] == (y_probs[in_bin] >= 0.5))
            avg_confidence_in_bin = np.mean(y_probs[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
            
    return float(ece)

def evaluate_predictions(y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, float]:
    """
    Computes comprehensive evaluation metrics suite:
    Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, EER, ECE, and Brier Score.
    """
    y_pred = (y_probs >= 0.5).astype(int)
    
    eer, eer_thresh = compute_eer(y_true, y_probs)
    roc_auc = float(roc_auc_score(y_true, y_probs))
    
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = float(auc(recall, precision))
    
    ece = compute_ece(y_true, y_probs)
    brier = float(brier_score_loss(y_true, y_probs))
    
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "eer": eer,
        "eer_threshold": eer_thresh,
        "ece": ece,
        "brier_score": brier
    }
