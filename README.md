# A Robust Self-Supervised Multi-Representation Framework for Generalizable Deepfake Speech Detection

A research-grade deepfake speech detection pipeline engineered in PyTorch. The framework integrates multi-branch feature representations, gated feature fusion, bidirectional cross-attention, temporal modeling, and probability calibration with out-of-distribution (OOD) detection.

---

## 1. Key Innovations & Architecture Overview

* **Branch A (SSL Speech Encoder)**: Pretrained `microsoft/wavlm-base-plus` (primary) and `facebook/wav2vec2-base` (comparison).
* **Branch B (Configurable Spectral Encoder)**: Configurable LFCC, Log-Mel spectrogram, and Constant-Q Transform (CQT) options with ResNet/CNN architectures.
* **Branch C (Acoustic & Forensic Features with Validity Flags)**: Spectral centroid, bandwidth, roll-off, ZCR, short-time energy, pitch ($F_0$), jitter, and shimmer paired with explicit validity flags $(M_{F0}, M_{\text{jitter}}, M_{\text{shimmer}})$.
* **Multi-Modal Gated Fusion & Bidirectional Cross-Attention**: Dynamically learns feature weights and cross-modal attention between SSL contextual features and spectral representations.
* **Temporal Transformer**: Sequence modeling across continuous audio frames and overlapping audio chunks.
* **Conditional Multi-Task Heads**: Utterance binary classification, dataset-conditional attack classification, and annotation-conditional frame/segment localization.
* **Calibration & OOD Detection**: Temperature scaling (fitted strictly on validation set), MC Dropout epistemic uncertainty, and Energy-based OOD detection.

---

## 2. Directory Layout

```
.
├── configs/
│   └── default_config.yaml
├── src/
│   ├── utils/          # Config, Logger, Random Seed utilities
│   ├── data/           # Dataset loading, Leakage checker, Synthetic test harness
│   ├── features/       # SSL, Spectral, and Acoustic feature extractors
│   ├── models/         # Gated fusion, Cross-attention, Temporal transformer, Multi-task heads, Full pipeline
│   ├── training/       # Loss functions, Trainer, Scheduler
│   ├── evaluation/     # Metrics (EER, ECE, ROC-AUC), Calibration, Uncertainty/OOD, Robustness suite
│   └── inference/      # Forensic report generation
├── tests/              # Automated unit and integration test suite
├── scripts/            # Training, Evaluation, and Inference entry points
├── requirements.txt
└── README.md
```

---

## 3. Quick Start (Windows PowerShell)

### Installation
```powershell
pip install -r requirements.txt
```

### Run Automated Unit Tests
```powershell
python -m pytest tests/
```

### Run Synthetic Training Verification
```powershell
python scripts/run_training.py --config configs/default_config.yaml
```

### Run Evaluation Suite & Robustness Matrix
```powershell
python scripts/run_evaluation.py --config configs/default_config.yaml
```

### Generate Forensic Inference Report
```powershell
python scripts/run_inference.py --audio_id sample_audio_001
```

---

## 4. Leakage Prevention Protocol

The system enforces strict dataset partitioning via `src/data/leakage_checker.py`:
1. **Speaker Isolation**: Ensures `train_speakers \cap test_speakers = \emptyset`.
2. **Recording Isolation**: Prevents overlapping audio chunks from the same audio file crossing train/test boundaries.
3. **Generator Isolation**: Enforces disjoint generator sets $G_{\text{train}} \cap G_{\text{test}} = \emptyset$ for zero-shot cross-generator generalization benchmarks.
