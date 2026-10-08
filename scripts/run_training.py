import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader, Dataset

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.config import load_config

from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.data.synthetic_test_harness import create_synthetic_audio_signal, generate_synthetic_dataset_manifest
from src.models.full_pipeline import DeepfakeSpeechDetectionPipeline
from src.training.losses import MultiTaskLoss
from src.training.trainer import Trainer

class SyntheticDataset(Dataset):
    def __init__(self, manifest):
        self.manifest = manifest

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        item = self.manifest[idx]
        wav = create_synthetic_audio_signal(duration_sec=2.0, is_fake=(item["label"] == "fake"))
        return {
            "waveform": torch.tensor(wav, dtype=torch.float32),
            "label": torch.tensor(item["label_id"], dtype=torch.long)
        }

def main():
    parser = argparse.ArgumentParser(description="Train Deepfake Speech Detector")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["system"]["seed"])
    logger = setup_logger("run_training")

    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Synthetic manifest for dry run / test verification
    train_manifest = generate_synthetic_dataset_manifest(num_samples=16)
    val_manifest = generate_synthetic_dataset_manifest(num_samples=8)

    train_loader = DataLoader(SyntheticDataset(train_manifest), batch_size=4, shuffle=True)
    val_loader = DataLoader(SyntheticDataset(val_manifest), batch_size=4, shuffle=False)

    model = DeepfakeSpeechDetectionPipeline(
        embed_dim=config["fusion"]["embed_dim"],
        ssl_model_name=config["features"]["ssl"]["model_name"],
        ssl_strategy=config["features"]["ssl"]["finetune_strategy"],
        spectral_type=config["features"]["spectral"]["type"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"])
    loss_fn = MultiTaskLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        save_dir="checkpoints"
    )

    logger.info("Starting training loop...")
    for epoch in range(1, config["training"]["epochs"] + 1):
        train_res = trainer.train_epoch(train_loader)
        val_res = trainer.evaluate(val_loader)
        logger.info(f"Epoch [{epoch}/{config['training']['epochs']}] - Train Loss: {train_res['train_loss']:.4f} | Val Loss: {val_res['val_loss']:.4f}")

    trainer.save_checkpoint(config["training"]["epochs"], "checkpoints/final_model.pt")
    logger.info("Training complete successfully!")

if __name__ == "__main__":
    main()
