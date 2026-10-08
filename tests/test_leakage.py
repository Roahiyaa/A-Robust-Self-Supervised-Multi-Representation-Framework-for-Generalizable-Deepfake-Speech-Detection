import pytest
from src.data.leakage_checker import verify_dataset_splits

def test_leakage_checker_clean():
    train_meta = [
        {"file_id": "f1", "speaker_id": "spk1", "generator_id": "real"},
        {"file_id": "f2", "speaker_id": "spk2", "generator_id": "gen1"}
    ]
    val_meta = [
        {"file_id": "f3", "speaker_id": "spk3", "generator_id": "real"},
        {"file_id": "f4", "speaker_id": "spk4", "generator_id": "gen1"}
    ]
    test_meta = [
        {"file_id": "f5", "speaker_id": "spk5", "generator_id": "real"},
        {"file_id": "f6", "speaker_id": "spk6", "generator_id": "gen2"}
    ]
    
    clean, violations = verify_dataset_splits(train_meta, val_meta, test_meta, check_generators=True)
    assert clean is True
    assert len(violations) == 0

def test_leakage_checker_speaker_leak():
    train_meta = [{"file_id": "f1", "speaker_id": "spk1", "generator_id": "real"}]
    val_meta = [{"file_id": "f2", "speaker_id": "spk1", "generator_id": "real"}]
    test_meta = [{"file_id": "f3", "speaker_id": "spk2", "generator_id": "real"}]
    
    clean, violations = verify_dataset_splits(train_meta, val_meta, test_meta)
    assert clean is False
    assert any("Speaker ID leakage" in v for v in violations)

def test_leakage_checker_generator_leak():
    train_meta = [{"file_id": "f1", "speaker_id": "spk1", "generator_id": "gen1"}]
    val_meta = [{"file_id": "f2", "speaker_id": "spk2", "generator_id": "gen1"}]
    test_meta = [{"file_id": "f3", "speaker_id": "spk3", "generator_id": "gen1"}]
    
    clean, violations = verify_dataset_splits(train_meta, val_meta, test_meta, check_generators=True)
    assert clean is False
    assert any("Generator overlap" in v for v in violations)
