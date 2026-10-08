from typing import Dict, List, Set, Tuple

def verify_dataset_splits(
    train_metadata: List[Dict[str, str]],
    val_metadata: List[Dict[str, str]],
    test_metadata: List[Dict[str, str]],
    check_generators: bool = False
) -> Tuple[bool, List[str]]:
    """
    Strictly verifies that no speaker, recording, or duplicate file leakage occurs
    across training, validation, and evaluation splits.
    
    Each item in metadata list must be a dict containing:
      - 'file_id': unique recording identifier
      - 'speaker_id': speaker identifier
      - 'generator_id' (optional): synthetic algorithm tag
    """
    violations = []
    
    # 1. Recording ID Leakage Check
    train_files = {item['file_id'] for item in train_metadata}
    val_files = {item['file_id'] for item in val_metadata}
    test_files = {item['file_id'] for item in test_metadata}
    
    file_overlap_tr_val = train_files.intersection(val_files)
    file_overlap_tr_te = train_files.intersection(test_files)
    file_overlap_val_te = val_files.intersection(test_files)
    
    if file_overlap_tr_val:
        violations.append(f"Recording ID leakage between Train and Val: {len(file_overlap_tr_val)} files overlap.")
    if file_overlap_tr_te:
        violations.append(f"Recording ID leakage between Train and Test: {len(file_overlap_tr_te)} files overlap.")
    if file_overlap_val_te:
        violations.append(f"Recording ID leakage between Val and Test: {len(file_overlap_val_te)} files overlap.")

    # 2. Speaker ID Leakage Check
    train_speakers = {item['speaker_id'] for item in train_metadata if item.get('speaker_id')}
    val_speakers = {item['speaker_id'] for item in val_metadata if item.get('speaker_id')}
    test_speakers = {item['speaker_id'] for item in test_metadata if item.get('speaker_id')}
    
    spk_overlap_tr_val = train_speakers.intersection(val_speakers)
    spk_overlap_tr_te = train_speakers.intersection(test_speakers)
    spk_overlap_val_te = val_speakers.intersection(test_speakers)
    
    if spk_overlap_tr_val:
        violations.append(f"Speaker ID leakage between Train and Val: {len(spk_overlap_tr_val)} speakers overlap.")
    if spk_overlap_tr_te:
        violations.append(f"Speaker ID leakage between Train and Test: {len(spk_overlap_tr_te)} speakers overlap.")
    if spk_overlap_val_te:
        violations.append(f"Speaker ID leakage between Val and Test: {len(spk_overlap_val_te)} speakers overlap.")

    # 3. Generator Leakage Check (For Unseen Generator Generalization Experiments)
    if check_generators:
        train_gens = {item['generator_id'] for item in train_metadata if item.get('generator_id') and item['generator_id'] != 'real'}
        test_gens = {item['generator_id'] for item in test_metadata if item.get('generator_id') and item['generator_id'] != 'real'}
        
        gen_overlap = train_gens.intersection(test_gens)
        if gen_overlap:
            violations.append(f"Generator overlap in zero-shot experiment: {gen_overlap} present in both Train and Test.")

    is_clean = len(violations) == 0
    return is_clean, violations
