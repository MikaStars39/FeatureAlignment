import torch
import transformers
import json
from tqdm import tqdm

from .test_refusal import check_refusal
def test_safety(
    result: list[tuple[str, str]],
):
    total_clean_refusal = 0
    total_corrupted_refusal = 0
    print("\n" + "="*60)
    print(f"{'Layer':^10}|{'Clean Refusal':^15}|{'Clean ASR':^15}|{'Corrupt Refusal':^15}|{'Corrupt ASR':^15}")
    print("="*60)
    
    for layer_idx in range(len(result)):
        clean_output = result[layer_idx]["clean_output"]
        corrupted_output = result[layer_idx]["corrupted_output"]
        
        # check refusal
        clean_refusal = check_refusal(clean_output)
        corrupted_refusal = check_refusal(corrupted_output)
        
        # Calculate ASR (Attack Success Rate)
        clean_asr = 1 - clean_refusal
        corrupted_asr = 1 - corrupted_refusal
        
        total_clean_refusal += clean_refusal
        total_corrupted_refusal += corrupted_refusal
        
        print(f"{layer_idx:^10}|{clean_refusal:^15.3f}|{clean_asr:^15.3f}|{corrupted_refusal:^15.3f}|{corrupted_asr:^15.3f}")
    
    print("-"*60)
    # Calculate averages
    avg_clean_refusal = total_clean_refusal / len(result)
    avg_corrupted_refusal = total_corrupted_refusal / len(result)
    avg_clean_asr = 1 - avg_clean_refusal
    avg_corrupted_asr = 1 - avg_corrupted_refusal
    
    print(f"{'Average':^10}|{avg_clean_refusal:^15.3f}|{avg_clean_asr:^15.3f}|{avg_corrupted_refusal:^15.3f}|{avg_corrupted_asr:^15.3f}")
    print("="*60 + "\n")

    
