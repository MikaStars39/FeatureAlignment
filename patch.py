import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
import json
from functools import partial
from transformer_lens import patching
import matplotlib.pyplot as plt

def logits_to_ave_logit_diff(
    logits: torch.Tensor,
    answer_tokens: torch.Tensor,
    per_prompt: bool = False
) -> torch.Tensor:
    '''
    Returns logit difference between the correct and incorrect answer.
    answer_tokens should contain [correct_token, incorrect_token]
    '''
    final_logits = logits[:, -1, :]
    answer_logits = final_logits.gather(dim=-1, index=answer_tokens)
    correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
    answer_logit_diff = correct_logits - incorrect_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()

def ioi_metric(
    logits: torch.Tensor, 
    answer_tokens: torch.Tensor,
    corrupted_logit_diff: float,
    clean_logit_diff: float,
) -> float:
    patched_logit_diff = logits_to_ave_logit_diff(logits, answer_tokens)
    return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)

def main():
    # Configuration
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    release = "llama_scope_lxm_8x"
    sae_id = "l31m_8x"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_pairs = 5  # Number of pairs to process
    
    # Load model and SAE
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        torch_dtype=torch.bfloat16
    )
    sae = SAE.from_pretrained(
        release,
        sae_id,
        device=device,
    )[0].to(torch.bfloat16)

    # Load noun-verb pairs
    with open('data/message.json', 'r') as f:
        pairs = json.load(f)[:num_pairs+1]  # Get pairs plus one extra for corruption
    
    template = [
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ]
    
    all_patching_results = []
    
    # Process each pair (except the last one)
    for i in range(num_pairs):
        # Create clean and corrupted prompts
        clean_prompt = template[0] + pairs[i]['noun'] + template[1]
        corrupted_prompt = template[0] + pairs[-1]['noun'] + template[1]  # Always use last pair as corruption
        
        # Tokenize prompts
        clean_tokens = model.to_tokens(clean_prompt).to(device)
        corrupted_tokens = model.to_tokens(corrupted_prompt).to(device)
        
        # Get verb tokens
        clean_verb_token = model.to_tokens(pairs[i]['verb'])[0, -1]
        corrupted_verb_token = model.to_tokens(pairs[-1]['verb'])[0, -1]
        answer_tokens = torch.tensor([[clean_verb_token, corrupted_verb_token]], device=device)

        # Run model with cache
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)
        
        # Calculate baseline logit differences
        clean_logit_diff = logits_to_ave_logit_diff(clean_logits, answer_tokens)
        corrupted_logit_diff = logits_to_ave_logit_diff(corrupted_logits, answer_tokens)
        print(clean_logit_diff, corrupted_logit_diff)
        
        # Get patching results
        act_patch_resid_pre = patching.get_act_patch_resid_pre(
            model=model,
            corrupted_tokens=corrupted_tokens,
            clean_cache=clean_cache,
            patching_metric=partial(
                ioi_metric,
                answer_tokens=answer_tokens,
                clean_logit_diff=clean_logit_diff,
                corrupted_logit_diff=corrupted_logit_diff
            )
        )

        print(act_patch_resid_pre)
        
        all_patching_results.append(act_patch_resid_pre)
    
    # Average results across all pairs
    avg_patching_results = torch.stack(all_patching_results).mean(dim=0)
    
    # Plot and save results
    plt.figure(figsize=(15, 10))
    plt.imshow(avg_patching_results.cpu().numpy(), cmap='RdBu', aspect='auto')
    plt.colorbar()
    plt.title('Average Patching Results Across {} Pairs'.format(num_pairs))
    plt.xlabel('Layer')
    plt.ylabel('Head')
    plt.savefig('patching_results.png')
    plt.close()

if __name__ == "__main__":
    main()