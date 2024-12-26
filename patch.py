import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE
import json
from functools import partial
from transformer_lens import patching
import matplotlib.pyplot as plt
import random
import einops

from data.construct_sample_data import (
    create_prompt_with_context,
    get_formal,
)

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


def draw_patching_results(
        patching_results, 
        token_labels, 
        patching_method,
    ):
    
    # Average results across all pairs and save
    avg_patching_results = torch.stack(patching_results).mean(dim=0)
    torch.save(avg_patching_results, f'outputs/patching_results_{patching_method}.pt')
    
    if patching_method == "every":
        # Rearrange results for plotting
        avg_patching_results = einops.rearrange(avg_patching_results, "act_type layer pos head -> act_type (layer head) pos")
        
        # Plot and save results
        plt.figure(figsize=(20, 15))
        for i, label in enumerate(["Output", "Query", "Key", "Value", "Pattern"]):
            plt.subplot(5, 1, i+1)
            plt.imshow(avg_patching_results[i].cpu().numpy(), cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
            plt.colorbar()
            plt.title(f'{label} Activation Patching Per Head (By Pos)')
            plt.xlabel('Position')
            plt.ylabel('Layer & Head')
            plt.xticks(range(len(token_labels)), token_labels, rotation=90)
        
        plt.tight_layout()
        plt.savefig(f'patching_results_{patching_method}.png', bbox_inches='tight')
        plt.close()
    elif patching_method == "attn_head":
        # Plot and save results
        # patching results is [layer, pos, head]
        # token_labels is [pos, token]
        # so we need to rearrange patching results to [layer, head, pos]
        avg_patching_results = einops.rearrange(avg_patching_results[0], "layer pos head -> layer head pos")
        plt.figure(figsize=(20, 15))
        plt.imshow(avg_patching_results.cpu().numpy(), cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
        plt.colorbar()
        plt.title("attn_head_out Activation Patching By Pos")
        plt.xlabel("Position")
        plt.ylabel("Head")
        plt.xticks(range(len(token_labels)), token_labels, rotation=90)
        plt.tight_layout()
        plt.savefig(f'patching_results_{patching_method}.png', bbox_inches='tight')
        plt.close()

def main():
    # login
    from huggingface_hub import login
    login(token="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")

    # Configuration
    # model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_name = "google/gemma-2-2b-it"
    model_type = "gemma" if "gemma" in model_name.lower() else "llama"
    release = "llama_scope_lxm_8x"
    sae_id = "l31m_8x"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_pairs = 1  # Increased number of pairs to process
    num_context = 2  # Number of context pairs
    use_formal = True
    formal_type = "translation"
    patching_method = "attn_head"

    # set seed
    random.seed(42)
    
    # Load model and SAE
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        torch_dtype=torch.bfloat16
    )
    # sae = SAE.from_pretrained(
    #     release,
    #     sae_id,
    #     device=device,
    # )[0].to(torch.bfloat16)

    # Load noun-verb pairs
    with open('data/message.json', 'r') as f:
        pairs = json.load(f)
        pairs = random.sample(pairs, num_pairs * num_context + 1)
    
    all_patching_results = []
    
    # Process each pair (except the last one)
    for i in range(num_pairs):
        # Create clean and corrupted prompts with context
        if use_formal:
            clean_prompt, clean_verb, corrupted_prompt, corrupted_verb = get_formal(
                model_type=model_type, 
                formal_type=formal_type,
            )
        else:
            clean_prompt, clean_verb = create_prompt_with_context(pairs, i, num_context, model_type=model_type)
            corrupted_prompt, corrupted_verb = create_prompt_with_context(pairs, -1, num_context, corrupted=True, model_type=model_type)
        
        # Tokenize prompts
        clean_tokens = model.to_tokens(clean_prompt).to(device)
        corrupted_tokens = model.to_tokens(corrupted_prompt).to(device)
        
        # Get verb tokens
        print(clean_verb)
        print(corrupted_verb)
        if use_formal and (formal_type == "math" or formal_type == "translation"):
            # gemma has 3 tokens for the verb
            clean_verb_token = model.to_tokens(clean_verb)[0, 1]
            corrupted_verb_token = model.to_tokens(corrupted_verb)[0, 1]
        else:
            clean_verb_token = model.to_tokens(clean_verb)[0, -1]
            corrupted_verb_token = model.to_tokens(corrupted_verb)[0, -1]
        answer_tokens = torch.tensor([[clean_verb_token, corrupted_verb_token]], device=device)

        # check if the clean tokens and corrupted tokens are the same length
        min_len = min(clean_tokens.shape[1], corrupted_tokens.shape[1])
        clean_tokens = clean_tokens[:, :min_len]
        corrupted_tokens = corrupted_tokens[:, :min_len]

        print(clean_tokens)
        print(corrupted_tokens)
        print(answer_tokens)

        # Run model with cache
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)
        
        # Calculate baseline logit differences
        clean_logit_diff = logits_to_ave_logit_diff(clean_logits, answer_tokens)
        corrupted_logit_diff = logits_to_ave_logit_diff(corrupted_logits, answer_tokens)

        print(clean_logit_diff, corrupted_logit_diff)
        
        # Get patching results
        if patching_method == "every":
            every_head_act_patch_result = patching.get_act_patch_attn_head_by_pos_every(
                model,
                corrupted_tokens,
                clean_cache,
                partial(
                ioi_metric,
                answer_tokens=answer_tokens,
                clean_logit_diff=clean_logit_diff,
                corrupted_logit_diff=corrupted_logit_diff
            )
            )
            all_patching_results.append(every_head_act_patch_result)
        elif patching_method == "attn_head":
            one_head_act_patch_result = patching.get_act_patch_attn_head_out_all_pos(
                model,
                corrupted_tokens,
                clean_cache,
                partial(
                ioi_metric,
                answer_tokens=answer_tokens,
                clean_logit_diff=clean_logit_diff,
                corrupted_logit_diff=corrupted_logit_diff
                )
            )
            print(one_head_act_patch_result)
            all_patching_results.append(one_head_act_patch_result)

    tokens = model.to_str_tokens(clean_tokens[0])
    token_labels = [f"{tok}_{i}" for i, tok in enumerate(tokens)]
    
    draw_patching_results(all_patching_results, token_labels, patching_method)

if __name__ == "__main__":
    main()