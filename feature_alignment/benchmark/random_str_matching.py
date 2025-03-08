import torch
from transformer_lens import HookedTransformer
from transformer_lens import patching
from functools import partial

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

def patch(
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    clean_verb_token: int,
    corrupted_verb_token: int,
    model: HookedTransformer, 
    device: str,
):

    answer_tokens = torch.tensor([[clean_verb_token, corrupted_verb_token]], device=device)

    # check if the clean tokens and corrupted tokens are the same length
    min_len = min(clean_tokens.shape[1], corrupted_tokens.shape[1])
    clean_tokens = clean_tokens[:, :min_len]
    corrupted_tokens = corrupted_tokens[:, :min_len]

    clean_logits, clean_cache = model.run_with_cache(clean_tokens)
    corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

    clean_logit_diff = logits_to_ave_logit_diff(clean_logits, answer_tokens)
    corrupted_logit_diff = logits_to_ave_logit_diff(corrupted_logits, answer_tokens)

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

    return one_head_act_patch_result

def generate_prefix(model, device, k=20):
    # Generate a random string of length k
    import random
    import string
    # Include all printable characters except whitespace
    all_chars = string.digits + string.ascii_letters + string.punctuation
    
    # Choose a random position (not first or last) to test
    pos = random.randint(1, k-2)
    
    # First choose the first_token and remove it from available chars
    first_token = random.choice(all_chars)
    available_chars = all_chars.replace(first_token, '')
    
    # Generate rest of string without the first_token
    random_string = ''.join(random.choices(available_chars, k=k))
    
    # Insert first_token at pos
    random_string = random_string[:pos] + first_token + random_string[pos:]
    
    # Get the tokens we need
    second_token = random_string[pos+1]
    
    # Choose another random char that's not the second token for corruption
    available_chars = available_chars.replace(second_token, '')
    another_token = random.choice(available_chars)
    
    # Create clean and corrupted strings
    clean_prompt = random_string
    corrupted_prompt = random_string[:pos+1] + another_token + random_string[pos+2:]

    # Convert to tokens
    tokenizer = model.tokenizer
    clean_tokens = tokenizer(clean_prompt, return_tensors="pt").input_ids.to(device)
    corrupted_tokens = tokenizer(corrupted_prompt, return_tensors="pt").input_ids.to(device)

    # Get token IDs for the answers
    clean_verb_token = tokenizer(second_token, add_special_tokens=False).input_ids[0]
    corrupted_verb_token = tokenizer(another_token, add_special_tokens=False).input_ids[0]

    qwen_chat_template = "<|im_start|>user\nHere is a string: {}\
        \n The character after {} is ?<|im_end|>\n<|im_start|>assistant\n The answer is"
    
    clean_prompt = qwen_chat_template.format(clean_prompt, first_token)
    corrupted_prompt = qwen_chat_template.format(corrupted_prompt, first_token)

    return {
        "clean_tokens": clean_tokens,
        "corrupted_tokens": corrupted_tokens,
        "clean_verb_token": clean_verb_token,
        "corrupted_verb_token": corrupted_verb_token
    }

def find_induction_head(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    device: str = "cuda",
    times: int = 16,
    token: str = "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",
):
    # login
    if token:
        from huggingface_hub import login
        login(token=token)

    # set seed
    # random.seed(42)
    
    # Load model and SAE
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        torch_dtype=torch.bfloat16
    )

    total_result = []
    for i in range(times):
        prefix = generate_prefix(model, device)
        result = patch(
            prefix["clean_tokens"], 
            prefix["corrupted_tokens"], 
            prefix["clean_verb_token"], 
            prefix["corrupted_verb_token"], 
            model, 
            device
        )
        total_result.append(result)
    
    # average the result
    average_result = sum(total_result) / len(total_result)
    
    # Get indices of top 20 values
    flattened = average_result.flatten()
    top_50_values, top_50_indices = torch.topk(flattened, 50)
    
    # Convert flat indices to 2D indices
    rows = top_50_indices // average_result.shape[1]
    cols = top_50_indices % average_result.shape[1]
    
    # Print results in requested format
    print("\nTop 20 results (layer, head, value):")
    for i in range(50):
        print(f"({rows[i]}, {cols[i]}, {top_50_values[i]:.3f})")
    
def benchmarking_repetition(
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    device: str = "cuda",
    times: int = 16,
    token: str = "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",
):
    # login
    if token:
        from huggingface_hub import login
        login(token=token)

    # set seed
    # random.seed(42)
    
    # Load model and SAE
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        torch_dtype=torch.bfloat16
    )

    total_result = []
    for i in range(times):
        prefix = generate_prefix(model, device)
        result = patch(
            prefix["clean_tokens"], 
            prefix["corrupted_tokens"], 
            prefix["clean_verb_token"], 
            prefix["corrupted_verb_token"], 
            model, 
            device
        )
        total_result.append(result)
    
    # average the result
    average_result = sum(total_result) / len(total_result)
    
    # Get indices of top 20 values
    flattened = average_result.flatten()
    top_50_values, top_50_indices = torch.topk(flattened, 50)
    
    # Convert flat indices to 2D indices
    rows = top_50_indices // average_result.shape[1]
    cols = top_50_indices % average_result.shape[1]
    
    # Print results in requested format
    print("\nTop 20 results (layer, head, value):")
    for i in range(20):
        print(f"({rows[i]}, {cols[i]}, {top_50_values[i]:.3f})")