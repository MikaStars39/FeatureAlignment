import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

qwen_chat_template = "<|im_start|>user\n{}|im_end|>\n<|im_start|>assistant\n"

def get_chat_template(model_name):
    if "Qwen" in model_name:
        return qwen_chat_template
    else:
        raise ValueError(f"Model {model_name} not supported with chat template")

def load_and_generate_llm(
    model_name,
    repeat_times: int,
    prefilled: int = 3,
    total_times: int = 100,
    batch_size: int = 16,
    max_length: int = 100,
    seed: int = 31,
    token: str = "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",
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # set the seed
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
        random.seed(seed)

    
    from huggingface_hub import login
    login(token=token)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    from feature_alignment.benchmark.repeat import build_question, counting_that_token

    total_count = 0
    highest_length = 0
    for batch_start in tqdm(range(0, total_times, batch_size)):
        batch_end = min(batch_start + batch_size, total_times)
        batch_size = batch_end - batch_start

        inputs = []
        target_tokens = []
        for _ in range(batch_size):
            prompt, token = build_question(
                tokenizer, 
                repeat_times=repeat_times,
                chat_template=get_chat_template(model_name),
                prefilled=prefilled
            )
            encoded_prompt = tokenizer(prompt, return_tensors="pt").to(device)
            inputs.append(encoded_prompt)
            target_tokens.append(token)

        # Pad sequences to the same length
        max_len = max(inp['input_ids'].size(1) for inp in inputs)
        batch_inputs = {
            'input_ids': torch.stack(
                [torch.nn.functional.pad(inp['input_ids'].squeeze(0), (0, max_len - inp['input_ids'].size(1)
            )) for inp in inputs]),
            'attention_mask': torch.stack(
                [torch.nn.functional.pad(inp['attention_mask'].squeeze(0), 
                (0, max_len - inp['attention_mask'].size(1)
            )) for inp in inputs])
        }
        
        outputs = model.generate(**batch_inputs, max_new_tokens=max_length)
        
        for i in range(batch_size):
            result = outputs[i]
            count = counting_that_token(result, target_tokens[i])
            highest_length = max(highest_length, count)
            total_count += count
        
        print(tokenizer.decode(outputs[0]))

    average_count = total_count / total_times
    print("-"*20, "Experiment Details:", "-"*20)
    print(f"Model: {model_name}")
    print(f"Repeat Times: {repeat_times}")
    print(f"Total Times: {total_times}")
    print(f"Batch Size: {batch_size}")
    print(f"Max Length: {max_length}")
    print(f"Average Count: {average_count}")
    print(f"Highest Length: {highest_length}")
    return average_count

load_and_generate_llm(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    repeat_times=1024,
    total_times=8,
    batch_size=8,
    max_length=2048,
    token="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",
    prefilled=128,
)