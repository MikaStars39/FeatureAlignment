from transformers import AutoTokenizer

def qwen_process_text(
    chosen: list,
    batch_size: int,
    tokenizer
):
    batch = []
    for idx in range(batch_size):
        each_batch = []
        for i in range(len(chosen)):
            content = chosen[i]['content'][idx]
            role = chosen[i]['role'][idx]
            each_batch.append({
                'content': content,
                'role': role
            })
        batch.append(tokenizer.apply_chat_template(each_batch, tokenize=False))
    return batch
        
        
        


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")

text =  [{'content': ['In journalctl, if the SystemMaxFiles is set is my .service file, does that apply to all services or just my service?'], 'role': ['user']}, {'content': ["The SystemMaxFiles entry only affects the maximum number of files that the kernel can create. If your service also sets the SystemMaxFiles limit, then only your service will be affected by it. If your service does not set the SystemMaxFiles limit, then it will be unaffected by the kernel's limit."], 'role': ['assistant']}]

print(tokenizer(qwen_process_text(
    text,
    1,
    tokenizer
)))