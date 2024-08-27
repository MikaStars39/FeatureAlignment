import torch
def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

def process_text(
    config,
    text,
    batch_size: int,
    tokenizer,
):
    if "Qwen1.5-0.5B" in config.model.name_or_path:
        return qwen_process_text(
            text,
            batch_size,
            tokenizer,
        )
    else: raise NotImplementedError(f"Model {config.model.name_or_path} not supported")

def qwen_process_text(
    chosen: list,
    batch_size: int,
    tokenizer,
):
    batch = []
    if len(chosen[0]['content']) < batch_size:
        batch_size = len(chosen[0]['content'])
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