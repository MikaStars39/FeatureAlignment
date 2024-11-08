from transformers import AutoModel
from huggingface_hub import login

login(token="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")
model = AutoModel.from_pretrained("google/gemma-2-2b")