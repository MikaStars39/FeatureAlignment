# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

login(token="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")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")