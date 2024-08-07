import torch
from src.sae import replace_sae_with_reture_feature_acts

sae_encoder = replace_sae_with_reture_feature_acts()
sae, cfg_dict, sparsity = sae_encoder.from_pretrained(
    release = "gemma-2b-res-jb",
    sae_id = "blocks.12.hook_resid_post",
)

input_tensor = torch.zeros(1, 1, 2048)

print(sae.encode(input_tensor))