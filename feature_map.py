import torch
import tqdm
import os
from transformers import AutoTokenizer
from datasets import load_dataset
from huggingface_hub import hf_hub_download, login
import numpy as np
from utils import disable_dropout


@torch.no_grad()
def get_feature_map(
    model_name_or_path: str,
    device: str,
    sae_encoder_name_or_path: int,
    sae_layer_id: int,
    temperature: float = 1.0,
    visualize: bool = True,
    cache_dir: str = ".cache",
    batch_size: int = 8,
    total_samples: int = 1000,
    release: bool = True,
):
    # load safe.json
    dataset = load_dataset("json", data_files="safe.json", split="train")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=8)
    

    # login with Hugging Face token
    login(token="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")

    if release:
        path_to_params = hf_hub_download(
            repo_id="google/gemma-scope-2b-pt-res",
            filename="layer_12/width_16k/average_l0_82/params.npz",
            force_download=False,
        )

        params = np.load(path_to_params)
        pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}

        from transformers_model.modeling_gemma2 import JumpReLUSAE
        sae_encoder = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
        sae_encoder.load_state_dict(pt_params)

    cache_file_chosen = os.path.join(cache_dir, f"{model_name_or_path}_layer_{sae_layer_id}_chosen_feature_map.pt")
    cache_file_rejected = os.path.join(cache_dir, f"{model_name_or_path}_layer_{sae_layer_id}_rejected_feature_map.pt")
    
    if os.path.exists(cache_file_chosen) and os.path.exists(cache_file_rejected):
        print(f"Loading cached feature maps from {cache_file_chosen} and {cache_file_rejected}")
        chosen_feature_map = torch.load(cache_file_chosen)
        rejected_feature_map = torch.load(cache_file_rejected)
    else:
        if "gemma-2" in model_name_or_path:
            from transformers_model.modeling_gemma2 import Gemma2ForCausalLM
            model = Gemma2ForCausalLM.from_pretrained(
                model_name_or_path, 
                low_cpu_mem_usage=True, 
            )
        elif "Qwen1.5-0.5B" in model_name_or_path:
            from transformers_model.modeling_qwen2 import Qwen2ForCausalLM
            model = Qwen2ForCausalLM.from_pretrained(
                model_name_or_path,
                low_cpu_mem_usage=True,
            )
        else:
            raise NotImplementedError(f"Model {model_name_or_path} not supported")
        
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        disable_dropout(model)

        model.model.layers[sae_layer_id].set_encoder(sae_encoder)

        model.to(device)
        model.eval()

        chosen_feature_map = None
        rejected_feature_map = None
        
        for i, batch in tqdm.tqdm(enumerate(dataloader), desc="Getting Feature map"):
            if i * batch_size > total_samples:
                break
            
            chosen = batch["chosen"]
            rejected = batch["rejected"]

            for i in range(len(chosen)):
                chosen[i] = [
                    {
                        'content': chosen[i],
                        'role': 'user',
                    },
                    {
                        'content': chosen[i],
                        'role': 'assistant',
                    }
                ]
                rejected[i] = [
                    {
                        'content': rejected[i],
                        'role': 'user',
                    },
                    {
                        'content': rejected[i],
                        'role': 'assistant',
                    }
                ]

            chosen = tokenizer.apply_chat_template(
                chosen, 
                return_tensors="pt", 
                padding=True, 
                return_dict=True, 
                truncation=True,
                max_length=1024,
                padding_side="left", 
            )

            rejected = tokenizer.apply_chat_template(
                rejected,
                return_tensors="pt",
                padding=True,
                return_dict=True,
                truncation=True,
                max_length=1024,
                padding_side="left",
            )

            chosen['input_ids'] = chosen['input_ids'].to(device)
            chosen['attention_mask'] = chosen['attention_mask'].to(device)
            rejected['input_ids'] = rejected['input_ids'].to(device)
            rejected['attention_mask'] = rejected['attention_mask'].to(device)

            chosen_feature_acts_reference = model(**chosen, use_cache=False).feature_acts
            rejected_feature_acts_reference = model(**rejected, use_cache=False).feature_acts

            if chosen_feature_map is None:
                chosen_feature_map = chosen_feature_acts_reference.mean(dim=[0, 1]).detach()
            else:
                chosen_feature_map += chosen_feature_acts_reference.mean(dim=[0, 1]).detach()

            if rejected_feature_map is None:
                rejected_feature_map = rejected_feature_acts_reference.mean(dim=[0, 1]).detach()
            else:
                rejected_feature_map += rejected_feature_acts_reference.mean(dim=[0, 1]).detach()
        
        # chosen_feature_map = (chosen_feature_map / temperature).softmax(dim=-1)
        # rejected_feature_map = (rejected_feature_map / temperature).softmax(dim=-1)
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(chosen_feature_map, cache_file_chosen)
        torch.save(rejected_feature_map, cache_file_rejected)
        print(f"Feature maps saved to {cache_file_chosen} and {cache_file_rejected}")

    return chosen_feature_map, rejected_feature_map, sae_encoder


get_feature_map(
    model_name_or_path="google/gemma-2-2b-it",
    device="cuda",
    sae_encoder_name_or_path="google/gemma-scope-2b-pt-res",
    sae_layer_id=0,
    temperature=1.0,
    visualize=True,
    cache_dir=".cache",
    release=True,
)