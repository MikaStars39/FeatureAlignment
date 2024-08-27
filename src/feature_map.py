import torch
import tqdm
import os
from transformers import AutoTokenizer
from .utils import disable_dropout
from .sae import replace_sae_with_reture_feature_acts
from .metric import feature_vis

@torch.no_grad()
def get_feature_map(
    model_name_or_path: str,
    device: str,
    dataloader: int,
    sae_encoder_name_or_path: int,
    sae_layer_id: int,
    temperature: float = 1.0,
    visualize: bool = True,
    cache_dir: str = ".cache",
    release: bool = False,
):
    cache_file_chosen = os.path.join(cache_dir, f"{model_name_or_path}_layer_{sae_layer_id}_chosen_feature_map.pt")
    cache_file_rejected = os.path.join(cache_dir, f"{model_name_or_path}_layer_{sae_layer_id}_rejected_feature_map.pt")
    
    if os.path.exists(cache_file_chosen) and os.path.exists(cache_file_rejected):
        print(f"Loading cached feature maps from {cache_file_chosen} and {cache_file_rejected}")
        chosen_feature_map = torch.load(cache_file_chosen)
        rejected_feature_map = torch.load(cache_file_rejected)
    else:
        if "Gemma2" in model_name_or_path:
            from .transformers_model.modeling_gemma2 import Gemma2ForCausalLM
            model = Gemma2ForCausalLM.from_pretrained(
                model_name_or_path, 
                low_cpu_mem_usage=True, 
            )
        elif "Qwen1.5-0.5B" in model_name_or_path:
            from .transformers_model.modeling_qwen2 import Qwen2ForCausalLM
            model = Qwen2ForCausalLM.from_pretrained(
                model_name_or_path,
                low_cpu_mem_usage=True,
            )
        else:
            raise NotImplementedError(f"Model {model_name_or_path} not supported")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        disable_dropout(model)

        sae_encoder = replace_sae_with_reture_feature_acts()
        if release:
            sae_encoder, _, _ = sae_encoder.from_pretrained(
                release=sae_encoder_name_or_path, 
                sae_id=sae_layer_id, 
            )
        else:
            sae_encoder = sae_encoder.load_from_pretrained(
                path=sae_encoder_name_or_path,
            )
        model.model.layers[sae_layer_id].sae_encoder = sae_encoder

        model.to(device)
        model.eval()

        chosen_feature_map = None
        rejected_feature_map = None
        
        for batch in tqdm.tqdm(dataloader, desc="Getting Feature map"):
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

            _, chosen_feature_acts_reference = model(**chosen, use_cache=False)
            _, rejected_feature_acts_reference = model(**rejected, use_cache=False)

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

    print("Visualizing feature maps")
    feature_vis(
        chosen_feature_map, 
        cache_dir=cache_dir,
        name="chosen",
    )
    feature_vis(
        rejected_feature_map,
        cache_dir=cache_dir,
        name="rejected",
        )

    return chosen_feature_map, rejected_feature_map