import torch

from .module.utils import clean_model

def get_intervene_function(model_name: str):
    if "Qwen" in model_name:
        from .module.qwen_attention_head import intervene_qwen_layer
        return intervene_qwen_layer
    elif "Llama" in model_name:
        from .module.llama_attention_head import intervene_llama_layer
        return intervene_llama_layer
    elif "Mistral" in model_name:
        from .module.mistral_attention_head import intervene_mistral_attention_head
        return intervene_mistral_attention_head
    else:
        raise ValueError(f"Model {model_name} not supported")

def generation(
        tokenizer, 
        model, 
        model_name_or_path: str,
        clean_prompt: str, 
        corrupted_prompt: str, 
        pairs: list[tuple[int, int]],
        generate_length: int,
        intervene_type: str,
        token_position: int,
        steering_type: str,
        if_output_cache: bool = False,
        load_cache: dict = None,
        addition_coefficient: float = 0.1
    ):
    # Process clean input
    clean_inputs = tokenizer(clean_prompt, return_tensors="pt", padding=True, truncation=False).to('cuda')
    clean_inputs.input_ids = clean_inputs.input_ids.to(torch.float16)
    clean_inputs.attention_mask = clean_inputs.attention_mask.to(torch.float16)

    # Process corrupted input  
    corrupted_inputs = tokenizer(corrupted_prompt, return_tensors="pt", padding=True, truncation=False).to('cuda')
    corrupted_inputs.input_ids = corrupted_inputs.input_ids.to(torch.float16)
    corrupted_inputs.attention_mask = corrupted_inputs.attention_mask.to(torch.float16)

    # Check if token lengths match
    if len(clean_inputs.input_ids[0]) != len(corrupted_inputs.input_ids[0]):
        print(f"Skipping pair due to length mismatch: {len(clean_inputs.input_ids[0])} vs {len(corrupted_inputs.input_ids[0])}")
        return "", "", None

    # clean the model
    model = clean_model(model)
    generate_length = int(generate_length)

    # Generate with clean input
    clean_outputs = model.generate(
        **clean_inputs,
        max_new_tokens=generate_length,
        do_sample=True,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    clean_result = tokenizer.decode(clean_outputs[0][-generate_length:], skip_special_tokens=True)

    INTERVENE_FUNCTION = get_intervene_function(model_name_or_path)
    model = INTERVENE_FUNCTION(
        model, 
        pairs, 
        token_position, 
        intervene_type, 
        steering_type,
        addition_coefficient=addition_coefficient
    )

    # Generate with corrupted input
    corrupted_outputs = model.generate(
        **corrupted_inputs, 
        max_new_tokens=1,
        do_sample=True,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    corrupted_outputs = model.generate(
        **clean_inputs, 
        max_new_tokens=generate_length,
        do_sample=True,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # # get the last layer's refusal vector
    # last_layer_refusal_vector = load_cache[-1]["mlp"][0, -5:, :]
    
    # # get the refusal vector of each layer
    # for i in range(len(model.model.layers)):
    #     if hasattr(model.model.layers[i], "refusal_vector_mlp"):
    #         mlp_refusal_vector = model.model.layers[i].intervened_res_mlp_output
    #         # cosine similarity between refusal_vector and last_layer_refusal_vector
    #         mlp_refusal_vector = mlp_refusal_vector[0, -5:, :]
            
    #         cos_similarity = torch.cosine_similarity(mlp_refusal_vector, last_layer_refusal_vector, dim=1).mean()

    #         # attn
    #         attn_refusal_vector = model.model.layers[i].intervened_res_attn_output
    #         attn_refusal_vector = attn_refusal_vector[0, -5:, :]
    #         attn_cos_similarity = torch.cosine_similarity(attn_refusal_vector, last_layer_refusal_vector, dim=1).mean()
    #         print(f"Layer {i} mlp cosine similarity: {cos_similarity}")
    #         print(f"Layer {i} attn cosine similarity: {attn_cos_similarity}")

    corrupted_result = tokenizer.decode(corrupted_outputs[0][-generate_length:], skip_special_tokens=True)
    # exit()

    # geth all the layer's cache
    if if_output_cache:
        output_cache = []
        for i in range(len(model.model.layers)):
            if hasattr(model.model.layers[i], "intervened_res_attn_output"):
                output_cache.append({"attn": model.model.layers[i].refusal_vector_attn,
                                      "mlp": model.model.layers[i].refusal_vector_mlp})
    else: output_cache = None

    return clean_result, corrupted_result, output_cache
