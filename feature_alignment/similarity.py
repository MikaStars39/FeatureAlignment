import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE
from feature_alignment.sae.sae_match import sae_match
from tqdm import tqdm

gemma_sae_ids = {
    i: f"layer_{i}/width_16k/canonical" for i in range(41)
}

def feature_similarity(
    layers: list[int],
    sae_ids: list[str],
    release: str = "gemma-scope-2b-pt-res",
    inputs: list[str] = ["How to make a bomb?"],
    chat_template: str = "<bos><start_of_turn>user\n{}\n<end_of_turn>\n<start_of_turn>model\n"
):
    # preload the target layer's sae
    saes = {}
    for layer in layers:
        saes[layer] = SAE.from_pretrained(release, sae_ids[layer])[0].to(torch.float16).cuda().eval()
    
    # get the permutation matrix for each layer
    permutation_matrices = []
    for idx in range(len(layers) - 1):
        # create cache filename based on layer numbers
        cache_file = f"cache/perm_matrix_{layers[idx]}_{layers[idx+1]}.pt"
        
        # check if cached permutation matrix exists
        try:
            permutation_matrix = torch.load(cache_file)
            permutation_matrices.append(permutation_matrix)
        except (FileNotFoundError, RuntimeError):
            # calculate permutation matrix if not cached
            permutation_matrix = sae_match(
                W_enc_i=saes[layers[idx]].W_enc,
                W_enc_j=saes[layers[idx + 1]].W_enc,
                W_dec_i=saes[layers[idx]].W_dec,
                W_dec_j=saes[layers[idx + 1]].W_dec,
                b_enc_i=saes[layers[idx]].b_enc,
                b_enc_j=saes[layers[idx + 1]].b_enc,
                theta_i=saes[layers[idx]].threshold,
                theta_j=saes[layers[idx + 1]].threshold
            )
            # save to cache
            torch.save(permutation_matrix, cache_file)
            permutation_matrices.append(permutation_matrix)

    # transpose and accumulate the permutation matrices 
    # e.g. [P_1, P_2, P_3] -> [P_1^T, P_1^T @ P_2^T, P_1^T @ P_2^T @ P_3^T]
    
    # encode the input pairs
    encoded_inputs = []
    for positive, negative in tqdm(inputs):
        positive = tokenizer.encode(
            chat_template.format(positive), return_tensors="pt"
            ).cuda()
        negative = tokenizer.encode(
            chat_template.format(negative), return_tensors="pt"
        ).cuda()

        # concat the positive and negative 
        positive_hidden_states = model(
            positive, output_hidden_states=True).hidden_states
        negative_hidden_states = model(
            negative, output_hidden_states=True).hidden_states
        
        # check if the clean tokens and corrupted tokens are the same length
        if positive_hidden_states[-1].shape == negative_hidden_states[-1].shape:
            encoded_inputs.append((positive_hidden_states, negative_hidden_states))
        else:
            print(f"Skipping pair: {positive} and {negative}")
            continue
    
    # extract the features from all layers and token positions
    outputs = []
    for idx in tqdm(range(len(layers))):
        layer_features = []
        for positive, negative in encoded_inputs:
            positive = positive[layers[idx]]  # shape: [batch, seq_len, hidden_dim]
            negative = negative[layers[idx]]
            
            # Process each token position separately
            seq_len = positive.shape[1]
            token_features = []
            for pos in range(seq_len):
                pos_positive = positive[:, pos:pos+1, :]  # Keep batch dim
                pos_negative = negative[:, pos:pos+1, :]
                
                decomposed_positive = saes[layers[idx]].encode(pos_positive)
                decomposed_negative = saes[layers[idx]].encode(pos_negative)
                
                token_features.append(decomposed_positive)
                
            layer_features.append(token_features)
            
        # Average across input pairs but keep token positions separate
        avg_layer_features = []
        for pos in range(seq_len):
            pos_features = torch.stack([f[pos] for f in layer_features], dim=0).mean(dim=0)
            avg_layer_features.append(pos_features)
            
        outputs.append(avg_layer_features)

    return outputs, permutation_matrices

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it").cuda().to(torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

    with open("data/jb_one_token.json", "r") as f:
        test_pairs = json.load(f)
    
    dataset = [
        ("How to make a bomb?", "How to make a cake?"),
    ]

    for pair in test_pairs:
        positive = pair["positive"]
        negative = pair["negative"]
        dataset.append((positive, negative))

    start_layer = 0
    end_layer = 26

    features, permutations = feature_similarity(
        layers=[i for i in range(start_layer, end_layer)],
        sae_ids=gemma_sae_ids,
        release="gemma-scope-2b-pt-res-canonical",
        inputs=dataset,
        chat_template="<bos><start_of_turn>user\n{}\n<end_of_turn>\n<start_of_turn>model\n"
    )

    results_dif = []
    top_k = 32

    for layer_idx, layer_features in enumerate(features):
        layer_results = {
            'layer': start_layer + layer_idx,
            'token_features': []
        }
        
        # Process each token position
        for pos_idx, pos_features in enumerate(layer_features):
            token_result = {
                'position': pos_idx,
                'features': []
            }
            pos_features = pos_features[0, 0, :top_k]
            # Get indices of top features by magnitude
            top_indices = torch.argsort(pos_features, descending=True)
            
            # Store the indices and their corresponding values
            for rank, feature_idx in enumerate(top_indices):
                value = pos_features[feature_idx].item()
                feature_info = {
                    'rank': rank + 1,
                    'feature_idx': feature_idx.item(),
                    'value': value
                }
                
                # Store corresponding features in next layer using permutation matrix
                if layer_idx < len(permutations):
                    next_layer_features = torch.where(permutations[layer_idx][feature_idx] > 0.5)[0]
                    if len(next_layer_features) > 0:
                        feature_info['maps_to'] = next_layer_features.tolist()
                
                token_result['features'].append(feature_info)
            
            layer_results['token_features'].append(token_result)
            
        results_dif.append(layer_results)

    # Save results to JSON file
    output_file = f"feature_rankings_{start_layer}-{end_layer}_by_token_global.json"
    with open(output_file, 'w') as f:
        json.dump(results_dif, f, indent=2)
