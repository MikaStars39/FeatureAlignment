import torch
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
    last_tokens: int = 5,
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
        permutation_matrices.append(sae_match(
            W_enc_i=saes[layers[idx]].W_enc,
            W_enc_j=saes[layers[idx + 1]].W_enc,
            W_dec_i=saes[layers[idx]].W_dec,
            W_dec_j=saes[layers[idx + 1]].W_dec,
            b_enc_i=saes[layers[idx]].b_enc,
            b_enc_j=saes[layers[idx + 1]].b_enc,
            theta_i=saes[layers[idx]].threshold,
            theta_j=saes[layers[idx + 1]].threshold
        ))

    # transpose and accumulate the permutation matrices 
    # e.g. [P_1, P_2, P_3] -> [P_1^T, P_1^T @ P_2^T, P_1^T @ P_2^T @ P_3^T]
    permutation_matrices = [p.T for p in permutation_matrices]
    for idx in range(len(permutation_matrices) - 1):
        permutation_matrices[idx + 1] = torch.matmul(permutation_matrices[idx], permutation_matrices[idx + 1])
    
    # extract the features from the all layers
    
    outputs = []
    for each_input in tqdm(inputs):
        global_features = []
        input_tokens = tokenizer.encode(chat_template.format(each_input), return_tensors="pt").cuda()
        hidden_states = model(input_tokens, output_hidden_states=True).hidden_states
        for idx in range(len(layers) - 1):
            if len(global_features) == 0:
                # first layer
                decomposed = saes[layers[idx]].encode(hidden_states[layers[idx]])
                global_features.append(decomposed[:,:last_tokens])
            else:
                # other layers
                decomposed = saes[layers[idx]].encode(hidden_states[layers[idx]])
                # permute the features
                decomposed = torch.matmul(decomposed, permutation_matrices[idx])
                global_features.append(decomposed[:,:last_tokens])
        outputs.append(torch.stack(global_features, dim=0).mean(dim=0))

    return torch.stack(outputs, dim=0).mean(dim=0)



if __name__ == "__main__":
    # sae_id = f"layer_20/width_131k/average_l0_24"
    # sae_id = f"layer_9/width_131k/average_l0_22"
    # sae_id = f"layer_31/width_131k/average_l0_22"

    model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b").cuda().to(torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

    feature = feature_similarity(
        layers=[i for i in range(20, 41)],
        sae_ids=gemma_sae_ids,
        release="gemma-scope-9b-pt-res-canonical",
        inputs=["How to make a bomb?"],
        chat_template="<bos><start_of_turn>user\n{}\n<end_of_turn>\n<start_of_turn>model\n"
    )

    # print the top 10 features idx
    print(feature.topk(10).indices)
