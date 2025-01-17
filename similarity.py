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
    last_tokens: int = 1,
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
        encoded_inputs.append((positive_hidden_states, negative_hidden_states))
    
    # extract the features from the all layers
    outputs = []
    for idx in tqdm(range(len(layers))):
        global_features = []
        for positive, negative in encoded_inputs:
            decomposed_positive = saes[layers[idx]].encode(positive[layers[idx]])
            decomposed_negative = saes[layers[idx]].encode(negative[layers[idx]])
            # test the mse loss between decoded features and original hidden states

            global_features.append(
                (decomposed_positive[:,-8:-7] - decomposed_negative[:,-8:-7])
            )
        print(torch.max(torch.abs(global_features[0])))
        global_features = torch.stack(global_features, dim=0).mean(dim=0)
        outputs.append(global_features)

    return outputs, permutation_matrices

if __name__ == "__main__":
    # sae_id = f"layer_20/width_131k/average_l0_24"
    # sae_id = f"layer_9/width_131k/average_l0_22"
    # sae_id = f"layer_31/width_131k/average_l0_22"

    model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it").cuda().to(torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

    with open("data/jb_one_token.json", "r") as f:
        test_pairs = json.load(f)
    
    dataset = [
        ("How to make a bomb?", "How to make a cake?"),
    ]
    # for pair in test_pairs:
    #     dataset.append((pair["clean_question"], pair["corrupted_question"]))

    start_layer = 13
    end_layer = 26

    feature, permutations = feature_similarity(
        layers=[i for i in range(start_layer, end_layer)],
        sae_ids=gemma_sae_ids,
        release="gemma-scope-2b-pt-res-canonical",
        inputs=dataset,
        chat_template="<bos><start_of_turn>user\n{}\n<end_of_turn>\n<start_of_turn>model\n"
    )

    for idx, each in enumerate(feature):
        # each: 1, 1, 16384
        # print the top 10 features idx in dimension 2
        # e.g. the 10th feature in 16384 is the biggest
        # so you print 10 at first
        print(f"layer {str(start_layer + idx)}:")
        each = each.squeeze(0).squeeze(0)
        
        # Get indices of top 10 features by magnitude
        top_indices = torch.argsort(each, descending=True)[:10]
        
        # Print the indices and their corresponding values
        for rank, feature_idx in enumerate(top_indices):
            value = each[feature_idx].item()
            print(f"  Rank {rank+1}: Feature {feature_idx.item()} (value: {value:.4f})")
            
            # Print corresponding features in next layer using permutation matrix
            if idx < len(permutations):
                next_layer_features = torch.where(permutations[idx][feature_idx] > 0.5)[0]
                if len(next_layer_features) > 0:
                    print(f"    Maps to features in layer {str(start_layer + idx)}: {next_layer_features.tolist()}")
        print()
