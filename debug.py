import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from sae_lens import SAE
from tqdm import tqdm

def test_reconstruction_loss(
    layers: list[int],
    sae_ids: dict,
    release: str = "gemma-scope-2b-pt-res-canonical",
    inputs: list[str] = ["How to make a bomb?"],
    chat_template: str = "<bos><start_of_turn>user\n{}\n<end_of_turn>\n<start_of_turn>model\n"
):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b").cuda().to(torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    
    # Preload SAEs for target layers
    saes = {}
    for layer in layers:
        saes[layer] = SAE.from_pretrained(release, sae_ids[layer])[0].to(torch.float16).cuda().eval()
    
    # Process inputs and get hidden states
    reconstruction_losses = {}
    for layer in tqdm(layers, desc="Processing layers"):
        layer_losses = []
        
        for input_text in tqdm(inputs, desc=f"Processing inputs for layer {layer}"):
            # Tokenize and get model hidden states
            input_ids = tokenizer.encode(
                chat_template.format(input_text), 
                return_tensors="pt"
            ).cuda()
            
            with torch.no_grad():
                hidden_states = model(input_ids, output_hidden_states=True).hidden_states
                layer_hidden = hidden_states[layer]
                
                # Encode and decode through SAE
                encoded_features = saes[layer].encode(layer_hidden)
                reconstructed_hidden = saes[layer].decode(encoded_features)
                
                # Calculate MSE loss
                mse_loss = torch.nn.functional.mse_loss(layer_hidden, reconstructed_hidden)
                layer_losses.append(mse_loss.item())
        
        # Store average loss for this layer
        reconstruction_losses[layer] = sum(layer_losses) / len(layer_losses)
    
    return reconstruction_losses

if __name__ == "__main__":
    # Define layers to test
    layers = list(range(0, 26))  # Test all layers
    
    # Define SAE IDs mapping
    gemma_sae_ids = {
        i: f"layer_{i}/width_16k/canonical" for i in range(41)
    }
    
    # Load test data
    with open("data/jb_one_token.json", "r") as f:
        test_pairs = json.load(f)
    
    # Prepare input dataset
    dataset = []
    for pair in test_pairs:
        dataset.extend([pair["clean_question"], pair["corrupted_question"]])
    
    # Test reconstruction loss
    losses = test_reconstruction_loss(
        layers=layers,
        sae_ids=gemma_sae_ids,
        release="gemma-scope-2b-pt-res-canonical",
        inputs=dataset,
        chat_template="<bos><start_of_turn>user\n{}\n<end_of_turn>\n<start_of_turn>model\n"
    )
    
    # Print results
    print("\nReconstruction Losses by Layer:")
    print("-" * 40)
    for layer, loss in losses.items():
        print(f"Layer {layer:2d}: {loss:.6f}")
