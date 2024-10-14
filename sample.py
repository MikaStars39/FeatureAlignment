import json
import torch
import argparse
import math
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Generate AlpacaEval responses with a pretrained model")
    parser.add_argument('--model_name_or_path', type=str, default='google/gemma-2-2b', help='Path to the model or model name from Hugging Face hub')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--dataset_name', type=str, default='tatsu-lab/alpaca_eval', help='Name of the dataset from Hugging Face')
    parser.add_argument('--split', type=str, default='eval', help='Dataset split to use (e.g., train, eval)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for generation')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum length of the generated output')
    parser.add_argument('--output_file', type=str, default='alpaca_eval_results.json', help='File to save the generated results in JSON format')
    parser.add_argument('--max_batches', type=int, default=100, help='Maximum number of batches to process')
    parser.add_argument('--temperature', type=float, default=1, help='Maximum number of batches to process')
    parser.add_argument('--entropy', type=bool, default=False, help='Maximum number of batches to process')
    parser.add_argument('--fm', type=bool, default=False, help='Maximum number of batches to process')
    return parser.parse_args()

# Batch generate responses
def generate_responses(model, tokenizer, instructions, template, max_length, temperature):
    prompts = [template.format(instruction) for instruction in instructions]
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).input_ids.to('cuda')
    outputs = model.generate(inputs, max_new_tokens=max_length, pad_token_id=tokenizer.eos_token_id, temperature=temperature, do_sample=True)
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return responses

def get_entropy(model, tokenizer, instructions, template, max_length, temperature):
    prompts = instructions 
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).input_ids.to('cuda')
    logits = model(inputs, return_dict=True).logits.detach()
    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1).mean(dim=-1)
    del logits, probs, log_probs
    return entropy

def get_fm(model, tokenizer, instructions, template, max_length, temperature, sae_encoder):
    # prompts = [template.format(instruction) for instruction in instructions]
    inputs = tokenizer(instructions, return_tensors='pt', padding=True, truncation=True).input_ids.to('cuda')
    hidden_states = model(inputs, return_dict=True, output_hidden_states=True).hidden_states
    fm = sae_encoder.encode(hidden_states[-1])
    return fm

@torch.no_grad
def main():
    # Parse the arguments
    args = parse_args()
    

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    sft_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Load checkpoint
    model.load_state_dict(torch.load(args.checkpoint_path)['state'], strict=False)
    sft_model.load_state_dict(torch.load("cache/sft-gemma-2-2b/LATEST/policy.pt")['state'], strict=False)
    model = model.to('cuda')
    sft_model = sft_model.to('cuda')

    if args.fm:
        # load sae
        from feature_map import get_feature_map
        sae_encoder = get_feature_map(
            model_name_or_path="google/gemma-2-2b-it",
            sae_encoder_name_or_path="google/gemma-scope-2b-pt-res",
            sae_layer_id=25,
            temperature=1.0,
            visualize=True,
            cache_dir=".cache",
            release=True,
        )
        sae_encoder = sae_encoder.to('cuda')
        sae_encoder.eval().half()

    # Enable half precision (fp16) for faster inference
    model.half()
    sft_model.half()

    # Load dataset from Hugging Face hub
    if "jsonl" in args.dataset_name:
        dataset = load_dataset('json', data_files=args.dataset_name, split=args.split)
    else: dataset = load_dataset(args.dataset_name, split=args.split)

    # Define input template
    template = "<|user|>{}<|assistant|>"

    # Set up the DataLoader
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    # Generate results
    results = []
    entropys = 0
    fm = 0
    for i, batch in tqdm(enumerate(dataloader), total=args.max_batches // args.batch_size + 1):
        if i >= (args.max_batches // args.batch_size + 1):
            break
        if "arena" in args.dataset_name:
            instructions = batch['turns'][0]['content']
        elif "ultrafeedback" in args.dataset_name:
            instructions = batch['rejected'][0]['content']
            responeses = batch['rejected'][1]['content']
            instructions_all = []
            for instruction, response in zip(instructions, responeses):
                instructions_all.append(template.format(instruction) + response)
            instructions = instructions_all
        
        if args.entropy:
            # compute the logit entropy of the model
            entropy = get_entropy(model, tokenizer, instructions, template, args.max_length, args.temperature)
            entropys += entropy
        elif args.fm:
            fm_one = get_fm(model, tokenizer, instructions, template, args.max_length, args.temperature, sae_encoder)
            fm_sft = get_fm(sft_model, tokenizer, instructions, template, args.max_length, args.temperature, sae_encoder)
            
            # calculate mse loss
            fm += torch.nn.functional.mse_loss(fm_one, fm_sft)
        else:
            responses = generate_responses(model, tokenizer, instructions, template, args.max_length, args.temperature)
            for instruction, response in zip(instructions, responses):
                result = {
                    "instruction": instruction,
                    "output": response,
                    "generator": "gemma",
                    "dataset": "helpful_base",  # This can be customized or dynamic based on dataset
                    "datasplit": args.split
                }
                results.append(result)

    if args.entropy:
        print(f"Average entropy: {entropys / (args.max_batches // args.batch_size + 1)}")
    elif args.fm:
        print(fm)
        # # draw the feature map
        # import matplotlib.pyplot as plt
        # import numpy as np
        # # fm = fm / (args.max_batches // args.batch_size + 1)
        # fm = fm.mean(dim=0)

        # # flatten fm as a 2D array
        # N = math.ceil(math.sqrt(fm.shape[0]))
        # fm = fm[:N*N].reshape(N, N)

        # fm = fm.cpu().numpy()
        # fm = np.squeeze(fm)
        # plt.imshow(fm, cmap='Blues', interpolation='nearest')
        # plt.savefig("fm_dpo.pdf")

        
    else:
        # Save results to JSON file
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()