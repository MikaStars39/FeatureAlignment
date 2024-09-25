import json
import torch
import argparse
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
    return parser.parse_args()

# Batch generate responses
def generate_responses(model, tokenizer, instructions, template, max_length, temperature):
    prompts = [template.format(instruction) for instruction in instructions]
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).input_ids.to('cuda')
    outputs = model.generate(inputs, max_new_tokens=max_length, pad_token_id=tokenizer.eos_token_id, temperature=temperature, do_sample=True)
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return responses

def main():
    # Parse the arguments
    args = parse_args()

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Load checkpoint
    model.load_state_dict(torch.load(args.checkpoint_path)['state'], strict=False)
    model = model.to('cuda')

    # Enable half precision (fp16) for faster inference
    model.half()

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
    for i, batch in tqdm(enumerate(dataloader), total=args.max_batches // args.batch_size + 1):
        if i >= (args.max_batches // args.batch_size + 1):
            break
        instructions = batch['turns'][0]['content']
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

    # Save results to JSON file
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()