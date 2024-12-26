import json
import fire
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from tqdm import tqdm
import torch

from feature_alignment.intervene.generation import generation

qwen_chat_template = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}"

def load_model_and_datasets(
        model_name: str, 
        dataset_name: str, 
    ):
    from huggingface_hub import login
    login(token="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")

    dataset = load_dataset("JailbreakBench/JBB-Behaviors", 'behaviors')
    dataset = dataset["harmful"][:128]["Goal"]

    set_seed(42)

    print("Beginning test the jailbreak")
    print("-" * 50)
    print(dataset[0])
    print("-" * 50)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    # model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda().to(torch.float16)

    return model, dataloader, tokenizer

def test_knowledge():
    model, dataloader, tokenizer = load_model_and_datasets(
        model_name="Qwen/Qwen2.5-7B-Instruct", 
        dataset_name="JailbreakBench/JBB-Behaviors",
    )

        # Load knowledge pairs from json
    with open("data/knowledge.json", "r") as f:
        knowledge_pairs = json.load(f)
    
    results = []

    for i in range(32):
        pairs = [
            (i, 0),
        ]
        clean_output = 0
        corrupted_output = 0

        # Process each knowledge pair
        clean_results = []
        corrupted_results = []

        total_count = 0
        for i, pair in tqdm(enumerate(knowledge_pairs)):
            clean_input = "Directly answer the question: " + pair["clean_question"] 
            corrupted_input = "Directly answer the question: " + pair["corrupted_question"]
            clean_answer = pair["clean_answer"]
            corrupted_answer = pair["corrupted_answer"]
            
            clean_result, corrupted_result = generation(
                tokenizer, 
                model, 
                clean_input, 
                corrupted_input, 
                pairs,
                clean_response=pair["clean_response"],
                corrupted_response=pair["corrupted_response"],
                generate_length=4,
            )

            if clean_result == "":
                continue
            else:
                total_count += 1

            # check if the clean answer in the clean result with lower case
            if clean_answer.lower() in clean_result.lower():
                clean_output += 1
            if corrupted_answer.lower() in corrupted_result.lower():
                corrupted_output += 1
            
            clean_results.append(clean_result)
            corrupted_results.append(corrupted_result)
            
        # save the clean result and corrupted result in jb_outputs.json
        with open("outputs/jb_outputs.json", "w") as f:
            json.dump(clean_results + corrupted_results, f, indent=4, sort_keys=True)

        print(f"Clean output: {clean_output / total_count}, \
              Corrupted output: {corrupted_output / total_count}")
        
        results.append({
            "clean_output": clean_output / total_count,
            "corrupted_output": corrupted_output / total_count,
        })

    for i, result in enumerate(results):
        print(f"Result {i}:")
        print(result)
        print("-" * 50)

def test_steering(
        model: torch.nn.Module,
        tokenizer,
        knowledge_pairs: list[dict],
        model_name_or_path: str,
        generate_length: int = 4,
        intervene_type: str = "attn_only",
        token_position: int = 0,
        steering_type: str = "patching",
    ):

    outputs = []
    for i in range(32):
        pairs = [
            (i, 0),
        ]
        clean_output = []
        corrupted_output = []
        
        # Process each knowledge pair
        for i, pair in tqdm(enumerate(knowledge_pairs)):
            clean_input = pair["clean_question"]
            corrupted_input = pair["corrupted_question"]

            clean_input = qwen_chat_template.format(clean_input, "")
            corrupted_input = qwen_chat_template.format(corrupted_input, "")
            
            clean_result, corrupted_result = generation(
                tokenizer, 
                model, 
                model_name_or_path,
                clean_input, 
                corrupted_input, 
                pairs,
                generate_length=generate_length,
                intervene_type=intervene_type,
                token_position=token_position,
                steering_type=steering_type,
            )

            clean_output.append(clean_result)
            corrupted_output.append(corrupted_result)

        outputs.append({
            "clean_output": clean_output,
            "corrupted_output": corrupted_output,
        })

    return outputs

def main(
    model_name_or_path: str,
    dataset_name_or_path: str,
    json_path: str,
    intervene_type: str,
    steering_type: str,
    token_position: int,
    generate_length: int,
    task_name: str,
):
    # load the model and dataset
    model, dataloader, tokenizer = load_model_and_datasets(
        model_name_or_path,
        dataset_name_or_path,
    )

    # test pairs
    with open(json_path, "r") as f:
        test_pairs = json.load(f) 

    result = test_steering(
        model, 
        tokenizer, 
        test_pairs, 
        model_name_or_path, 
        generate_length,
        intervene_type, 
        steering_type, 
        token_position, 
    )

    if task_name == "safety":
        from feature_alignment.benchmark.test_safety import test_safety
        test_safety(result)
    


# llama
# (26, 15),
# (26, 13),
# (27, 5),
# (24, 27),
# (20, 1),
# (15, 30),
# (16, 20),
# (20, 14)

# mistral
# (7, 12),
# (29, 27),
# (29, 25),
# (30, 22),
# (30, 28),
# (27, 15),

if __name__ == "__main__":
    # use fire
    fire.Fire(main)