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

def test_early_exit():
    pass

def test_steering(
        model: torch.nn.Module,
        tokenizer,
        knowledge_pairs: list[dict],
        model_name_or_path: str,
        generate_length: int = 4,
        intervene_type: str = "attn_only",
        token_position: int = 0,
        steering_type: str = "patching",
        layer_num: int = 32,
        addition_coefficient: float = 0.1
    ):
    """ 
        test the steering
        model: the model to test
        tokenizer: the tokenizer of the model
        knowledge_pairs: the knowledge pairs to test
        model_name_or_path: the name or path of the model
        generate_length: the length of the generated text
        intervene_type: the type of the intervention
        token_position: the position of the token to intervene
        steering_type: the type of the steering
        layer_num: the number of layers to intervene
        if_output_cache: whether to output the cache
        if_load_cache: whether to load the cache
        cache_path: the path to save the cache
    """

    outputs = []
    for i in range(0, layer_num):
        pairs = [
            (i, 0),
        ]
        output = []

        # Process each knowledge pair
        for i, pair in tqdm(enumerate(knowledge_pairs)):
            clean_input = pair["clean_question"]
            corrupted_input = pair["corrupted_question"]

            # if pair has "clean_response" and "corrupted_response"
            if "clean_response" in pair and "corrupted_response" in pair:
                clean_response = pair["clean_response"]
                corrupted_response = pair["corrupted_response"]
            else:
                clean_response = ""
                corrupted_response = "" 

            clean_input = qwen_chat_template.format(clean_input, clean_response)
            corrupted_input = qwen_chat_template.format(corrupted_input, corrupted_response)
            
            clean_result, corrupted_result, output_cache = generation(
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
                addition_coefficient=addition_coefficient
            )

            output.append({
                "clean_output": clean_result,
                "corrupted_output": corrupted_result,
                **pair
            })

        outputs.append(output)
    return outputs

def main(
    model_name_or_path: str,
    dataset_name_or_path: str,
    json_path: str,
    intervene_type: str,
    steering_type: str,
    token_position: int,
    generate_length: int,
    layer_num: int,
    task_name: str,
    addition_coefficient: float = 0.1
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
        token_position, 
        steering_type, 
        layer_num,
        addition_coefficient
    )

    if task_name == "safety":
        from feature_alignment.benchmark.test_safety import test_safety
        # json path = model_name_or_path + "_" + intervene_type + "_" + steering_type + "_" + token_position + ".json"
        json_path = f"outputs/{model_name_or_path[:4]}_{intervene_type}_{steering_type}_{token_position}.json"
        test_safety(result, json_path)
    elif task_name == "knowledge":
        from feature_alignment.benchmark.test_knowledge import test_knowledge
        test_knowledge(result, json_path)


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