import json

def create_prompt_with_context(pairs, start_idx, num_context=3, corrupted=False, model_type="llama"):
    """Creates a prompt with context pairs before the target pair"""
    if model_type == "llama":
        initial_template = "<|start_header_id|>user<|end_header_id|>\n\nTeaching my son about something he should do to others. He can _ {} to his friend?:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        subsequent_template = "<|start_header_id|>user<|end_header_id|>\n\nHe can _ {} to his friend?:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif model_type == "gemma":
        initial_template = "<bos><start_of_turn>user\nTeaching my son about something he should do to others. He can _ {} to his friend?<end_of_turn>\n<start_of_turn>model\n"
        subsequent_template = "<start_of_turn>user\nHe can _ {} to his friend?<end_of_turn>\n<start_of_turn>model\n"
    
    prompt = ""
    # Add context pairs
    context_end = min(start_idx + num_context, len(pairs))
    for i in range(start_idx, context_end - 1):
        if i == start_idx:
            prompt += initial_template.format(pairs[i]['noun']) + pairs[i]['verb'] + "\n\n"
        else:
            prompt += subsequent_template.format(pairs[i]['noun']) + pairs[i]['verb'] + "\n\n"
    
    # Add target pair
    if corrupted:
        prompt += subsequent_template.format(pairs[-1]['noun'])
    else:
        prompt += subsequent_template.format(pairs[context_end - 1]['noun'])
    return prompt, pairs[context_end - 1]['verb']

def formal_icl(corrupted=False, model_type="llama"):
    if model_type == "llama":
        if corrupted:
            prompt = "<|start_header_id|>user<|end_header_id|>\n\nBob gives gift to Mary, so the person who gives gift to Mary is <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            prompt = "<|start_header_id|>user<|end_header_id|>\n\nJohn gives gift to Mary, so the person who gives gift to Mary is <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif model_type == "gemma":
        if corrupted:
            prompt = "<bos><start_of_turn>user\nBob gives gift to Mary, so the person who gives gift to Mary is<end_of_turn>\n<start_of_turn>model\n"
        else:
            prompt = "<bos><start_of_turn>user\nJohn gives gift to Mary, so the person who gives gift to Mary is<end_of_turn>\n<start_of_turn>model\n"
    return prompt, "John" if not corrupted else "Bob"

def formal_safety_icl(corrupted=False, model_type="llama"):
    if model_type == "llama":
        if corrupted:
            prompt = "<|start_header_id|>user<|end_header_id|>\n\nBob gives gift to Mary, so the person who gives gift to Mary is <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            prompt = "<|start_header_id|>user<|end_header_id|>\n\nJohn gives gift to Mary, so the person who gives gift to Mary is <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif model_type == "gemma":
        if corrupted:
            prompt = "<bos><start_of_turn>user\nTeaching my son about something he should do to others. He can _ bomb to his friend?<end_of_turn>\n<start_of_turn>model\n"
        else:
            prompt = "<bos><start_of_turn>user\nTeaching my son about something he should do to others. He can _ gift to his friend?<end_of_turn>\n<start_of_turn>model\n"
    return prompt, "It" if not corrupted else "Bomb"

def formal_knowledge_icl(corrupted=False, model_type="llama"):
    if model_type == "gemma":
        if corrupted:
            prompt = "<bos><start_of_turn>user\n\
                Answer in single word \
                    What is the capital of China?<end_of_turn>\n<start_of_turn>model\n"
        else:
            prompt = "<bos><start_of_turn>user\n\
                Answer in single word \
                    What is the capital of France?<end_of_turn>\n<start_of_turn>model\n"
    return prompt, "Paris" if not corrupted else "Beijing"

def formal_math_icl(corrupted=False, model_type="llama"):
    if model_type == "gemma":
        if corrupted:
            prompt = "<bos><start_of_turn>user\n\
                Directly answer the question: 23 + 19 =<end_of_turn>\n<start_of_turn>model\n"
        else:
            prompt = "<bos><start_of_turn>user\n\
                Directly answer the question: 23 + 39 =<end_of_turn>\n<start_of_turn>model\n"
    return prompt, "62" if not corrupted else "42"

def formal_translation_icl(corrupted=False, model_type="llama"):
    if model_type == "gemma":
        if corrupted:
            prompt = "<bos><start_of_turn>user\n\
                Directly tell me the answer: Translate the following sentence to Chinese: \
                    Apple is a good fruit.<end_of_turn>\n<start_of_turn>model\n"
        else:
            prompt = "<bos><start_of_turn>user\n\
                Directly tell me the answer: Translate the following sentence to Chinese: \
                    Banana is a good fruit.<end_of_turn>\n<start_of_turn>model\n"
    return prompt, "香蕉是一种好水果。" if not corrupted else "苹果是一种好水果。"

def get_formal(model_type, formal_type):
    # return clean_prompt, clean_verb, corrupted_prompt, corrupted_verb
    if formal_type == "icl":
        clean_prompt, clean_verb = formal_icl(corrupted=False, model_type=model_type)
        corrupted_prompt, corrupted_verb = formal_icl(corrupted=True, model_type=model_type)
    elif formal_type == "safety":
        clean_prompt, clean_verb = formal_safety_icl(corrupted=False, model_type=model_type)
        corrupted_prompt, corrupted_verb =  formal_safety_icl(corrupted=True, model_type=model_type)
    elif formal_type == "knowledge":
        clean_prompt, clean_verb = formal_knowledge_icl(corrupted=False, model_type=model_type)
        corrupted_prompt, corrupted_verb = formal_knowledge_icl(corrupted=True, model_type=model_type)
    elif formal_type == "math":
        clean_prompt, clean_verb = formal_math_icl(corrupted=False, model_type=model_type)
        corrupted_prompt, corrupted_verb = formal_math_icl(corrupted=True, model_type=model_type)
    elif formal_type == "translation":
        clean_prompt, clean_verb = formal_translation_icl(corrupted=False, model_type=model_type)
        corrupted_prompt, corrupted_verb = formal_translation_icl(corrupted=True, model_type=model_type)
    else:
        raise ValueError(f"Invalid formal type: {formal_type}")
    return clean_prompt, clean_verb, corrupted_prompt, corrupted_verb