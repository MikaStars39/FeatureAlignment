import torch

@torch.no_grad()
def intervene_state(
    model: torch.nn.Module,
    original_state: torch.Tensor,
    intervene_type: str,
    steering_method: str,
    position: int,
    current_position: str,
):
    """
    Intervene the state of the model.
    model: the model to intervene
    original_state: the original state of the model
    intervened_state: the intervened state of the model
    intervene_type: the type of intervention
    steering_method: the method to steer the intervention e.g. activation patching or addition
    position: the position to intervene (typically last few chat template tokens), 0 means all
    """
    if len(model.intervened) > 0 and current_position in intervene_type:
        if "attn_only" in intervene_type:
            if model.intervened_attn_output is None:
                model.intervened_attn_output = original_state
            elif not hasattr(model, 'read_attn_output'):
                intervened_state = steering(
                    steering_method, 
                    position, 
                    original_state, 
                    model.intervened_attn_output,
                    model
                )
                model.read_attn_output = 1 # skip the next time
                return intervened_state 
            return original_state
        elif "res_attn" in intervene_type:
            if model.intervened_res_attn_output is None:
                model.intervened_res_attn_output = original_state
            elif not hasattr(model, 'read_res_attn_output'):
                intervened_state = steering(
                    steering_method, 
                    position, 
                    original_state, 
                    model.intervened_res_attn_output,
                    model
                )
                model.read_res_attn_output = 1 # skip the next time
                return intervened_state
            return original_state
        elif "mlp_only" in intervene_type:
            if model.intervened_mlp_output is None:
                model.intervened_mlp_output = original_state
            elif not hasattr(model, 'read_mlp_output'):
                intervened_state = steering(
                    steering_method, 
                    position, 
                    original_state,
                    model.intervened_mlp_output,
                    model
                )
                model.read_mlp_output = 1 # skip the next time
                return intervened_state
            return original_state
        elif "res_mlp" in intervene_type:
            if model.intervened_res_mlp_output is None:
                model.intervened_res_mlp_output = original_state
            elif not hasattr(model, 'read_res_mlp_output'):
                intervened_state = steering(
                    steering_method, 
                    position, 
                    original_state, 
                    model.intervened_res_mlp_output,
                    model
                )
                model.read_res_mlp_output = 1 # skip the next time
                return intervened_state
            return original_state
        elif "cos" in intervene_type:
            return original_state
        else:
            raise ValueError(f"Intervene type {intervene_type} not supported")
    return original_state
        
def steering(
    steering_method: str,
    position: int,
    original_state: torch.Tensor,
    intervene_state: torch.Tensor,
    model: torch.nn.Module,
):
    if steering_method == "patching":
        intervened_state = intervene_state
        if position != 0:
            intervened_state[:, -position:, :] = original_state[:, -position:, :]
    elif steering_method == "addition":
        # check if the model has a addition coefficient
        if hasattr(model, 'addition_coefficient'):
            intervened_state = original_state - (original_state - intervene_state) \
                * model.addition_coefficient
        else:
            raise ValueError("Addition coefficient not found in model")
    else:
        raise ValueError(f"Steering method {steering_method} not supported")

    return intervened_state

def clean_model(model: torch.nn.Module):
    for i in range(len(model.model.layers)):
        model.model.layers[i].intervened = []
        model.model.layers[i].intervened_attn_output = None
        model.model.layers[i].intervened_mlp_output = None
        model.model.layers[i].intervened_res_mlp_output = None
        model.model.layers[i].intervened_res_attn_output = None
        model.model.layers[i].token_position = None
        model.model.layers[i].intervened_type = ""
        model.model.layers[i].steering_method = ""
        if hasattr(model.model.layers[i], 'addition_coefficient'):
            delattr(model.model.layers[i], 'addition_coefficient')
        # delete the self.read_attn_output
        if hasattr(model.model.layers[i], 'read_attn_output'):
            delattr(model.model.layers[i], 'read_attn_output')
        
        if hasattr(model.model.layers[i], 'read_mlp_output'):
            delattr(model.model.layers[i], 'read_mlp_output')
        
        if hasattr(model.model.layers[i], 'read_res_mlp_output'):
            delattr(model.model.layers[i], 'read_res_mlp_output')   
        
        if hasattr(model.model.layers[i], 'read_res_attn_output'):
            delattr(model.model.layers[i], 'read_res_attn_output')

    return model