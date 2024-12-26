import torch
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.cache_utils import Cache
from typing import Optional, Tuple
from transformers.utils import logging
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from feature_alignment.intervene.utils import intervene_state
logger = logging.get_logger(__name__)

# Adapted from LlamaAttention.forward
def llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return LlamaAttention.forward(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # use -1 to infer num_heads and num_key_value_heads as they may vary if tensor parallel is used
    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    # if self.intervened is not empty, then we need to intervene the attention output
    if hasattr(self, 'intervened'):
        # split the hidden states into the shape of attention output
        splited_hidden_states = hidden_states.view(bsz, q_len, self.num_heads, self.head_dim)
        for head_idx in self.intervened:
            attn_output[:, :, head_idx] = splited_hidden_states[:, :, head_idx]
        
        attn_output = attn_output.contiguous()

    attn_output = attn_output.view(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value

def llama_model_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        if "attn_only" in self.intervened_type and len(self.intervened) > 0:
            if self.intervened_attn_output is None:
                self.intervened_attn_output = hidden_states
            elif not hasattr(self, 'read_attn_output'):
                #  hidden_states[:, :, :1024*1] = self.intervened_attn_output[:, :, :1024*1]
                hidden_states = self.intervened_attn_output
                self.read_attn_output = 1 # skip the next time

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
    
        if "res_attn" in self.intervened_type and len(self.intervened) > 0:
            if self.intervened_res_attn_output is None:
                self.intervened_res_attn_output = residual
            elif not hasattr(self, 'read_res_attn_output'):
                residual[:, :, :] = self.intervened_res_attn_output[:, :, :]
                # residual = self.intervened_res_attn_output
                self.read_res_attn_output = 1 # skip the next time

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if "mlp_only" in self.intervened_type and len(self.intervened) > 0:
            if self.intervened_mlp_output is None:
                self.intervened_mlp_output = hidden_states
            elif not hasattr(self, 'read_mlp_output'):
                hidden_states = self.intervened_mlp_output
                self.read_mlp_output = 1 # skip the next time

        hidden_states = self.mlp(hidden_states)
        
        if "res_mlp" in self.intervened_type and len(self.intervened) > 0:
            if self.intervened_res_mlp_output is None:
                self.intervened_res_mlp_output = residual
            elif not hasattr(self, 'read_res_mlp_output'):
                residual = self.intervened_res_mlp_output
                self.read_res_mlp_output = 1 # skip the next time

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        
        return outputs


def intervene_llama_layer(model, pairs, token_position, type="attn"):
    """
    Intervene attention heads with Flash Attention support for Qwen2 models.
    
    Args:
        model: The Qwen2 transformer model
        pairs: List of (layer_idx, attention_head_idx) tuples
        status: "save" or "load"
    """
    # Replace forward methods of attention layers
    for i in range(len(model.model.layers)):
        model.model.layers[i].forward = llama_model_forward.__get__(
            model.model.layers[i]
        )
        model.model.layers[i].intervened = []
        model.model.layers[i].intervened_attn_output = None
        model.model.layers[i].intervened_res_attn_output = None
        model.model.layers[i].intervened_mlp_output = None
        model.model.layers[i].intervened_res_mlp_output = None
        model.model.layers[i].token_position = token_position
        model.model.layers[i].intervened_type = type
        for layer_idx, attention_head_idx in pairs:
            if layer_idx == i:
                model.model.layers[i].intervened.append(attention_head_idx)
        
    return model