import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.utils import logging
from transformers.cache_utils import Cache
from typing import Optional, Tuple
from feature_alignment.intervene.module.utils import intervene_state
logger = logging.get_logger(__name__)

def mistral_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

    if query_states.device.type == "cuda" and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    is_causal = True if causal_mask is None and q_len > 1 else False
    
    sliding_window = getattr(self.config, "sliding_window", None)
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    
    # Intervention logic
    if hasattr(self, 'intervened'):
        splited_hidden_states = hidden_states.view(bsz, q_len, self.num_heads, self.head_dim)
        for head_idx in self.intervened:
            attn_output[:, :, head_idx] = splited_hidden_states[:, :, head_idx]
        attn_output = attn_output.contiguous()

    attn_output = attn_output.view(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value

def mistral_layer_forward(
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

        hidden_states = intervene_state(
            intervene_type=self.intervened_type,
            position=self.token_position,
            original_state=hidden_states,
            model=self,
            current_position="attn",
        )

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

        residual = intervene_state(
            intervene_type=self.intervened_type,
            position=self.token_position,
            original_state=residual,
            model=self,
            current_position="res_attn",
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        hidden_states = intervene_state(
            intervene_type=self.intervened_type,
            position=self.token_position,
            original_state=hidden_states,
            model=self,
            current_position="mlp",
        )

        hidden_states = self.mlp(hidden_states)

        residual = intervene_state(
            intervene_type=self.intervened_type,
            position=self.token_position,
            original_state=residual,
            model=self,
            current_position="res_mlp",
        )

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

def intervene_mistral_layer(
        model: torch.nn.Module, 
        pairs: list, 
        token_position: int, 
        intervened_type: str, 
        steering_method: str
    ):
    """
    Intervene attention heads with Flash Attention support for LLaMA and Mistral models.
    
    Args:
        model: The transformer model (LLaMA or Mistral)
        pairs: List of (layer_idx, attention_head_idx) tuples
        model_type: Either "llama" or "mistral"
    """
    for i in range(len(model.model.layers)):
        model.model.layers[i].forward = mistral_layer_forward.__get__(
            model.model.layers[i]
        )
        model.model.layers[i].intervened = []
        model.model.layers[i].intervened_attn_output = None
        model.model.layers[i].intervened_mlp_output = None
        model.model.layers[i].intervened_res_mlp_output = None
        model.model.layers[i].intervened_res_attn_output = None
        model.model.layers[i].intervened_type = intervened_type
        model.model.layers[i].token_position = token_position
        model.model.layers[i].steering_method = steering_method
        for layer_idx, attention_head_idx in pairs:
            if layer_idx == i:
                model.model.layers[i].intervened.append(attention_head_idx)

    return model
