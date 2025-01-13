import torch
from transformers.utils import logging
from transformers.cache_utils import Cache
from typing import Optional, Tuple, Callable
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
logger = logging.get_logger(__name__)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, eager_attention_forward
from .utils import intervene_state

def qwen_model_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:


        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,  # main diff with Llama
            **kwargs,
        )
        
        # Skip intervention during generation (when using kv cache)
        if attn_output.shape[1] > 1 and len(self.intervened) > 0:
            if self.intervened_attn_output is None:
                self.intervened_attn_output = attn_output
            else:
                # split into 28 attention heads
                for head_idx in self.intervened:
                    attn_output = \
                        self.intervened_attn_output
        else:
            pass

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

def qwen_layer_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        if "early_exit" in self.intervened_type and len(self.intervened) > 0:
            return hidden_states
        
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = intervene_state(
            self, 
            hidden_states, 
            self.intervened_type, 
            self.steering_method, 
            self.token_position,
            "attn_only"
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
            self, 
            residual, 
            self.intervened_type, 
            self.steering_method, 
            self.token_position,
            "res_attn"
        )

        hidden_states = residual + hidden_states

        if "cos" in self.intervened_type and len(self.intervened) > 0:
            if self.intervened_res_attn_output is None:
                self.intervened_res_attn_output = hidden_states
            else:
                # get the 27th layer's refusal vector 
                self.refusal_vector_attn = hidden_states - self.intervened_res_attn_output

        # Fully Connected
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = intervene_state(
            self, 
            hidden_states, 
            self.intervened_type, 
            self.steering_method, 
            self.token_position,
            "mlp_only"
        )

        hidden_states = self.mlp(hidden_states)

        residual = intervene_state(
            self, 
            residual, 
            self.intervened_type, 
            self.steering_method, 
            self.token_position,
            "res_mlp"
        )

        hidden_states = residual + hidden_states

        if "cos" in self.intervened_type and len(self.intervened) > 0:
            if self.intervened_res_mlp_output is None:
                self.intervened_res_mlp_output = hidden_states
            else:
                # get the 27th layer's refusal vector 
                self.refusal_vector_mlp = hidden_states - self.intervened_res_mlp_output

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

def intervene_qwen_layer(
        model: torch.nn.Module, 
        pairs: list, 
        token_position: int, 
        intervened_type: str, 
        steering_method: str,
        addition_coefficient: float = 0.1
    ):
    for i in range(len(model.model.layers)):
        model.model.layers[i].forward = qwen_layer_forward.__get__(
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
        model.model.layers[i].addition_coefficient = addition_coefficient
        for layer_idx, attention_head_idx in pairs:
            if layer_idx == i:
                model.model.layers[i].intervened.append(attention_head_idx)

    return model

