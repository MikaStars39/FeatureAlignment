import os
import torch
import transformers
import lightning as L
import torch.nn.functional as F
import numpy as np
import deepspeed

from torch import optim, nn, utils, Tensor
from transformers import AutoModelForCausalLM
from omegaconf import DictConfig
from lightning.pytorch.utilities import rank_zero_info, rank_zero_only
from .utils import disable_dropout
from .sae import replace_sae_with_reture_feature_acts
from .jump_relu_sae import JumpReLUSAE
from huggingface_hub import hf_hub_download
from typing import (
    Any, 
    Callable, 
    Literal, 
    Optional, 
    Tuple, 
    TypeVar, 
    Union, 
    overload, 
    Dict, 
    List
)

from .transformers_model.modeling_gemma import GemmaForCausalLM

# define the LightningModule
class FeatureLevelDPOModel(L.LightningModule):
    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__()
        self.config = config
        self.load_model()

    
    def load_model(
        self,
    ):
        # ---------- load policy and reference model -------------
        # policy model
        rank_zero_info('building policy model into lightning model')
        # model_kwargs = {'device_map': 'balanced'} if config.trainer == 'BasicTrainer' else {}
        policy_dtype = getattr(torch, self.config.model.policy_dtype)
        self.policy = GemmaForCausalLM.from_pretrained(
            self.config.model.name_or_path, 
            low_cpu_mem_usage=True, 
            # torch_dtype=policy_dtype,
            # **model_kwargs
        )
        disable_dropout(self.policy)
        rank_zero_info('successfully built policy model into lightning model')

        # reference model
        rank_zero_info('building reference model')
        reference_model_dtype = getattr(torch, self.config.model.reference_dtype)
        self.reference = GemmaForCausalLM.from_pretrained(
            self.config.model.name_or_path, 
            low_cpu_mem_usage=True, 
            # torch_dtype=reference_model_dtype
        )
        disable_dropout(self.reference)

        rank_zero_info('successfully built reference model')

        rank_zero_info('building sae model')
        if "gemma-scope-2b" in self.config.model.sae_encoder_name_or_path:
            path_to_params = hf_hub_download(
                repo_id=self.config.model.sae_encoder_name_or_path,
                filename=self.config.model.sae_id_name_or_path,
                force_download=False,
            )
            params = np.load(path_to_params)
            pt_params = {k: torch.from_numpy(v).cuda() for k, v in params.items()}
            sae_encoder = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
            sae_encoder.load_state_dict(pt_params)

        else:
            sae_encoder = replace_sae_with_reture_feature_acts()
            sae_encoder, _, _ = sae_encoder.from_pretrained(
                release = self.config.model.sae_encoder_name_or_path, # see other options in sae_lens/pretrained_saes.yaml
                sae_id = self.config.model.sae_id_name_or_path, # won't always be a hook point
            )
            rank_zero_info( self.policy.model.layers[self.config.model.sae_layer_id])
        
        self.policy.model.layers[self.config.model.sae_layer_id].sae_encoder = sae_encoder
        self.reference.model.layers[self.config.model.sae_layer_id].sae_encoder = sae_encoder
        rank_zero_info('successfully built sae model')

        for name, param in self.policy.named_parameters():
            if 'sae' in name:
                param.requires_grad = False

        # print all submodules of policy and reference model
        rank_zero_info('policy model submodules:')
        for name, module in self.policy.named_modules():
            rank_zero_info(f'{name}')
        rank_zero_info('reference model submodules:')
        for name, module in self.reference.named_modules():
            rank_zero_info(f'{name}')
        
        # for dpo, we set the reference model to be eval mode
        for param in self.reference.parameters():
            param.requires_grad = False

        self.tokenizer = transformers.AutoTokenizer.from_pretrained("google/gemma-2b-it")
        
    def training_step(
        self, 
        batch: Dict[str, Union[List, torch.LongTensor]], 
    ):
        chosen = batch["chosen"]
        rejected = batch["rejected"]
        for i in range(len(chosen)):
            chosen[i]['role'] = chosen[i]['role'][0]
            rejected[i]['role'] = rejected[i]['role'][0]

        chosen = self.tokenizer.apply_chat_template(
            chosen, 
            return_tensors="pt", 
            padding=True, 
            return_dict=True, 
            truncation=True,
            max_length=self.config.max_length,
            padding_side="left", 
            device=self.device
        )
        rejected = self.tokenizer.apply_chat_template(
            rejected, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=self.config.max_length,
            return_dict=True, 
            padding_side="left", 
            device=self.device
        )
        
        # move the batch to the device
        chosen['input_ids'] = chosen['input_ids'].to(self.device)
        chosen['attention_mask'] = chosen['attention_mask'].to(self.device)
        rejected['input_ids'] = rejected['input_ids'].to(self.device)
        rejected['attention_mask'] = rejected['attention_mask'].to(self.device)

        # get the features and output logits
        chosen_logits_policy, chosen_feature_acts_policy = self.policy(**chosen)
        rejected_logits_policy, rejected_feature_acts_policy = self.policy(**rejected)

        # check if nan in logits and policy
        if torch.isnan(chosen_logits_policy).any() or torch.isnan(chosen_feature_acts_policy).any() or torch.isnan(rejected_logits_policy).any() or torch.isnan(rejected_feature_acts_policy).any():
            raise ValueError("nan in logits or feature_acts")

        # print all the shape with rank_zero_info
        # rank_zero_info(f'chosen_logits_policy: {chosen_logits_policy}')
        # rank_zero_info(f'chosen_feature_acts_policy: {chosen_feature_acts_policy}')
        # rank_zero_info(f'rejected_logits_policy: {rejected_logits_policy}')
        # rank_zero_info(f'rejected_feature_acts_policy: {rejected_feature_acts_policy}')

        feature_acts_chosen = chosen_feature_acts_policy.float()
        feature_acts_rejected = rejected_feature_acts_policy.float()
        policy_chosen_logps=chosen_logits_policy.float()
        policy_rejected_logps=rejected_logits_policy.float()

        with torch.no_grad():
            chosen_logits_reference, chosen_feature_acts_reference = self.reference(**chosen)
            rejected_logits_reference, rejected_feature_acts_reference = self.reference(**rejected)
            reference_chosen_logps=chosen_logits_reference.float()
            reference_rejected_logps=rejected_logits_reference.float()

        # label_smoothing=self.config.loss.label_smoothing,
        beta=self.config.loss.beta

        logits = (feature_acts_chosen.sum(dim=-1).mean(dim=1) - feature_acts_rejected.sum(dim=-1).mean(dim=1))
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(logits)

        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)

        policy_chosen_logps_softmax = F.softmax(policy_chosen_logps, dim=-1) + 1e-5
        reference_chosen_logps_softmax = F.softmax(reference_chosen_logps, dim=-1) + 1e-5

        losses = losses + beta * F.kl_div(policy_chosen_logps_softmax.log(), reference_chosen_logps_softmax)
        self.log("loss", losses, prog_bar=True, on_step=True)

        return losses

        # loss, chosen_rewards, rejected_rewards = self.feature_level_loss(
        #     feature_acts_chosen=chosen_feature_acts_policy,
        #     feature_acts_rejected=rejected_feature_acts_policy,
        #     policy_chosen_logps=chosen_logits_policy,
        #     policy_rejected_logps=rejected_logits_policy,
        #     reference_chosen_logps=chosen_logits_reference,
        #     reference_rejected_logps=rejected_logits_reference,
        #     # label_smoothing=self.config.loss.label_smoothing,
        #     beta=self.config.loss.beta,
        # )


    def configure_optimizers(self):
        config = self.config
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=config.lr_final,
        )

        def lr_lambda(current_step):
            warmup_steps = config.warmup_steps

            if current_step < warmup_steps:
                warmup_factor = current_step / warmup_steps
                return warmup_factor
            else: return 1

        lr_scheduler = {
            'scheduler': optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [lr_scheduler]