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
from .metric import (
    tdpo_loss,
    tdpo_kl_loss,
    dpo_loss
)
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

from .transformers_model.modeling_gemma2 import Gemma2ForCausalLM

# define the LightningModule
class FeatureLevelDPOModel(L.LightningModule):
    def __init__(
        self,
        config: DictConfig,
        feature_map: Tuple,
    ):
        super().__init__()
        self.config = config
        self.policy = None
        self.reference = None
        self.feature_map = feature_map
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("google/gemma-2b-it")

    def configure_model(self):
        if self.policy is not None or self.reference is not None:
            return
        else:
            # ---------- load policy and reference model -------------
            # policy model
            rank_zero_info('building policy model into lightning model')
            # model_kwargs = {'device_map': 'balanced'} if config.trainer == 'BasicTrainer' else {}
            policy_dtype = getattr(torch, self.config.model.policy_dtype)
            self.policy = Gemma2ForCausalLM.from_pretrained(
                self.config.model.name_or_path, 
                # low_cpu_mem_usage=True, 
                # torch_dtype=policy_dtype,
                # **model_kwargs
            )
            disable_dropout(self.policy)
            rank_zero_info('successfully built policy model into lightning model')

            # reference model
            rank_zero_info('building reference model')
            reference_model_dtype = getattr(torch, self.config.model.reference_dtype)
            self.reference = Gemma2ForCausalLM.from_pretrained(
                self.config.model.name_or_path, 
                low_cpu_mem_usage=True, 
                # torch_dtype=reference_model_dtype
            )
            disable_dropout(self.reference)

            rank_zero_info('successfully built reference model')

            rank_zero_info('building sae model')

            if config.release:
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
                    # rank_zero_info( self.policy.model.layers[self.config.model.sae_layer_id])
            else:
                if "Qwen1.5-0.5" in self.config.model.name_or_path:
                    sae_encoder_ = sae_encoder.load_from_pretrained(
                        path=self.config.sae_encoder_name_or_path,
                    )
            
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
        
    def training_step(
        self, 
        batch: Dict[str, Union[List, torch.LongTensor]], 
    ):
        if self.config.datasets == 'HuggingFaceH4/ultrafeedback_binarized':
            chosen = batch["chosen"]
            rejected = batch["rejected"]
            for i in range(len(chosen)):
                chosen[i]['role'] = chosen[i]['role'][0]
                rejected[i]['role'] = rejected[i]['role'][0]
        elif self.config.datasets == 'PKU-Alignment/PKU-SafeRLHF':
            chosen = batch["response_0"]
            rejected = batch["response_1"]
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
        chosen_logits_policy, chosen_feature_acts_policy = self.policy(**chosen, use_cache=False)
        rejected_logits_policy, rejected_feature_acts_policy = self.policy(**rejected, use_cache=False)

        if torch.isnan(chosen_logits_policy).any() or torch.isnan(chosen_feature_acts_policy).any() or torch.isnan(rejected_logits_policy).any() or torch.isnan(rejected_feature_acts_policy).any():
            raise ValueError("nan in logits or feature_acts")

        feature_acts_chosen = chosen_feature_acts_policy.float()
        feature_acts_rejected = rejected_feature_acts_policy.float()
        policy_chosen_logps = chosen_logits_policy.float()
        policy_rejected_logps = rejected_logits_policy.float()

        with torch.no_grad():
            chosen_logits_reference, chosen_feature_acts_reference = self.reference(**chosen, use_cache=False)
            rejected_logits_reference, rejected_feature_acts_reference = self.reference(**rejected, use_cache=False)
            reference_chosen_logps = chosen_logits_reference.float()
            reference_rejected_logps = rejected_logits_reference.float()

        # Compute TDPO loss or other loss
        beta = self.config.loss.beta
        alpha = self.config.loss.alpha
        
        if self.config.loss.name == "dpo":
            losses, chosen_rewards, rejected_rewards = dpo_loss(
                feature_acts_chosen=feature_acts_chosen,
                feature_acts_rejected=feature_acts_rejected,
                policy_chosen_logps=policy_chosen_logps,
                reference_chosen_logps=reference_chosen_logps,
                beta=beta,
            )
        elif self.config.loss.name == "tdpo":
            sae_lambda = self.config.loss.sae_lambda
            losses, chosen_rewards, rejected_rewards, chosen_kl, rejected_kl = tdpo_loss(
                policy_chosen_logps[:, :-1, :],
                reference_chosen_logps[:, :-1, :],
                policy_rejected_logps[:, :-1, :],
                reference_rejected_logps[:, :-1, :],
                feature_acts_chosen=feature_acts_chosen,
                feature_acts_rejected=feature_acts_rejected,
                c_labels=chosen['input_ids'][:, 1:],
                r_labels=rejected['input_ids'][:, 1:],
                beta=beta,
                alpha=alpha,
                sae_lambda=sae_lambda,
                if_tdpo2=True,
                if_sae=self.config.loss.sae
            )
        elif self.config.loss.name == "tdpo_kl":
            losses, chosen_rewards, rejected_rewards, chosen_kl, rejected_kl = tdpo_kl_loss(
                policy_chosen_logps[:, :-1, :].contiguous(),
                reference_chosen_logps[:, :-1, :].contiguous(),
                policy_rejected_logps[:, :-1, :].contiguous(),
                reference_rejected_logps[:, :-1, :].contiguous(),
                pi_feature_acts_chosen=feature_acts_chosen,
                pi_feature_acts_rejected=feature_acts_rejected,
                ref_feature_acts_chosen=chosen_feature_acts_reference.float(),
                ref_feature_acts_rejected=rejected_feature_acts_reference.float(),
                c_labels=chosen['input_ids'][:, 1:].contiguous(),
                r_labels=rejected['input_ids'][:, 1:].contiguous(),
                beta=beta,
                alpha=alpha,
                delta=0.5,
                feature_map=self.feature_map
            )
        else: raise ValueError("loss name not recognized")

        reward_accuracies = (chosen_rewards > rejected_rewards).float().mean()
        reward_margins = (chosen_rewards - rejected_rewards).float().mean()

        # Log metrics

        # Return all necessary metrics for callback use
        return {
            "loss": losses,
            "chosen_rewards": float(chosen_rewards.detach().mean()),
            "rejected_rewards": float(rejected_rewards.detach().mean()),
            "reward_accuracies": float(reward_accuracies.detach()),
            "reward_margins": float(reward_margins.detach()),
            "kl_chosen": float(chosen_kl.detach()),
            "kl_rejected": float(rejected_kl.detach()),
            # "logps_chosen": float(policy_chosen_logps.detach()),
            # "logps_rejected": float(policy_rejected_logps.detach()),

            # "rejected_rewards": rejected_rewards.detach().cpu().numpy(),
            # "reward_accuracies": reward_accuracies.detach().cpu().numpy(),
            # "reward_margins": reward_margins.detach().cpu().numpy(),
            # "kl_chosen": feature_acts_chosen.detach().cpu().numpy(),
            # "kl_rejected": feature_acts_rejected.detach().cpu().numpy(),
            # "logps_chosen": policy_chosen_logps.detach().cpu().numpy(),
            # "logps_rejected": policy_rejected_logps.detach().cpu().numpy(),
        }

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