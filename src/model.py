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
from .utils import disable_dropout, process_text
from .sae import replace_sae_with_reture_feature_acts
from .jump_relu_sae import JumpReLUSAE
from .loss import (
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
from .transformers_model.modeling_qwen2 import Qwen2ForCausalLM

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
        self.chosen_feature_map, self.rejected_feature_map = feature_map
        if self.chosen_feature_map is not None:
            self.chosen_feature_map.to(self.device)
            self.rejected_feature_map.to(self.device)

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config.model.name_or_path)
    
    def _get_model_name(self):
        if "gemma2" in self.config.model.name_or_path:
            return Gemma2ForCausalLM
        elif "Qwen1.5" in self.config.model.name_or_path:
            return Qwen2ForCausalLM

    def configure_model(self):
        if self.policy is not None or self.reference is not None:
            return
        else:
            # ---------- load policy and reference model -------------
            MODEL = self._get_model_name()
            # policy model
            rank_zero_info(f'building policy model {self.config.model.name_or_path}')
            # model_kwargs = {'device_map': 'balanced'} if config.trainer == 'BasicTrainer' else {}
            policy_dtype = getattr(torch, self.config.model.policy_dtype)
            self.policy = MODEL.from_pretrained(
                self.config.model.name_or_path, 
                # low_cpu_mem_usage=True, 
                # torch_dtype=policy_dtype,
                # **model_kwargs
            )
            disable_dropout(self.policy)
            rank_zero_info('successfully built policy model into lightning model')

            # reference model
            rank_zero_info(f'building reference model {self.config.model.name_or_path}')
            reference_model_dtype = getattr(torch, self.config.model.reference_dtype)
            self.reference = MODEL.from_pretrained(
                self.config.model.name_or_path, 
                low_cpu_mem_usage=True, 
                # torch_dtype=reference_model_dtype
            )
            disable_dropout(self.reference)

            rank_zero_info('successfully built reference model')

            rank_zero_info('building sae model')
            sae_encoder = replace_sae_with_reture_feature_acts()
            rank_zero_info(f'building sae{self.config.model.sae_encoder_name_or_path}')
            if self.config.model.release:
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
                    # sae_encoder = replace_sae_with_reture_feature_acts()
                    sae_encoder, _, _ = sae_encoder.from_pretrained(
                        release = self.config.model.sae_encoder_name_or_path, # see other options in sae_lens/pretrained_saes.yaml
                        sae_id = self.config.model.sae_id_name_or_path, # won't always be a hook point
                    )
                    # rank_zero_info( self.policy.model.layers[self.config.model.sae_layer_id])
            else:
                if "Qwen1.5-0.5" in self.config.model.name_or_path:
                    sae_encoder = sae_encoder.load_from_pretrained(
                        path=self.config.model.sae_encoder_name_or_path,
                        dtype="float16"
                    )
            
            self.policy.model.layers[self.config.model.sae_layer_id].sae_encoder = sae_encoder
            self.reference.model.layers[self.config.model.sae_layer_id].sae_encoder = sae_encoder
            rank_zero_info('successfully built sae model')

            for name, param in self.policy.named_parameters():
                if 'sae' in name:
                    param.requires_grad = False

            # print all submodules of policy and reference model
            # rank_zero_info('policy model submodules:')
            # for name, module in self.policy.named_modules():
            #     rank_zero_info(f'{name}')
            # rank_zero_info('reference model submodules:')
            # for name, module in self.reference.named_modules():
            #     rank_zero_info(f'{name}')
            
            # for dpo, we set the reference model to be eval mode
            for param in self.reference.parameters():
                param.requires_grad = False
        
    def training_step(
        self, 
        batch: Dict[str, Union[List, torch.LongTensor]], 
    ):
        
        chosen, rejected = self._process_batch(batch)

        # get the features and output logits
        chosen_logits_policy, chosen_feature_acts_policy = self.policy(**chosen, use_cache=False)
        rejected_logits_policy, rejected_feature_acts_policy = self.policy(**rejected, use_cache=False)

        if torch.isnan(chosen_logits_policy).any() or \
            torch.isnan(chosen_feature_acts_policy).any() or \
            torch.isnan(rejected_logits_policy).any() or \
            torch.isnan(rejected_feature_acts_policy).any():
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

        c_labels=chosen['input_ids']
        r_labels=rejected['input_ids']
        
        if self.config.loss.name == "dpo":
            losses, chosen_rewards, rejected_rewards, chosen_kl, rejected_kl = dpo_loss(
                policy_chosen_logps=policy_chosen_logps,
                policy_rejected_logps=policy_rejected_logps,
                reference_chosen_logps=reference_chosen_logps,
                reference_rejected_logps=reference_rejected_logps,
                c_labels=c_labels,
                c_mask=chosen['attention_mask'],
                r_labels=r_labels,
                r_mask=rejected['attention_mask'],
                beta=beta,
            )
        elif self.config.loss.name == "tdpo":
            sae_lambda = self.config.loss.sae_lambda
            losses, chosen_rewards, rejected_rewards, chosen_kl, rejected_kl = tdpo_loss(
                policy_chosen_logps,
                reference_chosen_logps,
                policy_rejected_logps,
                reference_rejected_logps,
                feature_acts_chosen=feature_acts_chosen,
                feature_acts_rejected=feature_acts_rejected,
                c_labels=c_labels,
                r_labels=r_labels,
                c_mask=chosen['attention_mask'],
                r_mask=rejected['attention_mask'],
                beta=beta,
                alpha=alpha,
                sae_lambda=sae_lambda,
                if_tdpo2=True,
                if_sae=self.config.loss.sae
            )
        elif self.config.loss.name == "tdpo_kl":
            losses, chosen_rewards, rejected_rewards, chosen_kl, rejected_kl = tdpo_kl_loss(
                policy_chosen_logps,
                reference_chosen_logps,
                policy_rejected_logps,
                reference_rejected_logps,
                pi_feature_acts_chosen=feature_acts_chosen,
                pi_feature_acts_rejected=feature_acts_rejected,
                ref_feature_acts_chosen=chosen_feature_acts_reference,
                ref_feature_acts_rejected=rejected_feature_acts_reference,
                c_labels=c_labels,
                r_labels=r_labels,
                c_mask=chosen['attention_mask'],
                r_mask=rejected['attention_mask'],
                beta=beta,
                alpha=alpha,
                delta=0.5,
                chosen_feature_map=self.chosen_feature_map,
                rejected_feature_map=self.rejected_feature_map,
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
        }

    def test_step(
        self, 
        batch: Dict[str, Union[List, torch.LongTensor]], 
        batch_idx: int,
    ):
        # this is the test loop
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    def _process_batch(self, batch):
        if self.config.datasets == 'HuggingFaceH4/ultrafeedback_binarized':
            chosen = batch["chosen"]
            rejected = batch["rejected"]
        else: raise NotImplementedError("Not a support dataset")

        batch_size = len(chosen)

        # joint text process
        chosen = process_text(
                self.config,
                chosen,
                batch_size,
                self.tokenizer
            )
        rejected = process_text(
                self.config,
                rejected,
                batch_size,
                self.tokenizer
            )

        # connect two lists
        chosen.extend(rejected)

        combined = self.tokenizer(
            chosen,
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=self.config.max_length,
        )

        chosen = {k: v[:batch_size, :] for k, v in combined.items()}
        rejected = {k: v[batch_size:, :] for k, v in combined.items()}
        
        # move the batch to the device
        chosen['input_ids'] = chosen['input_ids'].to(self.device)
        chosen['attention_mask'] = chosen['attention_mask'].to(self.device)
        rejected['input_ids'] = rejected['input_ids'].to(self.device)
        rejected['attention_mask'] = rejected['attention_mask'].to(self.device)

        return chosen, rejected
    
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