import os
import torch
import transformers
import lightning as L
import torch.nn.functional as F

from torch import optim, nn, utils, Tensor
from transformers import AutoModelForCausalLM
from omegaconf import DictConfig
from lightning.pytorch.utilities import rank_zero_info, rank_zero_only
from utils import disable_dropout
from .sae import replace_sae_with_reture_feature_acts
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

# from dpo
# def preference_loss(
#     policy_chosen_logps: torch.FloatTensor,
#     policy_rejected_logps: torch.FloatTensor,
#     reference_chosen_logps: torch.FloatTensor,
#     reference_rejected_logps: torch.FloatTensor,
#     beta: float,
#     label_smoothing: float = 0.0,
#     ipo: bool = False,
#     reference_free: bool = False
# ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
#     """Compute the DPO loss for a batch of policy and reference model log probabilities.

#     Args:
#         policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
#         policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
#         reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
#         reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
#         beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
#         label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
#         ipo: If True, use the IPO loss instead of the DPO loss.
#         reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

#     Returns:
#         A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
#         The losses tensor contains the DPO loss for each example in the batch.
#         The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
#     """
#     pi_logratios = policy_chosen_logps - policy_rejected_logps
#     ref_logratios = reference_chosen_logps - reference_rejected_logps

#     if reference_free:
#         ref_logratios = 0

#     logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

#     if ipo:
#         losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
#     else:
#         # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
#         losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

#     chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
#     rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

#     return losses, chosen_rewards, rejected_rewards



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
    
    # def feature_level_loss(
    #     feature_acts_chosen: torch.Tensor,
    #     feature_acts_rejected: torch.Tensor,
    #     policy_chosen_logps: torch.FloatTensor,
    #     policy_rejected_logps: torch.FloatTensor,
    #     reference_chosen_logps: torch.FloatTensor,
    #     reference_rejected_logps: torch.FloatTensor,
    #     # label_smoothing: float,
    #     beta: float,

    # ):
    #     # compute the feature level loss
    #     logits = (feature_acts_chosen - feature_acts_rejected).mean(dim=1)

    #     # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
    #     losses = -F.logsigmoid(logits) * (1 - label_smoothing)

    #     chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    #     rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    #     policy_chosen_logps_softmax = F.softmax(policy_chosen_logps, dim=-1)
    #     reference_chosen_logps_softmax = F.softmax(reference_chosen_logps, dim=-1)

    #     losses = losses - beta * F.kl_div(policy_chosen_logps_softmax.log(), reference_chosen_logps_softmax)

    #     return losses, chosen_rewards, rejected_rewards
        
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

        logits = (feature_acts_chosen.mean(dim=1) - feature_acts_rejected.mean(dim=1))

        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(logits)

        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

        policy_chosen_logps_softmax = F.softmax(policy_chosen_logps, dim=-1)
        reference_chosen_logps_softmax = F.softmax(reference_chosen_logps, dim=-1)

        losses = losses - beta * F.kl_div(policy_chosen_logps_softmax.log(), reference_chosen_logps_softmax)

        return losses.mean()

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

        self.log('loss', losses.item())

        return losses


    # def get_batch_metrics(
    #     self, 
    #     beta: float,
    #     label_smoothing: float,
    #     policy_chosen_logits: torch.Tensor,
    #     policy_rejected_logits: torch.Tensor,
    #     reference_chosen_logits: torch.Tensor,
    #     reference_rejected_logits: torch.Tensor,
    #     train=True,
    # ):
    #     """Compute the SFT or DPO loss and other metrics for the given batch of inputs."""

    #     # policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch)
    #     # with torch.no_grad():
    #     #     reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(self.reference_model, batch)

    #     # loss_kwargs = {
    #     #     'beta': loss_config.beta, 
    #     #     'reference_free': loss_config.reference_free, 
    #     #     'label_smoothing': loss_config.label_smoothing, 
    #     #     'ipo': False
    #     # }

    #     losses, chosen_rewards, rejected_rewards = preference_loss(
    #         policy_chosen_logps=policy_chosen_logits, 
    #         policy_rejected_logps=policy_rejected_logits, 
    #         reference_chosen_logps=reference_chosen_logits, 
    #         reference_rejected_logps=reference_rejected_logits, 
    #         beta=beta,
    #         label_smoothing=label_smoothing,
    #     )

    #     reward_accuracies = (chosen_rewards > rejected_rewards).float()

    #     chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
    #     rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
    #     reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

    #     # metrics for 
    #     metrics = {}
    #     train_test = 'train' if train else 'eval'
    #     metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
    #     metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
    #     metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
    #     metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
    #     # TODO finish this
    #     # policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
    #     # metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()
    #     # policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
    #     # metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()
    #     # all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
    #     # metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()

    #     return losses.mean(), metrics

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer