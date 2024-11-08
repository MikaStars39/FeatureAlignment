import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Union, Tuple
import lightning as L
from .model import BasicModel
from ..utils.util import pad_to_length
from .dpo import DPOModel


class TDPOModel(DPOModel):
    """TDPO-1/2 Trainer."""

    def forward(
            self, 
            model: nn.Module, 
            batch: Dict[str, Union[List, torch.LongTensor]], 
            average_log_prob=False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
           Return two tensors of shape (batch size), one of the chosen examples, another of the rejected ones.

           Returns:
            chosen_logps: log probabilities of chosen examples (should be batch size / 2 if data was read in correctly)
            rejected_logps: log probabilities of rejected examples (should be batch size / 2 if data was read in correctly)
        """
        concatenated_batch = self.concatenated_inputs(batch)
        all_logits = model(concatenated_batch['concatenated_combined_input_ids'], attention_mask=concatenated_batch['concatenated_combined_attention_mask'], use_cache=(not self.is_mistral)).logits.to(self.policy_dtype)
        all_logps = get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=average_log_prob)
        chosen_logps = all_logps[:batch['chosen_combined_input_ids'].shape[0]]
        rejected_logps = all_logps[batch['chosen_combined_input_ids'].shape[0]:]
        return chosen_logps, rejected_logps

    def get_batch_metrics(
            self, 
            batch: Dict[str, Union[List, torch.LongTensor]], 
            mode: str=None,
    ) -> Tuple[torch.FloatTensor, Dict]:
        """Compute the loss and other metrics for the given batch of inputs."""
        
        metrics = {}
        if mode is None: mode = self.config.mode

        if self.reference_model is None:
            policy_chosen_logps, policy_rejected_logps = self.forward(self.policy, batch)
            losses, chosen_rewards, rejected_rewards = self.loss(policy_chosen_logps, policy_rejected_logps)
        else:
            policy_chosen_logps, policy_rejected_logps = self.forward(self.policy, batch)
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.forward(self.reference_model, batch)
            losses, chosen_rewards, rejected_rewards = self.loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps)

        # accuracy calculated on unpaired examples (for apples-to-apples comparison with UnpairedPreferenceTrainer)
        reward_accuracies = (chosen_rewards > rejected_rewards.flip(dims=[0])).float()

        # detach what needs to be detached
        chosen_rewards = chosen_rewards.detach().float().cpu()
        rejected_rewards = rejected_rewards.detach().float().cpu()
        reward_accuracies = reward_accuracies.detach().float().cpu()
        policy_chosen_logps = policy_chosen_logps.detach().float().cpu()
        policy_rejected_logps = policy_rejected_logps.detach().float().cpu()

        metrics[f'rewards_{mode}/chosen'] = chosen_rewards
        metrics[f'rewards_{mode}/rejected'] = rejected_rewards
        metrics[f'rewards_{mode}/accuracies'] = reward_accuracies
        metrics[f'rewards_{mode}/margins'] = (chosen_rewards - rejected_rewards)
        metrics[f'logps_{mode}/rejected'] = policy_rejected_logps
        metrics[f'logps_{mode}/chosen'] = policy_chosen_logps

        return losses.mean(), metrics