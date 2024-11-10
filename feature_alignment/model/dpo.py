import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Union, Tuple
import lightning as L
from .model import BasicModel
from ..utils.util import pad_to_length
from .sft import get_batch_logps

class DPOModel(BasicModel):
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        
        loss, metrics = self.get_batch_metrics(batch, mode="train")
        self.log_dict(metrics, sync_dist=True)
        self.log("loss", loss, prog_bar=True, on_step=True)
        return loss
    
    """A trainer for any loss that uses paired preference, like DPO."""
    def concatenated_inputs(
            self, 
            batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor. The first half is chosen outputs, the second half is rejected.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            
        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        max_length = max(batch['chosen_combined_input_ids'].shape[1], batch['rejected_combined_input_ids'].shape[1])
        concatenated_batch = {}
        for k in batch:
            if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
                pad_value = -100 if 'labels' in k else 0
                concatenated_key = k.replace('chosen', 'concatenated')
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
                pad_value = -100 if 'labels' in k else 0
                concatenated_key = k.replace('rejected', 'concatenated')
                concatenated_batch[concatenated_key] = torch.cat((
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value),
                ), dim=0)
        return concatenated_batch
    
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

        all_logits = model(
            concatenated_batch['concatenated_combined_input_ids'], 
            attention_mask=concatenated_batch['concatenated_combined_attention_mask'], use_cache=(not self.is_mistral)
        ).logits.to(self.precision)

        all_logps = get_batch_logps(
            all_logits, 
            concatenated_batch['concatenated_labels'], 
            average_log_prob=average_log_prob
        )
        
        chosen_logps = all_logps[:batch['chosen_combined_input_ids'].shape[0]]
        rejected_logps = all_logps[batch['chosen_combined_input_ids'].shape[0]:]
        return chosen_logps, rejected_logps, all_logits

    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str=None):
        """Compute the loss and other metrics for the given batch of inputs."""
        metrics = {}
        if mode is None: mode = self.config.mode

        if self.reference_model is None:
            policy_chosen_logps, policy_rejected_logps = self.forward(
                self.policy, batch
            )
            losses, chosen_rewards, rejected_rewards = self.loss(
                policy_chosen_logps, policy_rejected_logps
            )
        else:
            policy_chosen_logps, policy_rejected_logps, all_logits = self.forward(
                self.policy, batch
            )
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps, reference_all_logits = self.forward(
                    self.reference_model, batch
                )
            losses, chosen_rewards, rejected_rewards = self.loss(
                policy_chosen_logps, policy_rejected_logps, 
                reference_chosen_logps, reference_rejected_logps
            )
        
        chosen_rewards = chosen_rewards.float().detach()
        rejected_rewards = rejected_rewards.float().detach()
        policy_chosen_logps = policy_chosen_logps.float().detach()
        policy_rejected_logps = policy_rejected_logps.float().detach()

        # accuracy calculated on unpaired examples 
        # (for apples-to-apples comparison with UnpairedPreferenceTrainer)
        reward_accuracies = (
            chosen_rewards > rejected_rewards.flip(dims=[0])
        ).float()
        metrics[f'rewards_{mode}/chosen'] = chosen_rewards
        metrics[f'rewards_{mode}/rejected'] = rejected_rewards
        metrics[f'rewards_{mode}/accuracies'] = reward_accuracies
        metrics[f'rewards_{mode}/margins'] = (chosen_rewards - rejected_rewards)
        metrics[f'logps_{mode}/rejected'] = policy_rejected_logps
        metrics[f'logps_{mode}/chosen'] = policy_chosen_logps
        metrics[f'loss/{mode}'] = losses.mean()

        return losses.mean(), metrics

    def loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities."""
        
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        losses = -F.logsigmoid(self.config.loss.beta * logits)
        chosen_rewards = self.config.loss.beta * (
            policy_chosen_logps - reference_chosen_logps
        ).detach()
        rejected_rewards = self.config.loss.beta * (
            policy_rejected_logps - reference_rejected_logps
        ).detach()

        return losses, chosen_rewards, rejected_rewards