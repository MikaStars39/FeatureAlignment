import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Union, Tuple
from ..utils.util import detach_float_metrics
from .dpo import DPOModel, get_batch_logps

class SimPOModel(DPOModel):
    def loss(
        self, 
        chosen_logps_margin: torch.FloatTensor,
        rejected_logps_margin: torch.FloatTensor,
    ) :
        """Compute the TDPO loss for a batch of policy and reference model log probabilities.

        Args:
            chosen_logps_margin: The difference of log probabilities between the policy model and the reference model for the chosen responses. Shape: (batch_size,)
            rejected_logps_margin: The difference of log probabilities between the policy model and the reference model for the rejected responses. Shape: (batch_size,)
            chosen_position_kl: The difference of sequential kl divergence between the policy model and the reference model for the chosen responses. Shape: (batch_size,)
            rejected_position_kl: The difference of sequential kl divergence between the policy model and the reference model for the rejected responses. Shape: (batch_size,)


        Returns:
            A tuple of two tensors: (losses, rewards).
            The losses tensor contains the TDPO loss for each example in the batch.
            The rewards tensors contain the rewards for response pair.
        """

        chosen_values = chosen_logps_margin
        rejected_values = rejected_logps_margin
        chosen_rejected_logps_margin = chosen_logps_margin - rejected_logps_margin
        
        alpha = self.config.loss.alpha
        beta = self.config.loss.beta
        gamma = self.config.loss.gamma
        logits = beta * (alpha * chosen_rejected_logps_margin - gamma)
        losses = -F.logsigmoid(logits)

        chosen_rewards = beta * chosen_values.detach()
        rejected_rewards = beta * rejected_values.detach()

        return losses, chosen_rewards, rejected_rewards
    
    def forward(
        self, 
        model: nn.Module, 
        batch: Dict[str, Union[List, torch.LongTensor]], 
        average_log_prob=True, # simpo is always average log prob
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

        policy_chosen_logps, policy_rejected_logps, all_logits = self.forward(
            self.policy, batch
        )

        with torch.no_grad():
            reference_chosen_logps, reference_rejected_logps, reference_all_logits \
                = self.forward(
                self.reference_model, batch
            )
        
        losses, chosen_rewards, rejected_rewards = self.loss(
            policy_chosen_logps, policy_rejected_logps
        )

        # accuracy calculated on unpaired examples (for apples-to-apples comparison with UnpairedPreferenceTrainer)
        reward_accuracies = (
            chosen_rewards > rejected_rewards.flip(dims=[0])
        ).float()
        losses = losses.mean()

        metrics[f'rewards_{mode}/chosen'] = chosen_rewards
        metrics[f'rewards_{mode}/rejected'] = rejected_rewards
        metrics[f'rewards_{mode}/accuracies'] = reward_accuracies
        metrics[f'rewards_{mode}/margins'] = (chosen_rewards - rejected_rewards)
        metrics[f'logps_{mode}/rejected'] = policy_rejected_logps
        metrics[f'logps_{mode}/chosen'] = policy_chosen_logps
        metrics[f'loss/{mode}'] = losses.clone()

        metrics = detach_float_metrics(metrics)

        return losses, metrics 