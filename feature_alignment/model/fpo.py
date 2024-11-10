import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Union, Tuple
from ..utils.util import detach_float_metrics, instantiate
from .dpo import DPOModel

def fpo_get_batch_logps(
    logits: torch.FloatTensor, 
    reference_logits: torch.FloatTensor, 
    labels: torch.LongTensor,
    pi_fm: torch.FloatTensor = None,
    ref_fm: torch.FloatTensor = None,
    average_log_prob: bool = False,
    temperature: float = 1,
    k: int = 50,
):
    """Compute the kl divergence/log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        reference_logits: Logits of the reference model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        Several tensors of shape (batch_size,) containing the average/sum kl divergence/log probabilities of the given labels under the given logits.
    """

    assert logits.shape[:-1] == labels.shape
    assert reference_logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    vocab_logps = logits.log_softmax(-1)

    reference_vocab_ps = reference_logits.softmax(-1)
    reference_vocab_logps = reference_vocab_ps.log()

    per_position_kl = (reference_vocab_ps * (reference_vocab_logps - vocab_logps)).sum(-1)
    per_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2) * loss_mask
    per_reference_token_logps = torch.gather(reference_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2) * loss_mask
    logps_margin = (per_token_logps).sum(-1) / loss_mask.sum(-1) - (per_reference_token_logps).sum(-1) / loss_mask.sum(-1)

    if pi_fm is not None:
        pi_fm = pi_fm[:, :-1, :]
        ref_fm = ref_fm[:, :-1, :]
    
    if pi_fm is not None:
        ref_fm = (ref_fm * loss_mask.unsqueeze(-1)).mean(dim=1)
        pi_fm = (pi_fm * loss_mask.unsqueeze(-1)).mean(dim=1)

        # # L2 Norm
        # ref_fm = ref_fm / ref_fm.norm(dim=-1, keepdim=True)
        # pi_fm = pi_fm / pi_fm.norm(dim=-1, keepdim=True)

        pi_fm, indices = torch.topk(pi_fm, k, dim=-1)
        ref_fm = torch.gather(ref_fm, dim=-1, index=indices)

        fm_sae = (ref_fm - pi_fm).pow(2).mean(-1)
    else:
        fm_sae = torch.zeros_like(per_position_kl).sum(-1)


    if average_log_prob:
        return (logps_margin * loss_mask).sum(-1) / loss_mask.sum(-1), \
               (per_position_kl * loss_mask).sum(-1) / loss_mask.sum(-1),
    else:
        return logps_margin, \
            (per_position_kl * loss_mask).sum(-1), \
            fm_sae
            

class FPOModel(DPOModel):

    def configure_sae(self):
        from feature_alignment.sae.jump_relu_sae import load_jump_relu_sae
        self.sae_encoder = load_jump_relu_sae(self.config)

        # freeze
        for param in self.sae_encoder.parameters():
            param.requires_grad = False

    def loss(
        self, 
        chosen_logps_margin: torch.FloatTensor,
        rejected_logps_margin: torch.FloatTensor,
        chosen_position_mse: torch.FloatTensor,
        rejected_position_mse: torch.FloatTensor,
    ) :
        """Compute the TDPO loss for a batch of policy and reference model log probabilities.

        Args:
            chosen_logps_margin: The difference of log probabilities between the policy model and the reference model for the chosen responses. Shape: (batch_size,)
            rejected_logps_margin: The difference of log probabilities between the policy model and the reference model for the rejected responses. Shape: (batch_size,)
            chosen_position_mse: The difference of sequential kl divergence between the policy model and the reference model for the chosen responses. Shape: (batch_size,)
            rejected_position_mse: The difference of sequential kl divergence between the policy model and the reference model for the rejected responses. Shape: (batch_size,)


        Returns:
            A tuple of two tensors: (losses, rewards).
            The losses tensor contains the TDPO loss for each example in the batch.
            The rewards tensors contain the rewards for response pair.
        """

        chosen_values = chosen_logps_margin + chosen_position_mse
        rejected_values = rejected_logps_margin + rejected_position_mse

        chosen_rejected_logps_margin = chosen_logps_margin - rejected_logps_margin

        alpha = self.config.loss.alpha
        beta = self.config.loss.beta
        logits = chosen_rejected_logps_margin - \
            alpha * (rejected_position_mse - chosen_position_mse.detach())
        losses = -F.logsigmoid(beta * logits)

        chosen_rewards = beta * chosen_values.detach()
        rejected_rewards = beta * rejected_values.detach()

        return losses, chosen_rewards, rejected_rewards
       
    def forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], average_log_prob=False) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        """
        concatenated_batch = self.concatenated_inputs(batch)
        outputs = model(
            concatenated_batch['concatenated_combined_input_ids'], 
            attention_mask=concatenated_batch['concatenated_combined_attention_mask'], 
            use_cache=(not self.is_mistral),
            output_hidden_states=True,
        )
        all_logits = outputs.logits.to(self.precision)
        all_fm = outputs.hidden_states[-1].to(self.precision)
        all_fm = self.sae_encoder.encode(all_fm)
        
        with torch.no_grad():
            reference_outputs = self.reference_model(
                concatenated_batch['concatenated_combined_input_ids'], 
                attention_mask=concatenated_batch['concatenated_combined_attention_mask'], 
                use_cache=(not self.is_mistral),
                output_hidden_states=True,
            )
            reference_all_logits = reference_outputs.logits.to(self.precision)
            reference_all_fm = reference_outputs.hidden_states[-1].to(self.precision)
            reference_all_fm = self.sae_encoder.encode(reference_all_fm)

        all_logps_margin, all_position_kl, all_fm_mse = fpo_get_batch_logps(
            all_logits, 
            reference_all_logits, 
            concatenated_batch['concatenated_labels'], 
            all_fm, 
            reference_all_fm, 
        )

        chosen_logps_margin = all_logps_margin[:batch['chosen_input_ids'].shape[0]]
        rejected_logps_margin = all_logps_margin[batch['chosen_input_ids'].shape[0]:]
        chosen_position_kl = all_position_kl[:batch['chosen_input_ids'].shape[0]]
        rejected_position_kl = all_position_kl[batch['chosen_input_ids'].shape[0]:]
        chosen_fm_mse = all_fm_mse[:batch['chosen_input_ids'].shape[0]]
        rejected_fm_mse = all_fm_mse[batch['chosen_input_ids'].shape[0]:]

        return chosen_logps_margin, rejected_logps_margin, chosen_position_kl, rejected_position_kl, chosen_fm_mse, rejected_fm_mse

    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str=None):
        """Compute the loss and other metrics for the given batch of inputs."""
        metrics = {}
        if mode is None: mode = self.config.mode

        chosen_logps_margin, rejected_logps_margin, chosen_position_kl, rejected_position_kl, chosen_fm_mse, rejected_fm_mse \
                = self.forward(self.policy, batch)
        
        losses, chosen_rewards, rejected_rewards = self.loss(
            chosen_logps_margin,
            rejected_logps_margin,
            chosen_fm_mse, 
            rejected_fm_mse,
        )

        # accuracy calculated on unpaired examples (for apples-to-apples comparison with UnpairedPreferenceTrainer)
        reward_accuracies = (chosen_rewards > rejected_rewards.flip(dims=[0])).float()

        fm_mse = (chosen_fm_mse - rejected_fm_mse).detach()
        losses = losses.mean()

        metrics[f'rewards_{mode}/chosen'] = chosen_rewards
        metrics[f'rewards_{mode}/rejected'] = rejected_rewards
        metrics[f'rewards_{mode}/accuracies'] = reward_accuracies
        metrics[f'rewards_{mode}/margins'] = (chosen_rewards - rejected_rewards)
        metrics[f'kl_{mode}/chosen'] = chosen_position_kl
        metrics[f'kl_{mode}/rejected'] = rejected_position_kl
        metrics[f'kl_{mode}/margin'] = (chosen_position_kl - rejected_position_kl)
        metrics[f'kl_{mode}/fm margin'] = fm_mse
        metrics[f'kl_{mode}/fm chosen'] = chosen_fm_mse
        metrics[f'kl_{mode}/fm rejected'] = rejected_fm_mse
        metrics[f'loss/{mode}'] = losses.clone()

        metrics = detach_float_metrics(metrics)

        return losses, metrics   