import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Union
import lightning as L
from .model import BasicModel

def get_batch_logps(
    logits: torch.FloatTensor, 
    labels: torch.LongTensor, 
    average_log_prob: bool = False, 
    token_level: bool = False
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
        token_level: If true, return the token-level log probabilities (do not aggregate across tokens)

    Returns:
        The relevant log probabilities. Of shape (batch_size,) by default and shape (batch size, sequence length) if token_level.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0
    distribution_logps = logits.log_softmax(-1)

    per_token_logps = torch.gather(distribution_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if token_level: 
        return (per_token_logps * loss_mask)
    elif average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


class SFTModel(BasicModel):

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        
        metrics = self.get_batch_metrics(batch, mode="train")
        self.log_dict(metrics, sync_dist=True)
        self.log("loss", metrics["loss"], prog_bar=True, on_step=True)
        return metrics
        
    def get_batch_metrics(
        self, 
        batch: Dict[str, Union[List, torch.LongTensor]], 
        mode: str=None,
    ):
        """Compute the loss and other metrics for the given batch of inputs.
        
        Args:
            batch: dictionary of inputs for the batch (should contain 'target_attention_mask', 'target_input_input_ids', 
                'target_labels' where 'target' corresponds to the SFT example)
            mode: one of 'train', 'eval', 'sample'
        """
        metrics = {}
        if mode is None: mode = self.config.mode
        
        policy_chosen_logits = self.policy(
            batch['target_combined_input_ids'].to(self.device), 
            attention_mask=batch['target_combined_attention_mask'].to(self.device), 
            use_cache=(not self.is_mistral)
        ).logits

        policy_chosen_logps = get_batch_logps(
            policy_chosen_logits, 
            batch['target_labels'].to(self.device), 
            average_log_prob=True
        )
        loss = -policy_chosen_logps.mean()

        metrics['loss'] = loss
        metrics[f'logps_{mode}/chosen'] = policy_chosen_logps
        metrics[f'loss/{mode}'] = loss
        return metrics