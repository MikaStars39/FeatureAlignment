import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Union
import lightning as L
from .model import BasicModel
from .utils import get_batch_logps

class SFTModel(BasicModel):
    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], mode: str=None):
        """Compute the loss and other metrics for the given batch of inputs.
        
        Args:
            batch: dictionary of inputs for the batch (should contain 'target_attention_mask', 'target_input_input_ids', 
                'target_labels' where 'target' corresponds to the SFT example)
            mode: one of 'train', 'eval', 'sample'
        """
        metrics = {}
        if mode is None: mode = self.config.mode
        
        policy_chosen_logits = self.policy(
            batch['target_combined_input_ids'], 
            attention_mask=batch['target_combined_attention_mask'], 
            use_cache=(not self.is_mistral)
        ).logits

        policy_chosen_logps = get_batch_logps(
            policy_chosen_logits, 
            batch['target_labels'], 
            average_log_prob=True
        )
        loss = -policy_chosen_logps.mean()

        metrics['loss'] = loss
        metrics[f'logps_{mode}/chosen'] = policy_chosen_logps
        metrics[f'loss/{mode}'] = loss
        return metrics