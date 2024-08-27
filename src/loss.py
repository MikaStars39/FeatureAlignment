import torch
from typing import Tuple
import torch.nn.functional as F

def _get_batch_logps(
    logits: torch.FloatTensor, 
    labels: torch.LongTensor, 
    attention_mask: torch.LongTensor,
    average_log_prob: bool = False, 
    token_level: bool = False,
    log=True,
    eps: float = 1e-4,
):
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
    attention_mask = attention_mask[:, :-1].float()

    # dummy token; we'll ignore the losses on these tokens later
    labels[attention_mask == 0] = 0
    distribution_logps = (logits.softmax(dim=-1) + eps).log() if log else logits.softmax(dim=-1)

    per_token_logps = torch.gather(distribution_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if token_level: 
        return (per_token_logps * attention_mask)
    elif average_log_prob:
        return (per_token_logps * attention_mask).sum(-1) / attention_mask.sum(-1)
    else:
        return (per_token_logps * attention_mask).sum(-1)

def _get_feature_map(
    feature_map: torch.FloatTensor,
    labels: torch.LongTensor,
    attention_mask: torch.LongTensor,
    log=True,
):
    # not per token feature map
    assert feature_map.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    feature_map = feature_map[:, :-1, :].softmax(-1)
    attention_mask = attention_mask[:, :-1].float()

    if log:
        feature_map = (feature_map + 1e-4).log()

    feature_map = feature_map * attention_mask.unsqueeze(-1)

    return feature_map


def dpo_loss(
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        # labels
        c_labels: torch.LongTensor,
        r_labels: torch.LongTensor,
        # mask
        c_mask: torch.Tensor,
        r_mask: torch.Tensor,
        beta: float,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

    policy_chosen_logps = _get_batch_logps(policy_chosen_logps, c_labels, c_mask, average_log_prob=False)
    policy_rejected_logps = _get_batch_logps(policy_rejected_logps, r_labels, r_mask, average_log_prob=False)
    reference_chosen_logps = _get_batch_logps(reference_chosen_logps, c_labels, c_mask, average_log_prob=False)
    reference_rejected_logps = _get_batch_logps(reference_rejected_logps, r_labels, r_mask, average_log_prob=False)

    """Compute the DPO loss for a batch of policy and reference model log probabilities."""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    chosen_kl = (reference_chosen_logps * (reference_chosen_logps - policy_chosen_logps)).sum(-1)
    rejected_kl = (reference_rejected_logps * (reference_rejected_logps - policy_rejected_logps)).sum(-1)

    return losses.mean(), chosen_rewards, rejected_rewards, chosen_kl.mean(), rejected_kl.mean()


def tdpo_kl_loss(
    # logits
    pi_chosen_logits: torch.Tensor,
    ref_chosen_logits: torch.Tensor,
    pi_rejected_logits: torch.Tensor,
    ref_rejected_logits: torch.Tensor,
    # features
    pi_feature_acts_chosen: torch.Tensor,
    pi_feature_acts_rejected: torch.Tensor,
    ref_feature_acts_chosen: torch.Tensor,
    ref_feature_acts_rejected: torch.Tensor,
    # labels
    c_labels: torch.Tensor,
    r_labels: torch.Tensor,
    # mask
    c_mask: torch.Tensor,
    r_mask: torch.Tensor,
    # hyperparameters
    beta: float,
    alpha: float,
    delta: float,
    chosen_feature_map: torch.Tensor,
    rejected_feature_map: torch.Tensor,
) -> torch.FloatTensor:

    # feature softmax
    if chosen_feature_map is not None and rejected_feature_map is not None:
        # repeat feature map from h, to n, h
        chosen_fm, rejected_fm = chosen_feature_map, rejected_feature_map
        chosen_fm = chosen_fm.unsqueeze(0).repeat(pi_chosen_logits.size(1), 1).unsqueeze(0).to(pi_feature_acts_chosen.device)
        rejected_fm = rejected_fm.unsqueeze(0).repeat(pi_rejected_logits.size(1), 1).unsqueeze(0).to(pi_feature_acts_rejected.device)

        pi_feature_acts_chosen = pi_feature_acts_chosen * chosen_fm
        pi_feature_acts_rejected = pi_feature_acts_rejected * rejected_fm
        ref_feature_acts_chosen = ref_feature_acts_chosen * chosen_fm
        ref_feature_acts_rejected = ref_feature_acts_rejected * rejected_fm

    pi_feature_acts_chosen = _get_feature_map(pi_feature_acts_chosen, c_labels, c_mask, log=False)
    pi_feature_acts_rejected = _get_feature_map(pi_feature_acts_rejected, r_labels, r_mask, log=False)
    ref_feature_acts_chosen = _get_feature_map(ref_feature_acts_chosen, c_labels, c_mask, log=False)
    ref_feature_acts_rejected = _get_feature_map(ref_feature_acts_rejected, r_labels, r_mask, log=False)
    log_ref_chosen_feature_acts_chosen = (ref_feature_acts_chosen + 1e-4).log()
    log_ref_rejected_feature_acts_rejected = (ref_feature_acts_rejected + 1e-4).log()

    # token-level dpo
    pi_chosen_per_token_logps = _get_batch_logps(pi_chosen_logits, c_labels, c_mask, average_log_prob=False, token_level=True)
    ref_chosen_per_token_logps = _get_batch_logps(ref_chosen_logits, c_labels, c_mask, average_log_prob=False, token_level=True)
    pi_rejected_per_token_logps = _get_batch_logps(pi_rejected_logits, r_labels, r_mask, average_log_prob=False, token_level=True)
    ref_rejected_per_token_logps = _get_batch_logps(ref_rejected_logits, r_labels, r_mask, average_log_prob=False, token_level=True)

    chosen_rewards = pi_chosen_per_token_logps - ref_chosen_per_token_logps
    rejected_rewards = pi_rejected_per_token_logps - ref_rejected_per_token_logps
    
    # debug
    d0 = min(chosen_rewards.shape[0], rejected_rewards.shape[0])
    chosen_rewards = chosen_rewards[:d0].contiguous()
    rejected_rewards = rejected_rewards[:d0].contiguous()
    rewards = chosen_rewards - rejected_rewards

    # token kl
    ref_chosen_vocab_ps = _get_batch_logps(ref_chosen_logits, c_labels, c_mask, average_log_prob=False, token_level=True, log=False)
    ref_rejected_vocab_ps = _get_batch_logps(ref_rejected_logits, r_labels, r_mask, average_log_prob=False, token_level=True, log=False)

    token_chosen_kl = (ref_chosen_vocab_ps * (ref_chosen_per_token_logps - pi_chosen_per_token_logps))
    token_rejected_kl = (ref_rejected_vocab_ps * (ref_rejected_per_token_logps - pi_rejected_per_token_logps))

    # feature kl
    feature_chosen_kl = (ref_feature_acts_chosen * (log_ref_chosen_feature_acts_chosen - pi_feature_acts_chosen)).sum(-1)
    feature_rejected_kl = (ref_feature_acts_rejected * (log_ref_rejected_feature_acts_rejected - pi_feature_acts_rejected)).sum(-1)

    # total kl
    chosen_kl = token_chosen_kl * (1 - delta) + feature_chosen_kl * delta
    rejected_kl = token_rejected_kl * (1 - delta) + feature_rejected_kl * delta

    # loss
    values = rewards - alpha * (rejected_kl - chosen_kl)
    losses = -F.logsigmoid(beta * values)

    chosen_kl = token_chosen_kl.mean()
    rejected_kl = token_rejected_kl.mean()

    return losses.mean(), chosen_rewards, rejected_rewards, chosen_kl, rejected_kl

def tdpo_loss(
    pi_chosen_logits: torch.Tensor,
    ref_chosen_logits: torch.Tensor,
    pi_rejected_logits: torch.Tensor,
    ref_rejected_logits: torch.Tensor,
    feature_acts_chosen: torch.Tensor,
    feature_acts_rejected: torch.Tensor,
    c_labels: torch.Tensor,
    r_labels: torch.Tensor,
    c_mask: torch.Tensor,
    r_mask: torch.Tensor,
    beta: float,
    alpha: float,
    sae_lambda: float = 0.2,
    if_tdpo2: bool = True,
    if_sae: bool = True,
) -> torch.FloatTensor:

    pi_chosen_per_token_logps = _get_batch_logps(pi_chosen_logits, c_labels, c_mask, average_log_prob=False, token_level=True)
    ref_chosen_per_token_logps = _get_batch_logps(ref_chosen_logits, c_labels, c_mask, average_log_prob=False, token_level=True)
    pi_rejected_per_token_logps = _get_batch_logps(pi_rejected_logits, r_labels, r_mask, average_log_prob=False, token_level=True)
    ref_rejected_per_token_logps = _get_batch_logps(ref_rejected_logits, r_labels, r_mask, average_log_prob=False, token_level=True)

    chosen_rewards = pi_chosen_per_token_logps - ref_chosen_per_token_logps
    rejected_rewards = pi_rejected_per_token_logps - ref_rejected_per_token_logps
    rewards = chosen_rewards - rejected_rewards

    # token kl
    ref_chosen_vocab_ps = _get_batch_logps(ref_chosen_logits, c_labels, c_mask, average_log_prob=False, token_level=True, log=False)
    ref_rejected_vocab_ps = _get_batch_logps(ref_rejected_logits, r_labels, r_mask, average_log_prob=False, token_level=True, log=False)

    token_chosen_kl = (ref_chosen_vocab_ps * (ref_chosen_per_token_logps - pi_chosen_per_token_logps))
    token_rejected_kl = (ref_rejected_vocab_ps * (ref_rejected_per_token_logps - pi_rejected_per_token_logps))
    
    # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)

    if not if_tdpo2:
        values = rewards - (token_rejected_kl - token_chosen_kl)
    else:
        values = rewards - alpha * (token_rejected_kl - token_chosen_kl.detach())

    losses = -F.logsigmoid(beta * values)
    return losses.mean(), chosen_rewards, rejected_rewards, token_chosen_kl.mean(), token_rejected_kl.mean()

# def tdpo_loss(
#     chosen_labels: torch.Tensor,
#     rejected_labels: torch.Tensor,
#     policy_chosen_logps: torch.Tensor,
#     policy_rejected_logps: torch.Tensor,
#     reference_chosen_logps: torch.Tensor,
#     reference_rejected_logps: torch.Tensor,
#     chosen_loss_mask: torch.Tensor,
#     rejected_loss_mask: torch.Tensor,
#     feature_acts_chosen: torch.Tensor,
#     feature_acts_rejected: torch.Tensor,
#     beta: float, 
#     alpha: float = 0.5, 
#     sae_lambda: float = 1.0,  # SAE regularization coefficient
# ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

#     chosen_labels = chosen_labels[:, 1:].clone()
#     rejected_labels = rejected_labels[:, 1:].clone()

#     policy_chosen_logps = policy_chosen_logps[:, :-1, :]
#     policy_rejected_logps = policy_rejected_logps[:, :-1, :]
#     reference_chosen_logps = reference_chosen_logps[:, :-1, :]
#     reference_rejected_logps = reference_rejected_logps[:, :-1, :]

#     chosen_labels[chosen_loss_mask[:, :-1] == 0] = 0
#     rejected_labels[rejected_loss_mask[:, :-1] == 0] = 0

#     policy_chosen_logps = policy_chosen_logps.log_softmax(dim=-1)
#     policy_rejected_logps = policy_rejected_logps.log_softmax(dim=-1)
#     reference_chosen_logps = reference_chosen_logps.log_softmax(dim=-1)
#     reference_rejected_logps = reference_rejected_logps.log_softmax(dim=-1)

#     chosen_per_token_logps = torch.gather(policy_chosen_logps, dim=2, index=chosen_labels.unsqueeze(2)).squeeze(2)
#     rejected_per_token_logps = torch.gather(policy_rejected_logps, dim=2, index=rejected_labels.unsqueeze(2)).squeeze(2)
#     chosen_per_reference_token_logps = torch.gather(reference_chosen_logps, dim=2, index=chosen_labels.unsqueeze(2)).squeeze(2)
#     rejected_per_reference_token_logps = torch.gather(reference_rejected_logps, dim=2, index=rejected_labels.unsqueeze(2)).squeeze(2)



#     chosen_logps_margin = chosen_per_token_logps - chosen_per_reference_token_logps
#     rejected_logps_margin = rejected_per_token_logps - rejected_per_reference_token_logps

#     chosen_logps_margin = (chosen_logps_margin * chosen_loss_mask[:, :-1]) \
#         / (chosen_loss_mask[:, :-1].sum(dim=1) * chosen_loss_mask[:, :-1].shape[1])
#     rejected_logps_margin = (rejected_logps_margin * rejected_loss_mask[:, :-1]) \
#         / (rejected_loss_mask[:, :-1].sum(dim=1) * chosen_loss_mask[:, :-1].shape[1])
#     chosen_kl = (chosen_kl * chosen_loss_mask[:, :-1]).sum(dim=1) \
#     / (chosen_loss_mask[:, :-1].sum(dim=1) * chosen_loss_mask[:, :-1].shape[1])
#     rejected_kl = (rejected_kl * rejected_loss_mask[:, :-1]).sum(dim=1) \
#     / (rejected_loss_mask[:, :-1].sum(dim=1) * chosen_loss_mask[:, :-1].shape[1])

#     chosen_rejected_logps_margin = chosen_logps_margin.sum(dim=1) - rejected_logps_margin.sum(dim=1)
#     logits = chosen_rejected_logps_margin - alpha * (rejected_kl - chosen_kl.detach())  # TDPO2
#     losses = -F.logsigmoid(beta * logits)

#     # print(losses)

#     sae_reg = sae_lambda * ((feature_acts_chosen.mean(dim=1) - feature_acts_rejected.mean(dim=1)) ** 2).sum(dim=-1)
#     losses += sae_reg
#     # print(sae_reg)

#     chosen_rewards = beta * (chosen_logps_margin).detach().mean(dim=1)
#     rejected_rewards = beta * (rejected_logps_margin).detach().mean(dim=1)

#     return losses.mean(), chosen_rewards, rejected_rewards

# def tdpo_loss(
#     chosen_logps_margin, 
#     rejected_logps_margin, 
#     chosen_position_kl, 
#     rejected_position_kl, 
#     policy_chosen_logps, 
#     policy_rejected_logps,
# ) -> Tuple:

#     losses, chosen_rewards, rejected_rewards = tdpo_loss(chosen_logps_margin, rejected_logps_margin,
#                                                             chosen_position_kl, rejected_position_kl,
#                                                             beta=loss_config.beta, alpha=loss_config.alpha, if_tdpo2=loss_config.if_tdpo2)

#     reward_accuracies = (chosen_rewards > rejected_rewards).float()

#     chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
#     rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
#     reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

#     metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
#     metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
#     metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
#     metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

#     all_device_chosen_position_kl = all_gather_if_needed(chosen_position_kl.detach(), self.rank, self.world_size)
#     all_device_rejected_position_kl = all_gather_if_needed(rejected_position_kl.detach(), self.rank, self.world_size)

#     metrics[f'kl_{train_test}/chosen'] = all_device_chosen_position_kl.cpu().numpy().tolist()
#     metrics[f'kl_{train_test}/rejected'] = all_device_rejected_position_kl.cpu().numpy().tolist()
#     metrics[f'kl_{train_test}/margin'] = (all_device_chosen_position_kl - all_device_rejected_position_kl).cpu().numpy().tolist()

#     policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
#     metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()
