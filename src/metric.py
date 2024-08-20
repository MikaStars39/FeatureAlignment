import torch
from typing import Tuple
import torch.nn.functional as F

def dpo_loss(
    feature_acts_chosen: torch.Tensor,
    feature_acts_rejected: torch.Tensor,
    policy_chosen_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    beta: float,
):
    logits = (feature_acts_chosen.mean(dim=1) - feature_acts_rejected.mean(dim=1)).sum(dim=-1)
    # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
    losses = -F.logsigmoid(logits)
    policy_chosen_logps_softmax = F.softmax(policy_chosen_logps, dim=-1) + 1e-5
    reference_chosen_logps_softmax = F.softmax(reference_chosen_logps, dim=-1) + 1e-5

    losses = losses + beta * F.kl_div(policy_chosen_logps_softmax.log(), reference_chosen_logps_softmax)
    return losses, None, None


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
    # label
    c_labels: torch.Tensor,
    r_labels: torch.Tensor,
    # hyperparameters
    beta: float,
    alpha: float,
) -> torch.FloatTensor:

    # trunctuation
    min_len = min(pi_chosen_logits.size(1), pi_rejected_logits.size(1))
    pi_chosen_logits = pi_chosen_logits[:, :min_len, :]
    ref_chosen_logits = ref_chosen_logits[:, :min_len, :]
    pi_rejected_logits = pi_rejected_logits[:, :min_len, :]
    ref_rejected_logits = ref_rejected_logits[:, :min_len, :]
    pi_feature_acts_chosen = pi_feature_acts_chosen[:, :min_len, :]
    pi_feature_acts_rejected = pi_feature_acts_rejected[:, :min_len, :]
    ref_feature_acts_chosen = pi_feature_acts_chosen[:, :min_len, :]
    ref_feature_acts_rejected = pi_feature_acts_rejected[:, :min_len, :]
    c_labels = c_labels[:, :min_len]
    r_labels = r_labels[:, :min_len]

    # rewards
    pi_chosen_vocab_logps = (pi_chosen_logits.softmax(-1) + 1e-3).log()
    ref_chosen_vocab_ps = ref_chosen_logits.softmax(-1)
    ref_chosen_vocab_logps = (ref_chosen_vocab_ps + 1e-3).log()

    pi_rejected_vocab_logps = (pi_rejected_logits.softmax(-1) + 1e-3).log()
    ref_rejected_vocab_ps = ref_rejected_logits.softmax(-1)
    ref_rejected_vocab_logps = (ref_rejected_vocab_ps + 1e-3).log()

    pi_chosen_per_token_logps = torch.gather(pi_chosen_vocab_logps, dim=2, index=c_labels.unsqueeze(2)).squeeze(2)
    ref_chosen_per_token_logps = torch.gather(ref_chosen_vocab_logps, dim=2, index=c_labels.unsqueeze(2)).squeeze(2)

    pi_rejected_per_token_logps = torch.gather(pi_rejected_vocab_logps, dim=2, index=r_labels.unsqueeze(2)).squeeze(2)
    ref_rejected_per_token_logps = torch.gather(ref_rejected_vocab_logps, dim=2, index=r_labels.unsqueeze(2)).squeeze(2)

    chosen_rewards = pi_chosen_per_token_logps - ref_chosen_per_token_logps
    rejected_rewards = pi_rejected_per_token_logps - ref_rejected_per_token_logps
    rewards = chosen_rewards - rejected_rewards

    # kl
    pi_rejected_feature_logps = (pi_feature_acts_rejected.softmax(-1) + 1e-3).log()
    pi_chosen_feature_logps = (pi_feature_acts_chosen.softmax(-1) + 1e-3).log()
    ref_rejected_feature_logps = ref_feature_acts_rejected.softmax(-1)
    ref_chosen_feature_logps = ref_feature_acts_chosen.softmax(-1)

    token_chosen_kl = (ref_chosen_vocab_ps * (ref_chosen_vocab_logps - pi_chosen_vocab_logps)).sum(-1)
    token_rejected_kl = (ref_rejected_vocab_ps * (ref_rejected_vocab_logps - pi_rejected_vocab_logps)).sum(-1)

    # sum over the last dimension
    chosen_kl = F.kl_div(pi_rejected_feature_logps, ref_rejected_feature_logps, reduction='none').sum(-1)
    # chosen_kl = F.kl_div(pi_rejected_feature_logps, ref_rejected_feature_logps, reduction='none')
    rejected_kl = F.kl_div(pi_chosen_feature_logps, ref_chosen_feature_logps, reduction='none').sum(-1)

    print(rewards)

    values = rewards - alpha * (rejected_kl - chosen_kl.detach())

    losses = -F.logsigmoid(beta * values)

    return losses.mean(), chosen_rewards, rejected_rewards, token_chosen_kl.mean(), token_rejected_kl.mean()



def tdpo_loss(
    pi_chosen_logits: torch.Tensor,
    ref_chosen_logits: torch.Tensor,
    pi_rejected_logits: torch.Tensor,
    ref_rejected_logits: torch.Tensor,
    feature_acts_chosen: torch.Tensor,
    feature_acts_rejected: torch.Tensor,
    c_labels: torch.Tensor,
    r_labels,
    beta: float,
    alpha: float,
    sae_lambda: float = 0.2,
    if_tdpo2: bool = True,
    if_sae: bool = True,
) -> torch.FloatTensor:
    """
    计算 TDPO 损失函数。
    
    pi_chosen_logits: policy chosen logits, 形状: (batch_size, sequence_length, vocab_size)
    ref_chosen_logits: reference chosen logits, 形状: (batch_size, sequence_length, vocab_size)
    pi_rejected_logits: policy rejected logits, 形状: (batch_size, sequence_length, vocab_size)
    ref_rejected_logits: reference rejected logits, 形状: (batch_size, sequence_length, vocab_size)
    labels: 标签，用于计算 log 概率, 形状: (batch_size, sequence_length)
    beta: 控制 KL 惩罚项的强度
    alpha: 调整 KL 散度在每个 token 上的影响权重
    if_tdpo2: 使用 TDPO2 方法，默认为 True；如果为 False，则切换为 TDPO1 方法
    """

    min_len = min(pi_chosen_logits.size(1), pi_rejected_logits.size(1))
    
    # 只计算较短长度的序列
    pi_chosen_logits = pi_chosen_logits[:, :min_len, :]
    ref_chosen_logits = ref_chosen_logits[:, :min_len, :]
    pi_rejected_logits = pi_rejected_logits[:, :min_len, :]
    ref_rejected_logits = ref_rejected_logits[:, :min_len, :]
    feature_acts_chosen = feature_acts_chosen[:, :min_len, :]
    feature_acts_rejected = feature_acts_rejected[:, :min_len, :]
    c_labels = c_labels[:, :min_len]
    r_labels = r_labels[:, :min_len]

    # sae_reg = (feature_acts_chosen - feature_acts_rejected).sum(dim=-1)
    # use mse loss
    sae_reg = (feature_acts_chosen.mean(dim=1) - feature_acts_rejected.mean(dim=1)).sum(dim=-1)

    pi_chosen_vocab_logps = pi_chosen_logits.log_softmax(-1)
    ref_chosen_vocab_ps = ref_chosen_logits.softmax(-1)
    ref_chosen_vocab_logps = (ref_chosen_vocab_ps + 1e-3).log()

    pi_rejected_vocab_logps = pi_rejected_logits.log_softmax(-1)
    ref_rejected_vocab_ps = ref_rejected_logits.softmax(-1)
    ref_rejected_vocab_logps = (ref_rejected_vocab_ps + 1e-3).log()

    pi_chosen_per_token_logps = torch.gather(pi_chosen_vocab_logps, dim=2, index=c_labels.unsqueeze(2)).squeeze(2)
    ref_chosen_per_token_logps = torch.gather(ref_chosen_vocab_logps, dim=2, index=c_labels.unsqueeze(2)).squeeze(2)

    pi_rejected_per_token_logps = torch.gather(pi_rejected_vocab_logps, dim=2, index=r_labels.unsqueeze(2)).squeeze(2)
    ref_rejected_per_token_logps = torch.gather(ref_rejected_vocab_logps, dim=2, index=r_labels.unsqueeze(2)).squeeze(2)

    chosen_rewards = pi_chosen_per_token_logps - ref_chosen_per_token_logps
    rejected_rewards = pi_rejected_per_token_logps - ref_rejected_per_token_logps
    rewards = chosen_rewards - rejected_rewards

    chosen_kl = (ref_chosen_vocab_ps * (ref_chosen_vocab_logps - pi_chosen_vocab_logps)).sum(-1)
    
    rejected_kl = (ref_rejected_vocab_ps * (ref_rejected_vocab_logps - pi_rejected_vocab_logps)).sum(-1)

    # sae_logits = (feature_acts_chosen.mean(dim=1) - feature_acts_rejected.mean(dim=1)).sum(dim=-1)
    # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)

    if not if_tdpo2:
        values = rewards - (rejected_kl - chosen_kl)
    else:
        values = rewards - alpha * (rejected_kl - chosen_kl.detach())

    losses = (-F.logsigmoid(beta * values) + sae_lambda * sae_reg) if if_sae else -F.logsigmoid(beta * values)
    return losses.mean(), chosen_rewards + chosen_kl, rejected_rewards + rejected_kl, chosen_kl.mean(), rejected_kl.mean()

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
