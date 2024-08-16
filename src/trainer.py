import os, math, time, datetime, subprocess
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.utilities import rank_zero_info, rank_zero_only

class train_callback(L.Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        config = self.config
        real_step = trainer.global_step + config.epoch_begin * config.epoch_steps
        if trainer.is_global_zero:  # logging
            t_now = time.time_ns()
            kt_s = 0
            try:
                t_cost = (t_now - trainer.my_time_ns) / 1e9
                self.log("REAL it/s", 1.0 / t_cost, prog_bar=True, on_step=True)
            except:
                pass

            for param_group in trainer.optimizers[0].param_groups:
                lr = param_group["lr"]
                break

            trainer.my_lr = lr

            trainer.my_time_ns = t_now
        #             return {
        #     "loss": losses,
        #     "chosen_rewards": chosen_rewards.detach().cpu().numpy().tolist(),
        #     "rejected_rewards": rejected_rewards.detach().cpu().numpy().tolist(),
        #     "reward_accuracies": reward_accuracies.detach().cpu().numpy().tolist(),
        #     "reward_margins": reward_margins.detach().cpu().numpy().tolist(),
        #     "kl_chosen": feature_acts_chosen.detach().cpu().numpy().tolist(),
        #     "kl_rejected": feature_acts_rejected.detach().cpu().numpy().tolist(),
        #     "logps_chosen": policy_chosen_logps.detach().cpu().numpy().tolist(),
        #     "logps_rejected": policy_rejected_logps.detach().cpu().numpy().tolist(),
        # }

            self.log("loss", outputs["loss"] * config.gradient_accumulation_steps, prog_bar=True, on_step=True)
            self.log("chosen_rewards", outputs["chosen_rewards"], prog_bar=False, on_step=True)
            self.log("rejected_rewards", outputs["rejected_rewards"], prog_bar=False, on_step=True)
            self.log("reward_accuracies", outputs["reward_accuracies"], prog_bar=False, on_step=True)
            self.log("reward_margins", outputs["reward_margins"], prog_bar=False, on_step=True)
            self.log("kl_chosen", outputs["kl_chosen"], prog_bar=False, on_step=True)
            self.log("kl_rejected", outputs["kl_rejected"], prog_bar=False, on_step=True)
            # self.log("logps_chosen", outputs["logps_chosen"], prog_bar=False, on_step=True)
            # self.log("logps_rejected", outputs["logps_rejected"], prog_bar=False, on_step=True)
            self.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
            # self.log("s", real_step, prog_bar=True, on_step=True)

        #     if len(config.wandb) > 0:
        #         lll = {"loss": trainer.my_loss, "lr": trainer.my_lr, "wd": trainer.my_wd, "Gtokens": real_step * token_per_step / 1e9}
        #         if kt_s > 0:
        #             lll["kt/s"] = kt_s
        #         trainer.my_wandb.log(lll, step=int(real_step))
        # if (trainer.is_global_zero) or ('deepspeed_stage_3' in config.strategy): # save pth
        #     if config.magic_prime > 0:
        #         expand_factor = 2 if config.my_qa_mask > 0 else 1
        #         if int(real_step) == int(config.magic_prime * expand_factor // config.real_bsz) - 1 + int(config.my_random_steps):
        #             to_save_dict = pl_module.state_dict()
        #             my_save(
        #                 config, trainer,
        #                 to_save_dict,
        #                 f"{config.proj_dir}/rwkv-final.pth",
        #             )
                

    def on_train_epoch_start(self, trainer, pl_module):
        pass
    def on_train_epoch_end(self, trainer, pl_module):
        pass
    
    # def on_after_backward(self, trainer, pl_module):
    #     # calculate and log the gradnorm
    #     total_norm = 0
    #     for p in pl_module.parameters():
    #         if p.grad is not None:
    #             param_norm = p.grad.data.norm(2)
    #             total_norm += param_norm.item() ** 2
    #     total_norm = total_norm ** 0.5
    #     trainer.my_grad_norm = total_norm
    #     self.log("grad_norm", trainer.my_grad_norm, prog_bar=True, on_step=True)