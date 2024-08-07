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
            trainer.my_loss = outputs["loss"] * config.gradient_accumulation_steps
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