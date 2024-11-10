import time

import lightning as L
from lightning.pytorch.utilities import rank_zero_info, rank_zero_only


class BasicCallback(L.Callback):
    def __init__(self):
        super().__init__()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        real_step = trainer.global_step  # + config.epoch_begin * config.epoch_steps
        # TODO
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
        self.log("lr", lr, prog_bar=True, on_step=True)
        self.log("step", int(real_step), prog_bar=False, on_step=True)
