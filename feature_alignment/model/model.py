from typing import Dict

import lightning as L
import torch
import torch.nn as nn
from typing import Any, Dict, List, Tuple, Union
from omegaconf import DictConfig
from ..utils.util import instantiate

class BasicModel(L.LightningModule):
    """
    A `LightningModule`
    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.is_mistral = False
        self.policy = None
        self.configuration()

    def on_train_start(self) -> None:
        # Get the rank of the current process after the trainer is attached
        pass

    # --------------------- forward function ---------------------

    def forward(self) -> torch.Tensor:
        """
        This will be called if we directly call the model
        e.g. model(noisy_latents, timesteps)
        TODO: check if this is needed
        """
        pass

    # ------ training / testing / validation / inference loop ------

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        pass

    def test_step(self, batch: Dict, batch_idx: int):
        pass

    def validation_step(self, batch: Dict, batch_idx: int):
        pass

    def predict_step(self, batch: Dict, batch_idx: int, dataloader_idx: int = None):
        pass

    # ------------------- configure everything -------------------

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
            betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
            eps=self.config.optimizer.adam_epsilon,
        )

        def lr_lambda(current_step):
            warmup_steps = self.config.optimizer.warmup_steps

            if current_step < warmup_steps:
                warmup_factor = current_step / warmup_steps
                return warmup_factor
            else:
                return 1

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [lr_scheduler]

    def configuration(self):
        """
        Our customised configuration, for getting the scheduler, vae, etc.
        NOTICE: must be freezed models
        """
        # precision config 
        if "bf" in self.config.trainer.precision:
            self.precision = torch.bfloat16
        elif "fp16" in self.config.trainer.precision:
            self.precision = torch.float16
        else: self.precision = torch.float32

    def configure_model(self):
        """
        Get the trainable models. Don't use self.xxx = xxx in __init__ because
        this will result in initializing the model on all GPUs.
        docs: https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html#speed-up-model-initialization
        """
        if self.policy is None:
            self.policy = instantiate(self.config.model, instantiate_module=False)
            if self.config.model.hf_model_name_or_path is not None:
                self.policy = self.policy.from_pretrained(self.config.model.hf_model_name_or_path)
                self.policy.to(self.device).to(self.precision)
            else: raise ValueError("No model name or path provided")
        


