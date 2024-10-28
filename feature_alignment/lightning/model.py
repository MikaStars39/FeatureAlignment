from typing import Dict

import lightning as L
import torch
import torch.nn as nn
from typing import Any, Dict, List, Tuple, Union
from omegaconf import DictConfig

class BasicModel(L.LightningModule):
    """
    A `LightningModule`
    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(self, config: DictConfig):
        super().__init__()

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
        return self.get_batch_metrics(batch, mode="train")

    def test_step(self, batch: Dict, batch_idx: int):
        pass

    def validation_step(self, batch: Dict, batch_idx: int):
        pass

    def predict_step(self, batch: Dict, batch_idx: int, dataloader_idx: int = None):
        pass

    # ------------------- configure everything -------------------

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        get the optimizer and scheduler
        docs: https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        """
        pass

    def configuration(self):
        """
        Our customised configuration, for getting the scheduler, vae, etc.
        NOTICE: must be freezed models
        """
        pass

    def configure_model(self):
        """
        Get the trainable models. Don't use self.xxx = xxx in __init__ because
        this will result in initializing the model on all GPUs.
        docs: https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html#speed-up-model-initialization
        """

    # --------------------- internal / custom methods ---------------------

    def get_batch_metrics(
        self, 
        batch: Dict[str, Union[List, torch.LongTensor]], 
        mode: str=None
    ) -> Tuple[torch.FloatTensor, Dict]:
        """Compute the loss and other metrics for the given batch of inputs.
        
        Arg:
            batch: dictionary of inputs for the batch (what is required will vary depending on the trainer)
            mode: one of 'train', 'eval', 'sample'
        """
        raise NotImplementedError

    def a_custom_method(self):
        pass
