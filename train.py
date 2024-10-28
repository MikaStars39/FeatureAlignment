"""
Main script for training.

Sample use is:

python train.py loss=ppo model=llama30b datasets=[shp,hh,oasst] exp_name=archangel_sft+ppo_llama30b mode=train \
     ++cache_dir=/data/models/archangel ++model.load_from=archangel_sft_llama30b/LATEST/policy.pt

where
- loss should have a file under config/loss that specifies the trainer in trainers.py and dataloader in dataloader.py
- model should have a file under config/model
- datasets is a list of datasets, each of which has a get_{name} function in dataloader.py
- exp_name is the experiment name (on WANDB); model will be saved to the cache_dir/exp_name
- model.load_from should be used for aligning a model that has already been finetuned

Remember to allocate enough RAM before running this (you need aroundd 800 GB for Llama-13B).
"""
import fire
from lightning import Trainer, seed_everything
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer
from feature_alignment.utils.util import instantiate


def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""

    # ----------- check missing key in config -----------
    missing_keys = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    # ----------------- seed everything -----------------
    seed_everything(config.seed)

    # ----------------- load callbacks ------------------
    rank_zero_info(f"Loading callbacks from {config.callbacks}")
    callbacks = [instantiate(cb) for cb in config.callbacks]

    # # ------------------- load logger -------------------
    # if config.debug == False:
    #     if hasattr(config.logger, "neptune_api_token") and config.logger.neptune_api_token is not None:
    #         from lightning.pytorch.loggers import NeptuneLogger

    #         logger = NeptuneLogger(
    #             api_key=config.logger.neptune_api_token,
    #             project=config.logger.neptune_project,
    #         )
    #     else:
    #         from lightning.pytorch.loggers import WandbLogger

    #         logger = WandbLogger(
    #             project=config.logger.wandb.project,
    #             name=config.exp_name,
    #         )
    #         logger.log_hyperparams(config)
    # else:
    #     logger = None

    # # ----------------- load trainer -------------------
    # rank_zero_info(f"Loading trainer from {config.trainer}")
    # trainer = Trainer(**config.trainer, callbacks=callbacks, logger=logger)

    # # ----------------- load model ---------------------
    # module = instantiate(config.model, instantiate_module=False)
    # rank_zero_info(f"Loading model from {config.model.module_name}.{config.model.class_name}")
    # if config.resume_ckpt is not None:
    #     model = module.load_from_checkpoint(config.resume_ckpt)
    # else:
    #     model = module(config=config)
    
    # # ----------------- load tokenizer -----------------
    # rank_zero_info(f'Loading tokenizer {config.model.tokenizer_name_or_path}')
    # tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_name_or_path)
    # if tokenizer.pad_token_id is None:
    #     tokenizer.pad_token_id = tokenizer.eos_token_id

    # ----------------- load data -----------------------
    rank_zero_info("Loading data")
    data_iterator_kwargs = dict(
        max_length=config.model.max_length,
        max_prompt_length=config.model.max_prompt_length,
        human_prefix=config.data.human_prefix,
        human_suffix=config.data.human_suffix,
        assistant_prefix=config.data.assistant_prefix,
        assistant_suffix=config.data.assistant_suffix,
        seed=config.seed,
        frac_unique_desirable=config.data.frac_unique_desirable,
        frac_unique_undesirable=config.data.frac_unique_undesirable,
        # control tokens taken from Korbak et al.'s (2023) "Pretraining Models with Human Feedback"
        # SFTDataLoader will use them for sampling; ConditionalSFTDataLoader for training
        chosen_control_token=(config.loss.chosen_control_token if config.loss.name == "csft" else None),
        rejected_control_token=(config.loss.rejected_control_token if config.loss.name == "csft" else None),
    )
    data_loader_class = instantiate(config.dataloader, instantiate_module=False)
    train_iterator = data_loader_class(
        config.datasets, 
        tokenizer,
        split='train',
        batch_size=config.model.batch_size,
        n_epochs=config.n_epochs,
        n_examples=config.n_examples, 
        **data_iterator_kwargs
    )
    eval_iterator = data_loader_class(
        config.datasets, 
        tokenizer,
        split='test',
        batch_size=config.model.eval_batch_size,
        n_examples=config.n_eval_examples, 
        n_epochs=(1 if config.n_eval_examples is None else None),
        **data_iterator_kwargs
    )
    
    # ----------------- train model ---------------------
    trainer.fit(model, train_iterator, eval_iterator)


def run(config="./configs/config.yaml"):
    config = OmegaConf.load(config)
    main(config)


if __name__ == "__main__":
    fire.Fire(run)
