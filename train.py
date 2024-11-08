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
import hydra
from omegaconf import DictConfig
from lightning import Trainer, seed_everything
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import DictConfig, OmegaConf
from feature_alignment.utils.util import instantiate


def configure_date(config: DictConfig, tokenizer):
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
    data_loader_class = instantiate(config.loss.dataloader, instantiate_module=False)
    train_iterator = data_loader_class(
        config.datasets, 
        tokenizer,
        split='train',
        batch_size=config.train_bs,
        n_epochs=1e7 if config.n_examples is None else config.n_epochs,
        n_examples=config.n_examples, 
        **data_iterator_kwargs
    )
    eval_iterator = data_loader_class(
        config.datasets, 
        tokenizer,
        split='test',
        batch_size=config.eval_bs,
        n_examples=config.n_eval_examples, 
        n_epochs=(1 if config.n_eval_examples is None else None),
        **data_iterator_kwargs
    )
    return train_iterator, eval_iterator
    

@hydra.main(config_path="config", config_name="config")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""

    # ----------- check missing key in config -----------
    missing_keys = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    # ----------------- seed everything and login -----------------
    seed_everything(config.seed)
    from huggingface_hub import login
    login(token=config.hf_token)

    # ----------------- load callbacks ------------------
    rank_zero_info(f"Loading callbacks from {config.callbacks}")
    callbacks = [instantiate(cb) for cb in config.callbacks]

    # # ------------------- load logger -------------------
    if config.debug == False:
        if hasattr(config.logger, "neptune_api_token") and config.logger.neptune_api_token is not None:
            from lightning.pytorch.loggers import NeptuneLogger

            logger = NeptuneLogger(
                api_key=config.logger.neptune_api_token,
                project=config.logger.neptune_project,
            )
        else:
            from lightning.pytorch.loggers import WandbLogger

            logger = WandbLogger(
                project=config.logger.wandb.project,
                name=config.exp_name,
            )
            logger.log_hyperparams(config)
    else:
        logger = None

    # # ----------------- load trainer -------------------
    rank_zero_info(f"Loading trainer from {config.trainer}")
    if "FSDP" in config.trainer.strategy:
        from lightning.pytorch.strategies import FSDPStrategy
        strategy = FSDPStrategy(
            sharding_strategy=config.trainer.fsdp_sharding_strategy,
            state_dict_type=config.trainer.fsdp_state_dict_type,
        )
        config.trainer.strategy = strategy
    trainer = Trainer(
        **config.trainer, 
        callbacks=callbacks, 
        logger=logger,
    )

    # # ----------------- load model ---------------------
    module = instantiate(config.loss.model, instantiate_module=False)
    rank_zero_info(f"Loading model from {config.loss.model.module_name}.{config.loss.model.class_name}")
    if config.resume_ckpt is not None:
        model = module.load_from_checkpoint(config.resume_ckpt)
    else:
        model = module(config=config)
    
    # ----------------- load tokenizer -----------------
    rank_zero_info(f'Loading tokenizer {config.model.hf_tokenizer_name_or_path}')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.hf_tokenizer_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ----------------- load data -----------------------
    rank_zero_info("Loading data")
    train_dataloader, eval_dataloader = configure_date(config, tokenizer)

    # ----------------- train model ---------------------
    trainer.fit(
        model,
        train_dataloader,
    )


if __name__ == "__main__":
    main()
