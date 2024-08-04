import torch
torch.backends.cuda.matmul.allow_tf32 = True
import transformers
import os
import time
import hydra
import wandb
import lightning as L
import torch.distributed as dist

from omegaconf import OmegaConf, DictConfig
from lightning.pytorch.strategies import FSDPStrategy, DDPStrategy
from typing import Set
from utils import get_local_run_dir
from src.model import FeatureLevelDPOModel



OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))

# def worker_main(rank: int, world_size: int, config: DictConfig, policy: nn.Module, reference_model: Optional[nn.Module] = None):
#     """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
#     if 'FSDP' in config.trainer:
#         init_distributed(rank, world_size, port=config.fsdp_port)
    
#     if config.debug:
#         wandb.init = lambda *args, **kwargs: None
#         wandb.log = lambda *args, **kwargs: None

#     if rank == 0 and config.wandb.enabled:
#         os.environ['WANDB_CACHE_DIR'] = get_local_dir(config.local_dirs)
#         wandb.init(
#             entity=config.wandb.entity,
#             project=config.wandb.project,
#             config=OmegaConf.to_container(config),
#             dir=get_local_dir(config.local_dirs),
#             name=config.exp_name,
#         )

#     TrainerClass = getattr(trainers, config.trainer)
#     print(f'Creating trainer on process {rank} with world size {world_size}')
#     trainer = TrainerClass(policy, config, config.seed, config.local_run_dir, reference_model=reference_model, rank=rank, world_size=world_size)

#     trainer.train()
#     trainer.save()

def check_and_modify_config(config: DictConfig):

    # ---------- check config ---------------
    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    if config.eval_every % config.batch_size != 0:
        print('WARNING: eval_every must be divisible by batch_size')
        print('Setting eval_every to', config.eval_every - config.eval_every % config.batch_size)
        config.eval_every = config.eval_every - config.eval_every % config.batch_size
    
    # lightning so we don't use that
    # if 'FSDP' in config.trainer and config.fsdp_port is None:
    #     free_port = get_open_port()
    #     print('no FSDP port specified; using open port for FSDP:', free_port)
    #     config.fsdp_port = free_port

    # set the local dictionary

    # ---------- set local dir ---------------
    current_directory = os.getcwd()
    # we save all our logs and result under results
    # name is like {exp_name}_{model.name_or_path}_{loss.name}_{time}
    current_time = time.localtime()
    folder_name = f'{config.exp_name}_{config.model.name_or_path}_{config.loss.name} \
    _{current_time.tm_year}-{current_time.tm_mon}-{current_time.tm_mday}- \
    {current_time.tm_hour}-{current_time.tm_min}-{current_time.tm_sec}'
    if config.local_run_dir is None or not os.path.exists(config.local_run_dir):
        config.local_run_dir = os.path.join(current_directory, 'results', folder_name)
        os.makedirs(config.local_run_dir)

    # ---------- save config ---------------
    print(OmegaConf.to_yaml(config))
    config_path = os.path.join(config.local_run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    print('=' * 80)
    print(f'Writing to {config.local_run_dir}')
    print('=' * 80)
 
    os.environ['XDG_CACHE_HOME'] = config.local_dirs
    # print('building policy')
    # model_kwargs = {'device_map': 'balanced'} if config.trainer == 'BasicTrainer' else {}
    # policy_dtype = getattr(torch, config.model.policy_dtype)
    # policy = transformers.AutoModelForCausalLM.from_pretrained(
    #     config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=policy_dtype, **model_kwargs)
    # disable_dropout(policy)

    # if config.loss.name in {'dpo', 'ipo'}:
    #     print('building reference model')
    #     reference_model_dtype = getattr(torch, config.model.reference_dtype)
    #     reference_model = transformers.AutoModelForCausalLM.from_pretrained(
    #         config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=reference_model_dtype, **model_kwargs)
    #     disable_dropout(reference_model)
    # else:
    #     reference_model = None

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    # """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""

    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)
    check_and_modify_config(config)

    # seed everything
    L.seed_everything(config.seed)

    # load model
    print("-------- loading model -----------")
    model = FeatureLevelDPOModel(config=config)

    # wandb settings
    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    # rank = int(os.environ["RANK"])
    # if rank == 0 and config.wandb.enabled:
    #     os.environ['WANDB_CACHE_DIR'] = config.local_dirs
    #     wandb.init(
    #         project="feature-level-dpo",
    #         config=OmegaConf.to_container(config),
    #         dir=config.local_dirs,
    #         name=config.exp_name,
    #     )
    
    # get trainer
    trainer = L.Trainer(
        accelerator="cuda", 
        strategy=FSDPStrategy() if "FSDP" in config.trainer else DDPStrategy(),
        devices=config.devices,
        # num_nodes=args.num_nodes,
        precision=config.precision,
        logger="wandb",
        # callbacks=[train_callback(args)],
        # max_epochs=args.max_epochs,
        # check_val_every_n_epoch=args.check_val_every_n_epoch,
        # num_sanity_val_steps=args.num_sanity_val_steps,
        # log_every_n_steps=args.log_every_n_steps,
        enable_checkpointing=config.activation_checkpointing,
        accumulate_grad_batches=config.gradient_accumulation_steps,
        gradient_clip_val=config.max_grad_norm,
    )

    # get dataloader
    dataloader = None

    # begin training
    trainer.fit(model, dataloader)

    # if config.model.archive is not None:
    #     state_dict = torch.load(config.model.archive, map_location='cpu')
    #     step, metrics = state_dict['step_idx'], state_dict['metrics']
    #     print(f'loading pre-trained weights at step {step} from {config.model.archive} with metrics {json.dumps(metrics, indent=2)}')
    #     policy.load_state_dict(state_dict['state'])
    #     if config.loss.name in {'dpo', 'ipo'}:
    #         reference_model.load_state_dict(state_dict['state'])
    #     print('loaded pre-trained weights')
    
    # if 'FSDP' in config.trainer:
    #     world_size = torch.cuda.device_count()
    #     print('starting', world_size, 'processes for FSDP training')
    #     soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    #     resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    #     print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
    #     mp.spawn(worker_main, nprocs=world_size, args=(world_size, config, policy, reference_model), join=True)
    # else:
    #     print('starting single-process worker')
    #     worker_main(0, 1, config, policy, reference_model)


if __name__ == '__main__':
    main()