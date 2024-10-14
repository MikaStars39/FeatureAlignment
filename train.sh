
export XDG_CACHE_HOME=cache/hf

WANDB_API_KEY="0347e3e57dd05c7b2dc58c0b5625ec8c98de7f4d" CUDA_VISIBLE_DEVICES=0,7 python3 train.py loss=tdpo-kl model=gemma-2-9b datasets='[ultrabin]' exp_name=fpo-gemma-2-9b mode=train debug=false model.batch_size=2 model.eval_batch_size=2 model.gradient_accumulation_steps=16 model.use_flash_attention=true lr=5e-7 ++model.load_from=sft-gemma-2-9b/LATEST/policy.pt
# mamba activate qy
# cd hongwei/qy_workspace/emo-dit-qy/emo-fsdp
