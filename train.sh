
export XDG_CACHE_HOME=.cache
CUDA_VISIBLE_DEVICES=0,6 python3 train.py loss=tdpo-kl model=gemma-2-2b datasets='[ultrabin]' exp_name=tdpo-kl-gemma-2-2b mode=train debug=true model.batch_size=1 model.eval_batch_size=1 model.gradient_accumulation_steps=8 model.use_flash_attention=true lr=5e-7 optimizer=AdamW 
# ++model.load_from=sft-gemma-2-2b/LATEST/policy.pt
