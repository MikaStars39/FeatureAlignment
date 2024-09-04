export XDG_CACHE_HOME=.cache

CUDA_VISIBLE_DEVICES=2,3 python3 train.py loss=tdpo-kl model=gemma-2-2b datasets='[ultrabin]' exp_name=tdpo2-gemma-2-2b mode=train debug=false model.batch_size=8 model.eval_batch_size=8 model.gradient_accumulation_steps=1 model.use_flash_attention=true lr=5e-7 optimizer=AdamW ++model.load_from=sft-gemma-2-2b/LATEST/policy.pt
