
export XDG_CACHE_HOME=/mnt/weka/qy/cache/hf
CUDA_VISIBLE_DEVICES=5 python3 train.py loss=tdpo-kl model=gemma-2-2b datasets='[ultrabin]' exp_name=tdpo-kl mode=train debug=true model.batch_size=1 model.eval_batch_size=1 model.gradient_accumulation_steps=32 model.use_flash_attention=true lr=5e-7 optimizer=AdamW ++model.load_from=/mnt/weka/qy/cache/cache/sft-gemma-2-2b/LATEST/policy.pt
# mamba activate qy
# cd hongwei/qy_workspace/emo-dit-qy/emo-fsdp