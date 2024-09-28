
export XDG_CACHE_HOME=cache/hf
CUDA_VISIBLE_DEVICES=4 python3 train.py loss=sft model=gemma-2-9b datasets='[ultrabin]' exp_name=sft-gemma-2-9b mode=train debug=true model.batch_size=1 model.eval_batch_size=1 model.gradient_accumulation_steps=32 model.use_flash_attention=true lr=5e-7
#++model.load_from=sft-gemma-2-2b/LATEST/policy.pt
# mamba activate qy
# cd hongwei/qy_workspace/emo-dit-qy/emo-fsdp