
# export XDG_CACHE_HOME=.cache
# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 train.py loss=tdpo-kl model=gemma-2-2b datasets='[ultrabin]' exp_name=tdpo-kl-gemma-2-2b-dpo mode=train debug=false model.batch_size=1 model.eval_batch_size=1 model.gradient_accumulation_steps=32 model.use_flash_attention=true lr=5e-7 optimizer=AdamW ++model.load_from=sft-gemma-2-2b/LATEST/policy.pt
# mamba activate qy
# cd hongwei/qy_workspace/emo-dit-qy/emo-fsdp
export XDG_CACHE_HOME=/mnt/weka/qy/cache/hf
CUDA_VISIBLE_DEVICES=7 python3 sample.py \
    --model_name_or_path google/gemma-2-2b \
    --checkpoint_path /mnt/weka/qy/cache/sft-gemma-2-2b/LATEST/policy.pt \
    --dataset_name arena_questions.jsonl \
    --split train \
    --batch_size 1 \
    --max_length 2000 \
    --output_file sft.json \
    --max_batches 500
export OPENAI_API_KEY=sk-XsCVDLd3COd5LTGcC89c09393cE444C1A1C8A6Cf2fF1D3B2
export OPENAI_BASE_URL=https://api.shubiaobiao.cn/v1
alpaca_eval --model_outputs 'alpaca_eval_results.json' --reference_outputs='sft.json'
