
export XDG_CACHE_HOME=.cache
CUDA_VISIBLE_DEVICES=0,1 python3 train.py loss=tdpo2 model=gemma-2-2b datasets='[ultrabin]' exp_name=tdpo-kl-gemma-2-2b-dpo mode=train debug=false model.batch_size=1 model.eval_batch_size=1 model.gradient_accumulation_steps=32 model.use_flash_attention=true lr=5e-7 optimizer=AdamW
# mamba activate qy
# cd hongwei/qy_workspace/emo-dit-qy/emo-fsdp
# export XDG_CACHE_HOME=cache/hf
# CUDA_VISIBLE_DEVICES=5 python3 sample.py \
#     --model_name_or_path google/gemma-2-2b \
#     --checkpoint_path cache/dpo-gemma-2-2b-beta10/LATEST/policy.pt \
#     --dataset_name HuggingFaceH4/ultrafeedback_binarized \
#     --split test_prefs \
#     --batch_size 1 \
#     --max_length 2000 \
#     --output_file samples/sft-06.json \
#     --max_batches 32 \
#     --temperature 1 \
#     --entropy true \
#     # --fm true \

# # CUDA_VISIBLE_DEVICES=7 python3 sample.py \
# #     --model_name_or_path google/gemma-2-2b \
# #     --checkpoint_path /mnt/weka/qy/cache/cache/tdpo-kl-gemma-2-2b-dpo/LATEST/policy.pt \
# #     --dataset_name arena_questions.jsonl \
# #     --split train \
# #     --batch_size 8 \
# #     --max_length 2000 \
# #     --output_file samples/tdpo-kl-06.json \
# #     --max_batches 100 \
# #     --temperature 0.6
    
# # export OPENAI_API_KEY=sk-XsCVDLd3COd5LTGcC89c09393cE444C1A1C8A6Cf2fF1D3B2
# # export OPENAI_BASE_URL=https://api.shubiaobiao.cn/v1
# # alpaca_eval --model_outputs 'samples/tdpo-kl-06.json' --reference_outputs='samples/sft-06.json'
