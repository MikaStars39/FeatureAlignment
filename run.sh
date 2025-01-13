
# export XDG_CACHE_HOME=/mnt/weka/hw_workspace/qy_workspace/lightning/.cache
# CUDA_VISIBLE_DEVICES=7 python patch.py
    # --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    # --release "llama_scope_lxm_8x" \
    # --sae-id "l31m_8x" \
    # --device "cuda"

# cd /mnt/weka/hw_workspace/qy_workspace/lightning
# XDG_CACHE_HOME=/mnt/weka/hw_workspace/qy_workspace/lightning/.cache CUDA_VISIBLE_DEVICES=6 python jailbreak.py \
#     --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
#     --dataset_name_or_path "JailbreakBench/JBB-Behaviors" \
#     --json_path "data/knowledge.json" \
#     --intervene_type "res_attn" \
#     --steering_type "addition" \
#     --token_position 0 \
#     --generate_length 8 \
#     --task_name "knowledge" \
#     --layer_num 28 \
#     --addition_coefficient 1

# cd /mnt/weka/hw_workspace/qy_workspace/lightning
# XDG_CACHE_HOME=/mnt/weka/hw_workspace/qy_workspace/lightning/.cache CUDA_VISIBLE_DEVICES=0 python find_ih.py
# cd /mnt/weka/hw_workspace/qy_workspace/lightning
XDG_CACHE_HOME=/mnt/weka/hw_workspace/qy_workspace/lightning/.cache CUDA_VISIBLE_DEVICES=6 python similarity.py