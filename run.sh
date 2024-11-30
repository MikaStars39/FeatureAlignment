
export XDG_CACHE_HOME=/mnt/weka/hw_workspace/qy_workspace/lightning/.cache
CUDA_VISIBLE_DEVICES=1 python harmful_test.py
    # --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    # --release "llama_scope_lxm_8x" \
    # --sae-id "l31m_8x" \
    # --device "cuda"