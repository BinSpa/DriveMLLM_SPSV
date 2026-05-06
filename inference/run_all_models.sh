#!/bin/bash
# Driver script to run inference sequentially for all SURDS benchmark models
# Each model runs 6 task files. Results saved to inference/vlm_outputs_hf/

set -euo pipefail

VQAS_DIR="$HOME/autodl-tmp/spsv/eval_vqas_reasoning"
SAVE_DIR="inference/vlm_outputs_hf"
SCRIPT="inference/get_vlm_output_hf.py"

# Models in priority order (4B first for speed, then 8B)
MODELS=(
    # 4B models
    "Qwen/Qwen3-VL-4B-Instruct"
    "OpenGVLab/InternVL3_5-4B-HF"
    "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"
    "google/gemma-3-4b-it"
    "allenai/Molmo2-4B"
    # 8B models
    "Qwen/Qwen3-VL-8B-Instruct"
    "OpenGVLab/InternVL3-8B-Instruct"
    "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
    "allenai/Molmo2-8B"
    "openbmb/MiniCPM-V-4_5"
    "BytedanceDouyinContent/SAIL-VL2-8B"
)

LOG_DIR="inference/logs"
mkdir -p "$LOG_DIR"

for model in "${MODELS[@]}"; do
    echo "============================================"
    echo "[$(date)] Starting model: $model"
    echo "============================================"

    short_name=$(echo "$model" | tr '/' '_')
    log_file="$LOG_DIR/${short_name}.log"

    if python "$SCRIPT" \
        --model_path "$model" \
        --vqas_dir "$VQAS_DIR" \
        --save_dir "$SAVE_DIR" 2>&1 | tee "$log_file"; then
        echo "[$(date)] SUCCESS: $model"
    else
        echo "[$(date)] FAILED: $model (see $log_file)"
    fi

    echo ""
done

echo "[$(date)] All models done!"
