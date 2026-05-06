#!/bin/bash
# Auto-run remaining SURDS models and evaluate
# Skips already-completed models
set -e

VQAS_DIR="$HOME/autodl-tmp/spsv/eval_vqas_direct"
SAVE_DIR="inference/vlm_outputs_hf_direct"
EVAL_SAVE="evaluation/eval_result_direct"
INFERENCE_SCRIPT="inference/get_vlm_output_hf.py"
EVAL_SCRIPT="evaluation/eval_from_json.py"

# Check which models already have complete output
is_done() {
    local model_safe="$1"
    local count=$(ls "$SAVE_DIR/$model_safe/"*_output.json 2>/dev/null | wc -l)
    [ "$count" -eq 6 ]
}

run_model() {
    local model_path="$1"
    local model_safe="${model_path//\//--}"

    if is_done "$model_safe"; then
        echo "[SKIP] $model_path (already done)"
    else
        echo "[RUN ] $model_path"
        python "$INFERENCE_SCRIPT" \
            --model_path "$model_path" \
            --vqas_dir "$VQAS_DIR" \
            --save_dir "$SAVE_DIR"
    fi

    # Evaluate
    echo "[EVAL] $model_path"
    python "$EVAL_SCRIPT" \
        --vqas_dir "$VQAS_DIR" \
        --eval_root_dir "$SAVE_DIR" \
        --eval_model_path "$model_safe" \
        --save_dir "$EVAL_SAVE"
}

echo "=== Starting at $(date) ==="

# Priority order: 4B models first (faster), then 8B
for model in \
    "OpenGVLab/InternVL3_5-4B-HF" \
    "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct" \
    "google/gemma-3-4b-it" \
    "allenai/Molmo2-4B" \
    "OpenGVLab/InternVL3-8B-Instruct" \
    "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct" \
    "openbmb/MiniCPM-V-4_5" \
    "allenai/Molmo2-8B" \
    "BytedanceDouyinContent/SAIL-VL2-8B"
do
    run_model "$model"
    echo "---"
done

echo "=== All done at $(date) ==="

# Print all results
echo ""
echo "===== FINAL RESULTS ====="
python -c "
import os, json
eval_dir = '$EVAL_SAVE'
if os.path.exists(eval_dir):
    for d in sorted(os.listdir(eval_dir)):
        csv_path = os.path.join(eval_dir, d, 'eval_result.csv')
        if os.path.exists(csv_path):
            import pandas as pd
            df = pd.read_csv(csv_path)
            print(f'\n--- {d} ---')
            for _, row in df.iterrows():
                print(row.to_dict())
"
