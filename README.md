# SURDS: Benchmarking Spatial Understanding and Reasoning in Driving Scenarios with Vision Language Models

## Model Evaluation Results (from this repo)

We evaluated 6 open-source VLMs on the SURDS benchmark using direct (non-reasoning) prompts on 2× NVIDIA A800 80GB GPUs. The original paper uses reasoning prompts with `<think>` chain-of-thought; our direct-prompt approach is ~100× faster but yields lower scores than the paper's reported results.

### Results (Direct Prompts, No CoT Reasoning)

| # | Model | Size | Yaw | XY2D | Depth | Dis | LR | FB | **Avg** |
|---|-------|------|-----|------|-------|-----|----|----|---------|
| - | Random Baseline | - | 0.068 | 0.015 | 0.343 | 0.083 | 0.106 | 0.110 | **0.121** |
| 1 | Qwen3-VL-4B-Instruct | 4B | 0.024 | 0.041 | 0.442 | 0.292 | 0.066 | 0.059 | **0.154** |
| 2 | Gemma-3-4B-it | 4B | 0.038 | 0.007 | 0.538 | 0.416 | 0.175 | 0.041 | **0.203** |
| 3 | LLaVA-OneVision-1.5-4B | 4B | 0.019 | 0.000 | 0.377 | 0.476 | 0.603 | 0.000 | **0.246** |
| 4 | Molmo2-4B | 4B | 0.077 | 0.009 | 0.324 | 0.530 | 0.514 | **0.158** | **0.269** |
| 5 | InternVL3_5-4B-HF | 4B | 0.042 | 0.027 | 0.365 | 0.590 | **0.765** | 0.102 | **0.315** |
| 6 | **Qwen3-VL-8B-Instruct** | 8B | **0.046** | **0.060** | **0.558** | **0.764** | 0.710 | 0.041 | **0.363** |

*Metrics: Yaw (orientation), XY2D (2D localization), Depth (distance), Dis (closer/farther), LR (left/right), FB (front/back). Score = accuracy over all ground-truth questions (unparseable = wrong).*

### Key Findings

- **Qwen3-VL-8B leads** at 0.363 avg (3× random baseline of 0.121), winning on 4/6 tasks
- **InternVL3_5-4B is best 4B model** at 0.315; remarkable LR score of 0.765
- **FB (front/back) is hardest** — only Molmo2-4B (0.158) beats random
- **Output format compliance** critical — 8B models achieve 98%+ valid response rate vs 73% for 4B
- **Scale matters** — 8B models significantly outperform 4B (0.363 vs 0.154–0.315)

### Failed / Untested Models

| Model | Issue |
|-------|-------|
| InternVL3-8B-Instruct | `InternVLChatModel` uses different API (no `visual_encode`); needs custom inference |
| LLaVA-OneVision-1.5-8B | Flash attention varlen import error with transformers 4.57 |
| MiniCPM-V-4_5 | Processor has no chat template; requires custom inference code |
| SAIL-VL2-8B | Loads as `SAILVLModel` via AutoModel; processor lacks vision support (`Qwen2TokenizerFast`) |
| Molmo2-8B | Running in background (est. ~2 hrs); results will appear in `evaluation/eval_result_direct/` |

### Running Evaluation with This Repo

#### 1. Generate VQAs
```shell
# Reasoning prompts (paper-style, slow ~96s/VQA)
python hfdata_to_eval_vqa.py \
  --hf_dataset bonbon-rj/SURDS_eval \
  --prompt_dir prompt/prompts_reasoning \
  --vqas_save_dir eval_vqas_reasoning

# Direct prompts (fast ~0.5s/VQA, no CoT)
python hfdata_to_eval_vqa.py \
  --hf_dataset bonbon-rj/SURDS_eval \
  --prompt_dir prompt/prompts_direct \
  --vqas_save_dir eval_vqas_direct
```

#### 2. Run Inference (HF Transformers)
```shell
# General-purpose HF inference (works for most VLMs)
python inference/get_vlm_output_hf.py \
  --model_path <MODEL_ID> \
  --vqas_dir eval_vqas_direct \
  --save_dir inference/vlm_outputs_hf_direct

# Random baseline
python inference/get_vlm_output_random.py \
  --save_dir inference/vlm_outputs_hf_direct \
  --vqas_dir eval_vqas_direct
```

#### 3. Evaluate
```shell
python evaluation/eval_from_json.py \
  --vqas_dir eval_vqas_direct \
  --eval_root_dir inference/vlm_outputs_hf_direct \
  --eval_model_path <MODEL_SAFE_NAME> \
  --save_dir evaluation/eval_result_direct
```

#### Run All Models Sequentially
```shell
bash inference/run_remaining_models.sh
```

### New Files in This Repo

| File | Purpose |
|------|---------|
| `inference/get_vlm_output_hf.py` | General VLM inference via HF transformers (supports Qwen3-VL, InternVL3/3.5, LLaVA-OV, Molmo2, Gemma3, MiniCPM-V, SAIL-VL2) |
| `inference/run_all_models.sh` | Auto-run all available models sequentially |
| `inference/run_remaining_models.sh` | Auto-run only not-yet-completed models |
| `prompt/prompts_direct/` | Direct prompt templates (no reasoning, ~100× faster) |
| `evaluation/eval_result_direct/` | Per-model evaluation CSV files |

---

## Original README (Upstream)

## Update
We have changed the title from "DriveMLLM: A Benchmark for Spatial Understanding with Multimodal Large Language Models in Autonomous Driving" to "SURDS: Benchmarking Spatial Understanding and Reasoning in Driving Scenarios with Vision Language Models". If you use the data from the first version of DriveMLLM, you can use the v1 branch.

## Dataset

We extracted and processed data from the [nuScenes](https://www.nuscenes.org/) dataset to create our own [SURDS](https://huggingface.co/datasets/bonbon-rj/SURDS) dataset for training and evaluation purposes.  Due to the large size of the training data, we also provide a separate evaluation-only version: [SURDS_eval](https://huggingface.co/datasets/bonbon-rj/SURDS_eval).  A `metadata.jsonl` file is included for all images, allowing users to conveniently access properties such as `xy2Ds`.



## Getting Started

### Environment Setup

To get started, follow the steps below to set up the environment:

```shell
# Clone the repository and add it to PYTHONPATH
git clone https://github.com/XiandaGuo/Drive-MLLM.git
cd Drive-MLLM
echo 'export PYTHONPATH=$(pwd):$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc

# Create a Conda environment and install core dependencies
conda create -n surds python=3.10 
source activate surds
pip install -r requirements.txt

# Set up the Qwen2-VL environment
git clone https://github.com/QwenLM/Qwen2-VL.git
cd Qwen2-VL
pip install -r requirements_web_demo.txt
pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830 accelerate
pip install qwen-vl-utils[decord]
pip install flash-attn --no-build-isolation --no-cache-dir  # (Recommended) 
pip install transformers==4.50.0 # Stable version for this project
cd ..

# Install SGLang with acceleration support
pip install --upgrade pip
pip install uv
uv pip install "sglang[all]==0.4.4.post4" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python # Different versions of SGLang may adopt varying acceleration strategies
```



**Reference Links**:

- [Qwen2-VL Official Github Website](https://github.com/QwenLM/Qwen2-VL)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [SGLang installation](https://docs.sglang.ai/start/install.html)



### VQAs Generation

To generate Visual Question-Answering (VQA) examples for evaluation, run the script below. It downloads the dataset from Hugging Face, applies the prompts provided in `<prompt_dir>`, and stores the generated VQAs in the `<vqas_save_dir>` directory.

```shell
python hfdata_to_eval_vqa.py \
--hf_dataset bonbon-rj/SURDS_eval \
--prompt_dir prompt/prompts_reasoning \
--vqas_save_dir eval_vqas_reasoning
```



### Inference

#### Running Inference with SGLang

To perform inference on the `vqas_dir` prompts using [SGLang](https://github.com/sgl-project/sglang), execute the script below. 

The example below demonstrates inference with the `Qwen/Qwen2.5-VL-3B-Instruct` model on 8 × 80 GB GPUs:

```shell
python inference/get_vlm_output_sglang.py \
--save_dir inference/vlm_outputs \
--save_sub_dir qwen \
--vqas_dir eval_vqas_reasoning \
--bs_per_req 1850 \
--sglang_model "Qwen/Qwen2.5-VL-3B-Instruct" \
--sglang_tpl qwen2-vl \
--sglang_dtype bfloat16 \
--sglang_mem 0.9 \
--sglang_maxreq 64 \
--sglang_dp 8 \
--sglang_tp 1
```

The results will be saved to the directory: `<save_dir>/<save_sub_dir>/<sglang_model>`.



#### Generating Random Outputs

To obtain random outputs on the `<vqas_dir>` prompts, run:

```shell
python inference/get_vlm_output_random.py \
--save_dir inference/vlm_outputs \
--vqas_dir eval_vqas_reasoning 
```

The results will be saved to the directory: `<save_dir>/random/random`.



#### Adapting Unsupported Models

If your target model is not yet supported by SGLang, you can use `get_vlm_output_random.py` as a template and replace the `generate_random_output` function with your model’s inference implementation.



### Evaluation

To evaluate all model outputs stored in `<eval_root_dir>`, you can run the following script:

```shell
python evaluation/eval_from_json.py \
--vqas_dir eval_vqas_reasoning \
--eval_root_dir inference/vlm_outputs \
--eval_model_path all \
--save_dir evaluation/eval_result 
```

Alternatively, to evaluate a specific model's output under `<eval_root_dir>`, specify the desired `<eval_model_path>`:

```shell
python evaluation/eval_from_json.py \
--vqas_dir eval_vqas_reasoning \
--eval_root_dir inference/vlm_outputs \
--eval_model_path qwen/Qwen2.5-VL-3B-Instruct \
--save_dir evaluation/eval_result 
```

After running the scripts, the evaluation results will be stored in the directory: `<save_dir>`.



### Training

We employ [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for supervised fine-tuning (SFT), and adopt the [VLM-R1](https://github.com/om-ai-lab/VLM-R1) framework to train the model using Group Relative Policy Optimization (GRPO).

**Note**: This code is only used for academic purposes; people cannot use this code for anything that might be considered commercial use.
To prepare SFT data with chain-of-thought (CoT) reasoning, use the provided scripts: `summarize_rules.py` and `gen_cot.py`.

For reinforcement learning, the GRPO implementation is available in `grpo.py`.

## Citation
```
@inproceedings{guo2025surds,
  title={SURDS: Benchmarking Spatial Understanding and Reasoning in Driving Scenarios with Vision Language Models},
  author={Guo, Xianda and Zhang, Ruijun and Duan, Yiqun and He, Yuhang and Nie, Dujun and Huang, Wenke and Zhang, Chenming and Liu, Shuai and Zhao, Hao and Chen, Long},
  booktitle={NeurIPS},
  year={2025}
}
```


