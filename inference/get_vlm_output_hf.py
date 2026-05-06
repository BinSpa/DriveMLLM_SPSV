"""
General-purpose HuggingFace inference script for VLM evaluation on SURDS benchmark.
Uses AutoModelForImageTextToText (primary) or AutoModelForVision2Seq (fallback).
"""
import argparse
import json
import logging
import os
import time
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch

os.environ.setdefault('HF_HOME', '/root/autodl-tmp/hf_cache')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

# Monkey-patch for LLaVA models on newer transformers (flash_attn_varlen_func removed)
try:
    from transformers import modeling_flash_attention_utils as _fau
    if not hasattr(_fau, 'flash_attn_varlen_func'):
        _fau.flash_attn_varlen_func = None
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SURDS_TASK_FILES = [
    "00_yaw_vqas.json",
    "01_xy2d_vqas.json",
    "02_depth_vqas.json",
    "03_dis_vqas.json",
    "04_lr_vqas.json",
    "05_fb_vqas.json",
]

LOAD_KWARGS = dict(dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True, local_files_only=True)
# Fall back to auto if model doesn't fit on single GPU
USE_AUTO_DEVICE_MAP = False  # set True for models >40GB


def load_model_and_processor(model_path: str):
    """Load model and processor, trying AutoModelForImageTextToText first, then fallbacks."""
    from transformers import AutoProcessor, AutoTokenizer
    from transformers import AutoModelForImageTextToText, AutoModelForVision2Seq
    from transformers import AutoModel, AutoConfig

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    arch = config.architectures[0] if hasattr(config, 'architectures') and config.architectures else ""
    logger.info(f"Model: {model_path}, Arch: {arch}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

    # Try AutoModelForImageTextToText first (best modern support)
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_path, **LOAD_KWARGS)
        logger.info(f"Loaded via AutoModelForImageTextToText: {type(model).__name__}")
        return model, processor
    except (ValueError, KeyError) as e:
        logger.info(f"AutoModelForImageTextToText failed: {e}")

    # Fall back to AutoModelForVision2Seq
    try:
        model = AutoModelForVision2Seq.from_pretrained(model_path, **LOAD_KWARGS)
        logger.info(f"Loaded via AutoModelForVision2Seq: {type(model).__name__}")
        return model, processor
    except (ValueError, KeyError) as e:
        logger.info(f"AutoModelForVision2Seq failed: {e}")

    # Fall back to AutoModel (for custom architectures like InternVL chat, MiniCPM)
    try:
        model = AutoModel.from_pretrained(model_path, **LOAD_KWARGS)
        logger.info(f"Loaded via AutoModel: {type(model).__name__}")
        return model, processor
    except Exception as e:
        logger.info(f"AutoModel failed: {e}")

    # Last resort: AutoModel with SDPA (bypass flash attention issues)
    try:
        sdpa_kwargs = {**LOAD_KWARGS, "attn_implementation": "sdpa"}
        model = AutoModel.from_pretrained(model_path, **sdpa_kwargs)
        logger.info(f"Loaded via AutoModel+SDPA: {type(model).__name__}")
        return model, processor
    except Exception as e:
        logger.info(f"AutoModel+SDPA failed: {e}")

    raise ValueError(f"Could not load model {model_path}")


def generate_response(model, processor, prompt: str, image: Image.Image,
                      max_new_tokens: int = None) -> str:
    """Generate a response for an image + prompt pair. Uses standard chat template approach."""
    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS if 'MAX_NEW_TOKENS' in dir() else 512

    # Resize image to reduce vision tokens and speed up processing
    image = image.resize((672, 378), Image.LANCZOS)

    # Handle InternVLChatModel with its specific chat() method
    model_type = type(model).__name__
    if model_type == 'InternVLChatModel' and hasattr(model, 'chat'):
        pixel_values = model.visual_encode(image)
        question = f"<image>\n{prompt}"
        response, _ = model.chat(
            tokenizer=processor, pixel_values=pixel_values, question=question,
            generation_config=dict(max_new_tokens=max_new_tokens, do_sample=False)
        )
        return response.strip()

    # Standard chat template approach for all other models
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # Decode only the newly generated tokens
    input_len = inputs.input_ids.shape[1]
    generated_ids = generated_ids[:, input_len:]
    output = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return output.strip()


def run_inference_on_task(model, processor, vqa_file: Path, save_dir: Path, model_name: str):
    """Run inference on all VQA entries in a single task file."""
    with open(vqa_file, 'r') as f:
        vqas = json.load(f)
    logger.info(f"Processing {vqa_file.name}: {len(vqas)} entries")

    save_json_dir = save_dir / model_name
    save_json_dir.mkdir(exist_ok=True, parents=True)
    save_json_file = save_json_dir / f"{vqa_file.stem}_output.json"

    if save_json_file.exists():
        logger.info(f"Skipping {save_json_file} (already exists)")
        return

    vlm_outputs = []
    start_time = time.time()
    errors = 0

    for vqa_idx, vqa in enumerate(tqdm(vqas, desc=f"  {vqa_file.stem}")):
        image_path = Path(vqa['image_path'])
        image = Image.open(str(image_path)).convert("RGB")
        prompt = vqa['prompt']

        try:
            output = generate_response(model, processor, prompt, image)
        except Exception as e:
            errors += 1
            if errors <= 3:
                logger.warning(f"Error at index {vqa_idx}: {e}")
            output = ""

        vlm_outputs.append(dict(
            vqa_idx=vqa_idx,
            image=image_path.name,
            prompt=prompt,
            output=output,
        ))

    with open(str(save_json_file), 'w') as f:
        json.dump(vlm_outputs, f, indent=4)

    elapsed = time.time() - start_time
    logger.info(f"Saved {len(vlm_outputs)} outputs ({errors} errors) to {save_json_file} ({elapsed:.1f}s)")


def main(config):
    model_path = config.model_path
    vqas_dir = Path(config.vqas_dir)
    save_dir = Path(config.save_dir)
    model_name = model_path.replace("/", "--")

    logger.info(f"Loading model: {model_path}")
    model, processor = load_model_and_processor(model_path)

    for task_file_name in SURDS_TASK_FILES:
        vqa_file = vqas_dir / task_file_name
        if not vqa_file.exists():
            logger.warning(f"VQA file not found: {vqa_file}, skipping")
            continue
        run_inference_on_task(model, processor, vqa_file, save_dir, model_name)

    logger.info(f"Done with model: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VLM inference via HuggingFace transformers")
    parser.add_argument('--model_path', type=str, required=True,
                        help='HuggingFace model ID or path')
    parser.add_argument('--vqas_dir', type=str, required=True,
                        help='Directory containing VQA JSON files')
    parser.add_argument('--save_dir', type=str, default='inference/vlm_outputs_hf',
                        help='Directory to save output JSON files')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Maximum tokens to generate (512 default, 2048 for reasoning)')
    args = parser.parse_args()

    # Pass max_tokens to generate_response via a module-level config
    global MAX_NEW_TOKENS
    MAX_NEW_TOKENS = args.max_new_tokens
    main(args)
