"""
Batched HuggingFace inference for VLM evaluation on SURDS benchmark.
Processes VQAs in batches for 4-8x speedup vs single-item processing.
"""
import argparse, json, logging, os, time
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch

os.environ.setdefault('HF_HOME', '/root/autodl-tmp/hf_cache')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

# Monkey-patch for LLaVA models on newer transformers
try:
    from transformers import modeling_flash_attention_utils as _fau
    if not hasattr(_fau, 'flash_attn_varlen_func'):
        _fau.flash_attn_varlen_func = None
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SURDS_TASK_FILES = [
    "00_yaw_vqas.json", "01_xy2d_vqas.json", "02_depth_vqas.json",
    "03_dis_vqas.json", "04_lr_vqas.json", "05_fb_vqas.json",
]

LOAD_KWARGS = dict(dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True, local_files_only=True)


def load_model_and_processor(model_path: str):
    """Load model and processor with fallbacks."""
    from transformers import AutoProcessor, AutoTokenizer, AutoModel
    from transformers import AutoModelForImageTextToText, AutoModelForVision2Seq, AutoConfig

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    arch = config.architectures[0] if hasattr(config, 'architectures') and config.architectures else ""
    logger.info(f"Model: {model_path}, Arch: {arch}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

    for loader_name, loader in [("ImageTextToText", AutoModelForImageTextToText),
                                 ("Vision2Seq", AutoModelForVision2Seq)]:
        try:
            model = loader.from_pretrained(model_path, **LOAD_KWARGS)
            logger.info(f"Loaded via {loader_name}: {type(model).__name__}")
            return model, processor
        except (ValueError, KeyError) as e:
            logger.info(f"{loader_name} failed: {e}")

    for extra_kwargs in [{}, {"attn_implementation": "sdpa"}]:
        try:
            kw = {**LOAD_KWARGS, **extra_kwargs}
            model = AutoModel.from_pretrained(model_path, **kw)
            logger.info(f"Loaded via AutoModel{'+SDPA' if extra_kwargs else ''}: {type(model).__name__}")
            return model, processor
        except Exception as e:
            logger.info(f"AutoModel{' +SDPA' if extra_kwargs else ''} failed: {e}")

    raise ValueError(f"Could not load {model_path}")


def generate_batch(model, processor, batch_items: list, model_type: str, max_new_tokens: int) -> list:
    """Generate responses for a batch of items. Each item is (prompt, image)."""
    prompts = [item[0] for item in batch_items]
    images = [item[1] for item in batch_items]

    # --- Model-specific batch handling ---
    if model_type == 'MiniCPMV' and hasattr(model, 'chat'):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path,
            trust_remote_code=True, local_files_only=True)
        results = []
        for i in range(len(batch_items)):
            msgs = json.dumps([{"role": "user", "content": prompts[i]}])
            try:
                r = model.chat(image=images[i], msgs=msgs, tokenizer=tokenizer,
                              sampling=False, max_new_tokens=max_new_tokens)
                results.append(r.strip())
            except Exception as e:
                logger.warning(f"MiniCPM error at batch: {e}")
                results.append("")
        return results

    if model_type in ('InternVLChatModel', 'SAILVLModel') and hasattr(model, 'chat'):
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        results = []
        for i in range(len(batch_items)):
            try:
                img_tensor = preprocess(images[i]).unsqueeze(0).to(model.device).to(torch.bfloat16)
                question = f"<image>\n{prompts[i]}"
                r = model.chat(tokenizer=processor, pixel_values=img_tensor, question=question,
                               generation_config=dict(max_new_tokens=max_new_tokens, do_sample=False))
                # chat returns str directly (not tuple) for InternVL
                results.append(r.strip() if isinstance(r, str) else str(r).strip())
            except Exception as e:
                logger.warning(f"{model_type} error at batch: {e}")
                results.append("")
        return results

    # --- Standard batch processing for all other models ---
    messages_list = []
    for prompt, image in zip(prompts, images):
        messages_list.append([{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}])

    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # Decode only new tokens
    input_lens = inputs.input_ids.shape[1]
    outputs = []
    for i in range(generated_ids.shape[0]):
        gen = generated_ids[i, input_lens:]
        text = processor.decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        outputs.append(text.strip())
    return outputs


def run_inference_on_task(model, processor, vqa_file: Path, save_dir: Path, model_name: str,
                          batch_size: int, max_new_tokens: int):
    """Run batched inference on all VQA entries in a task file."""
    with open(vqa_file, 'r') as f:
        vqas = json.load(f)
    logger.info(f"Processing {vqa_file.name}: {len(vqas)} entries (batch_size={batch_size})")

    save_json_dir = save_dir / model_name
    save_json_dir.mkdir(exist_ok=True, parents=True)
    save_json_file = save_json_dir / f"{vqa_file.stem}_output.json"

    if save_json_file.exists():
        logger.info(f"Skipping {save_json_file} (already exists)")
        return

    model_type = type(model).__name__

    # Pre-load and resize all images
    logger.info("  Pre-loading images...")
    prepped = []
    for vqa in tqdm(vqas, desc="  Load images"):
        img = Image.open(vqa['image_path']).convert('RGB').resize((672, 378), Image.LANCZOS)
        prepped.append((vqa['prompt'], img, vqa['image_path']))

    # Batch process
    vlm_outputs = []
    errors = 0
    start_time = time.time()

    for batch_start in tqdm(range(0, len(prepped), batch_size), desc=f"  {vqa_file.stem}"):
        batch = prepped[batch_start:batch_start + batch_size]
        batch_items = [(b[0], b[1]) for b in batch]

        try:
            outputs = generate_batch(model, processor, batch_items, model_type, max_new_tokens)
        except Exception as e:
            logger.warning(f"Batch error at {batch_start}: {e}")
            outputs = [""] * len(batch)

        for (_, _, img_path), output in zip(batch, outputs):
            vlm_outputs.append(dict(
                vqa_idx=len(vlm_outputs),
                image=Path(img_path).name,
                prompt=batch[0][0],  # will be overwritten per-item
                output=output,
            ))

        if not outputs or all(not o for o in outputs):
            errors += len(batch)

    # Fix vqa_idx and prompt (they got scrambled by batching)
    for i, vqa in enumerate(vqas):
        if i < len(vlm_outputs):
            vlm_outputs[i]['vqa_idx'] = i
            vlm_outputs[i]['prompt'] = vqa['prompt']

    with open(str(save_json_file), 'w') as f:
        json.dump(vlm_outputs, f, indent=4)

    elapsed = time.time() - start_time
    logger.info(f"Saved {len(vlm_outputs)} outputs ({errors} errors) to {save_json_file} ({elapsed:.1f}s, {len(vqas)/elapsed:.1f} it/s)")


def main(config):
    model_path = config.model_path
    vqas_dir = Path(config.vqas_dir)
    save_dir = Path(config.save_dir)
    batch_size = config.batch_size
    max_new_tokens = config.max_new_tokens
    model_name = model_path.replace("/", "--")

    logger.info(f"Loading model: {model_path}")
    model, processor = load_model_and_processor(model_path)

    for task_file_name in SURDS_TASK_FILES:
        vqa_file = vqas_dir / task_file_name
        if not vqa_file.exists():
            continue
        run_inference_on_task(model, processor, vqa_file, save_dir, model_name, batch_size, max_new_tokens)

    logger.info(f"Done: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batched VLM inference for SURDS")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--vqas_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='inference/vlm_outputs_hf')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    args = parser.parse_args()
    main(args)
