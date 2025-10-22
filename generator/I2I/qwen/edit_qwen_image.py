'''
Qwen-Image Image-to-Image Editing

Model Configuration: Qwen-Image-Edit
Author: Minki Hong (⚠️ Contact Me: jackyh1@dgu.ac.kr)
'''

import re
from pathlib import Path
from PIL import Image
import torch
from diffusers import QwenImageEditPipeline
from tqdm import tqdm

ROOT = Path("YOUR_ROOT_PATH")
TRUE_CFG_SCALE = 4.0
NEGATIVE_PROMPT = " "
NUM_STEPS = 50
SEED = 0
pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
print("pipeline loaded")
pipeline.to(torch.bfloat16).to("cuda")
pipeline.set_progress_bar_config(disable=None)

num_rounds = 4

def strip_trailing_numbers(stem: str):
    m = re.search(r"_(\d+)$", stem)
    if m:
        n = int(m.group(1))
        base = stem[:m.start()]
        return base, n
    return stem, None

def make_prompt_from_stem(stem: str) -> str:
    base, _ = strip_trailing_numbers(stem)
    if "_" in base:
        rest = base.split("_", 1)[1]
    else:
        rest = base
    rest = rest.replace("_", " ")
    return f'Change the image to represent {rest}.'

for r in range(num_rounds+1):
    src_dir = ROOT / f"edit_ver{r}" # CHANGE YOUR PATH
    dst_dir = ROOT / f'edit_ver{r+1}' # CHANGE YOUR PATH
    dst_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted(src_dir.rglob("*.png"))
    print(f"[Round {r+1}] {src_dir} -> {dst_dir} | {len(png_files)} files")

    for i, img_path in enumerate(tqdm(png_files, desc=f"Round {r+1}")):
        try:
            stem = img_path.stem
            base, n = strip_trailing_numbers(stem)
            curr_n = n if n is not None else 1
            next_n = curr_n + 1

            out_name = f'{base}_{next_n}.png'
            out_path = dst_dir / out_name
            if out_path.exists():
                continue
            
            image = Image.open(img_path).convert("RGB")
            prompt = make_prompt_from_stem(stem)

            inputs = {
                'image': image,
                'prompt': prompt,
                'generator': torch.manual_seed(0),
                "true_cfg_scale": 4.0,
                "negative_prompt": " ",
                "num_inference_steps": 50,
            }
            print(f'Image: {img_path}')
            print(f'Prompt: {prompt}')
            with torch.inference_mode():
                output = pipeline(**inputs)
                output_image = output.images[0]
                output_image.save(out_path)
        except Exception as e:
            print(f"[ERROR][Round {r}] {img_path}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

print("✅ All rounds done: edit_ver0 → edit_ver5")