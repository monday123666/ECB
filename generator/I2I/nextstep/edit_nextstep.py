'''
NextStep-1 Image-to-Image Editing

Model Configuration: NextStep-1-Large-Edit
Author: Minki Hong (⚠️ Contact Me: jackyh1@dgu.ac.kr)
'''

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
import re
from transformers import AutoTokenizer, AutoModel
from models.gen_pipeline import NextStepPipeline
from utils.aspect_ratio import center_crop_arr_with_buckets

ROOT_DIR = Path('YOUR_ROOT_PATH')
NUM_ROUNDS = 4

# NextStep model and pipeline configuration
HF_HUB = "stepfun-ai/NextStep-1-Large-Edit"
IMG_SIZE = 512
NEGATIVE_PROMPT = "Copy original image, blur, blurry, ugly, duplicate"
NUM_SAMPLING_STEPS = 50
CFG = 7.5
CFG_IMG = 2.0
SEED = 42

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(HF_HUB, local_files_only=True, trust_remote_code=True)
model = AutoModel.from_pretrained(HF_HUB, local_files_only=True, trust_remote_code=True)
pipeline = NextStepPipeline(tokenizer=tokenizer, model=model).to(device=f"cuda")

def parse_filename(stem: str):
    match = re.search(r"(.+?)_(\d+)$", stem)
    if match:
        base_name = match.group(1)
        number = int(match.group(2))
        return base_name, number
    return stem, 0

def make_nextstep_prompt(base_name: str) -> str:
    if "_" in base_name:
        rest = base_name.split("_", 1)[1]
    else:
        rest = base_name
    rest = rest.replace("_", " ")
    return f'<image> Change the image to represent {rest}'

for r in range(5):
    current_round = r
    src_dir = ROOT_DIR / f"edit_ver{current_round}" # CHANGE YOUR PATH

    dst_dir = ROOT_DIR / f"edit_ver{current_round+1}" # CHANGE YOUR PATH
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not src_dir.is_dir():
        print(f"[Warning][Round {current_round}] can't find: {src_dir}. Skip this round")
        continue
    
    png_files = sorted(src_dir.rglob("*.png"))
    print(f"\n--- [Round {current_round}] ---")
    print(f"Input: '{src_dir.name}', Output: '{dst_dir.name}' | File: {len(png_files)}개")

    for img_path in tqdm(png_files, desc=f"Round {current_round+1} Editing..."):
        try:
            base_name, curr_n = parse_filename(img_path.stem)
            next_n = curr_n + 1
            out_name = f'{base_name}_{next_n}.png'
            out_path = dst_dir / out_name

            if out_path.exists():
                continue
            
            prompt = make_nextstep_prompt(base_name)

            image_pil = Image.open(img_path).convert("RGB")
            image_processed = center_crop_arr_with_buckets(image_pil, buckets=[IMG_SIZE])

            print(f'Round: {next_n}\nPrompt: {prompt}')

            with torch.inference_mode():
                output_image = pipeline.generate_image(
                    prompt,
                    images=[image_processed],
                    hw=(IMG_SIZE, IMG_SIZE),
                    num_images_per_caption=1,
                    positive_prompt=None,
                    negative_prompt=NEGATIVE_PROMPT,
                    cfg=CFG,
                    cfg_img=CFG_IMG,
                    cfg_schedule="constant",
                    use_norm=True,
                    num_sampling_steps=NUM_SAMPLING_STEPS,
                    timesteps_shift=3.2,
                    seed=SEED,
                )[0]

            output_image.save(out_path)

        except Exception as e:
            print(f"[Error][Round {current_round}] {img_path.name}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

print(f"\n✅ Editing ends")