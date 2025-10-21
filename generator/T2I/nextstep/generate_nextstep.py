'''
NextStep-1 Text-to-Image Generation

Model Configuration: NextStep-1-Large
Author: Minki Hong (⚠️ Contact Me: jackyh1@dgu.ac.kr)
'''

import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from models.gen_pipeline import NextStepPipeline
import polars as pl
import warnings, logging
import os
from tqdm import tqdm
import hashlib
import pandas as pd

HF_HUB = "stepfun-ai/NextStep-1-Large"

tokenizer = AutoTokenizer.from_pretrained(HF_HUB, local_files_only=True, trust_remote_code=True)
model = AutoModel.from_pretrained(HF_HUB, local_files_only=True, trust_remote_code=True)
pipeline = NextStepPipeline(tokenizer=tokenizer, model=model).to(device="cuda", dtype=torch.bfloat16)

positive_prompt = "masterpiece, best quality, highly detailed, sharp focus, ultra-detailed, photorealistic, 4k-cinematic"
negative_prompt = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, "
    "signature, watermark, username, blurry"
)

# Image generation prompt csv file
CSV_PATH = 'YOUR_CSV_PATH'
OUTPUT_DIR = 'YOUR_PATH'

def generate_images_from_csv(csv_path, output_dir):
    if not csv_path.is_ifle():
        print(f"❌ Error. Can't find csv file.")
        return
    df = pl.read_csv(csv_path)
    
    for row in tqdm(df.iter_rows(named=True), total=len(df), desc='Generating...'):
        prompt = row.get('prompt')
        relative_path = row.get('image_path')
        if not prompt or not relative_path:
            print(f"\n[Warning] There is no path row.")
            continue
        full_save_path = output_dir / relative_path
        if full_save_path.exists():
            continue
        full_save_path.parent.mkdir(parents=True, exist_ok=True)
        IMG_SIZE=512
        imgs = pipeline.generate_image(
            captions=prompt,
            images=None,
            hw=(IMG_SIZE, IMG_SIZE),
            num_images_per_caption=1,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            cfg=7.5,
            cfg_img=1.0,
            cfg_schedule="constant",
            use_norm=False,
            num_sampling_steps=28,
            timesteps_shift=1.0,
            seed=3407,
        )
        img: Image.Image = imgs[0]
        img.save(full_save_path)
        print(f"✅ Save successed: {full_save_path}")

if __name__ == "__main__":
    generate_images_from_csv(CSV_PATH, OUTPUT_DIR)