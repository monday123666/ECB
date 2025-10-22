'''
Qwen-Image Text-to-Image Generation

Model Configuration: Qwen-Image
Author: Minki Hong (⚠️ Contact Me: jackyh1@dgu.ac.kr)
'''

from diffusers import DiffusionPipeline
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import os

model_name = "Qwen/Qwen-Image"

# Load the pipeline
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
pipe = pipe.to(device)

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
}
negative_prompt = " "

# Image generation prompt csv file
try:
    df = pd.read_csv('YOUR CSV FILE')
    print('✅ Load CSV file')
except FileNotFoundError:
    print('❌ Error')
    exit()

# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

width, height = aspect_ratios["16:9"]

out_dir = './qwen_image_outputs'
os.makedirs(out_dir, exist_ok=True)

def generate_and_save(caption: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = pipe(
        prompt=caption + positive_magic['en'],
        negative_prompt=negative_prompt,
        width=1024,
        height=1024,
        num_inference_steps=50,
        true_cfg_scale=1.0,
        generator=torch.Generator(device="cuda").manual_seed(42)
    ).images[0]
    image.save(f'{out_path}.jpg')

current_category = ""
current_subcategory = ""

for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Generating images from CSV'):
    if pd.notna(row['Category']):
        current_category = row['Category']
    if pd.notna(row['Subcategory']):
        current_subcategory = row['Subcategory']

    country = row['Country']
    
    if pd.isna(country):
        continue
    
    country_str = str(country).replace(" ", "_").lower()
    category_str = str(current_category).replace(" ", "_").lower()
    subcategory_str = str(current_subcategory).replace(" ", "_").lower() 
    
    for prompt_type in ['Traditional Prompt', 'Modern Prompt', 'Generic Prompt']:
        prompt_text = row[prompt_type]
        
        if isinstance(prompt_text, str) and prompt_text.strip():
            variant_name = prompt_type.replace(" Prompt", "").lower()            
            out_path = Path(out_dir) / f"qwenimage_{country_str}_{category_str}_{subcategory_str}_{variant_name}"            
            generate_and_save(prompt_text, out_path)