from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
from src.utils.config import DEFAULTS, OUTPUT_DIR

pipe = StableDiffusionPipeline.from_pretrained(
    DEFAULTS["image_model"],
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
if torch.cuda.is_available():
    pipe.to("cuda")

def generate_image(prompt: str, out_name="image.png"):
    image = pipe(prompt).images[0]
    out_path = OUTPUT_DIR / out_name
    image.save(out_path)
    return str(out_path)