from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "examples" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULTS = {
    "image_model": "stabilityai/stable-diffusion-2-1-base",
    "tts_model": "tts_models/en/ljspeech/tacotron2-DDC",
    "music_model": "facebook/musicgen-tiny",
}