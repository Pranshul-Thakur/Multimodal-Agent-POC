import torch
import soundfile as sf
from pathlib import Path
import shutil
import numpy as np
from src.utils.config import DEFAULTS, OUTPUT_DIR

def generate_music(prompt: str, out_name="music.wav", duration=8, model_name=None):
    out_path = OUTPUT_DIR / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_name = model_name or DEFAULTS.get("music_model")

    try:
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_name)
        model = MusicgenForConditionalGeneration.from_pretrained(model_name)
        inputs = processor(text=[prompt], padding=True, return_tensors="pt")
        audio_values = model.generate(**inputs, max_new_tokens=256)
        audio_arr = audio_values[0, 0].cpu().numpy()
        sr = getattr(model.config, "sampling_rate", getattr(model.config, "audio_encoder", {}).get("sampling_rate", 32000))
        sf.write(str(out_path), audio_arr, sr)
        return str(out_path)
    except Exception as e:
        placeholder = Path("examples/assets/placeholder_music.wav")
        if placeholder.exists():
            try:
                shutil.copy(str(placeholder), str(out_path))
                return str(out_path)
            except Exception:
                pass
        sr = 22050
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        freq = 220.0
        audio = 0.1 * np.sin(2 * np.pi * freq * t)
        sf.write(str(out_path), audio, sr)
        return str(out_path)
