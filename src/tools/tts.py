from pathlib import Path
from src.utils.config import DEFAULTS, OUTPUT_DIR

try:
    from TTS.api import TTS
    HAS_TTS = True
except Exception:
    HAS_TTS = False

def synthesize_speech(text: str, out_name="speech.wav", model_name=None):
    out_path = OUTPUT_DIR / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_name = model_name or DEFAULTS["tts_model"]

    if not HAS_TTS:
        import wave
        with wave.open(str(out_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(b"\x00\x00" * 22050)
        return str(out_path)

    tts = TTS(model_name)
    tts.tts_to_file(text=text, file_path=str(out_path))
    return str(out_path)