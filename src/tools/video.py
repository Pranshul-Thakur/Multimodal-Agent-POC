from pathlib import Path
from pydub import AudioSegment
from moviepy.editor import ImageClip, AudioFileClip
from src.utils.config import OUTPUT_DIR

def _ensure_audio_segment(path):
    return AudioSegment.from_file(str(path))

def mix_audio(speech_wav, music_wav, out_name="mixed.wav"):
    out_path = OUTPUT_DIR / out_name
    try:
        speech = _ensure_audio_segment(speech_wav)
        music = _ensure_audio_segment(music_wav)

        if len(music) < len(speech):
            music = music * (len(speech) // len(music) + 1)

        music = music - 10
        mixed = speech.overlay(music)
        mixed.export(str(out_path), format="wav")
        return str(out_path)
    except Exception as e:
        print(f"Audio mix failed: {e}")
        return speech_wav

def assemble_video(image_path, audio_path, duration=8, out_name="video.mp4"):
    out_path = OUTPUT_DIR / out_name
    clip = ImageClip(str(image_path)).set_duration(duration)
    audio = AudioFileClip(str(audio_path))
    clip = clip.set_audio(audio)
    clip.write_videofile(str(out_path), fps=24)
    return str(out_path)