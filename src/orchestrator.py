import argparse
from datetime import datetime
from src.utils.logger import log_info, write_manifest
from src.utils.config import DEFAULTS
from src.tools.image_gen import generate_image
from src.tools.tts import synthesize_speech
from src.tools.music_gen import generate_music
from src.tools.video import mix_audio, assemble_video

def plan_tasks(user_prompt: str):
    return {
        "user_prompt": user_prompt,
        "image_prompt": f"{user_prompt} -- cinematic, 4k, high detail",
        "speech_text": user_prompt,
        "music_prompt": f"ambient background for: {user_prompt}",
        "video_length": 8,
    }

def run_pipeline(user_prompt: str):
    log_info("Planning...")
    plan = plan_tasks(user_prompt)

    log_info("Generating image...")
    image_path = generate_image(plan["image_prompt"], out_name="sample_image.png")

    log_info("Synthesizing speech...")
    speech_path = synthesize_speech(plan["speech_text"], out_name="sample_speech.wav")

    log_info("Generating music...")
    music_path = generate_music(plan["music_prompt"], out_name="sample_music.wav")

    log_info("Mixing audio...")
    final_audio = mix_audio(speech_path, music_path, out_name="final_audio.wav")

    log_info("Assembling video...")
    video_path = assemble_video(
        image_path,
        final_audio,
        duration=plan["video_length"],
        out_name="final_video.mp4",
    )

    manifest = {
        "prompt": user_prompt,
        "plan": plan,
        "artifacts": {
            "image": str(image_path),
            "speech": str(speech_path),
            "music": str(music_path),
            "audio": str(final_audio),
            "video": str(video_path),
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    write_manifest(manifest)
    return manifest

def run_graph_mode(user_prompt: str):
    from src.agent.run_graph import load_graph_and_run
    return load_graph_and_run(user_prompt)

def set_fast_defaults():
    DEFAULTS.update({
        "image_model": "stabilityai/stable-diffusion-2-1-base",
        "tts_model": "tts_models/en/ljspeech/tacotron2-DDC",
        "music_model": "facebook/musicgen-tiny",
    })

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("linear", "graph", "fast"), default="linear")
    p.add_argument("--prompt", type=str, default="A calm blue mountain landscape at sunset with birds flying")
    args = p.parse_args()

    if args.mode == "fast":
        set_fast_defaults()
        m = run_pipeline(args.prompt)
        print(m)
    elif args.mode == "graph":
        out = run_graph_mode(args.prompt)
        print(out)
    else:
        m = run_pipeline(args.prompt)
        print(m)
