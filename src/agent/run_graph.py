# src/agent/run_graph.py
from pathlib import Path
from src.rag.retriever import SimpleRetriever
from src.rag.generator import RagGenerator
from src.agent.langgraph import LangGraph
from src.tools.image_gen import generate_image
from src.tools.tts import synthesize_speech
from src.tools.music_gen import generate_music
from src.utils.config import PROJECT_ROOT

GRAPH_JSON_PATH = Path(PROJECT_ROOT) / "langgraph" / "graph.json"

def simple_planner(inp: dict):
    up = inp.get("user_prompt", "")
    return {"user_prompt": up, "query": up}

def build_demo_retriever():
    docs = []
    readme = Path(PROJECT_ROOT) / "README.md"
    if readme.exists():
        docs.append(readme.read_text(encoding="utf-8"))
    docs.append("Multimodal agent POC demo knowledge: generate image, tts, music, video.")
    r = SimpleRetriever()
    r.build(docs)
    return r

def make_tools():
    retr = build_demo_retriever()
    gen = RagGenerator()
    tools = {
        "planner": simple_planner,
        "retriever": retr,
        "generator": gen,
        "image_gen": lambda prompt: generate_image(prompt, out_name="rag_image.png"),
        "tts": lambda text: synthesize_speech(text, out_name="rag_speech.wav"),
        "music_gen": lambda prompt: generate_music(prompt, out_name="rag_music.wav"),
    }
    return tools

def load_graph_and_run(prompt: str):
    tools = make_tools()
    g = LangGraph.from_json(str(GRAPH_JSON_PATH), tools=tools)
    out = g.run("n1", {"user_prompt": prompt})
    return out

if __name__ == "__main__":
    print(load_graph_and_run("Describe a calm blue mountain landscape and produce an image prompt"))