# Multimodal-Agent-POC

**Multimodal-Agent-POC** is an end-to-end demo showcasing how a single text prompt can be transformed into image, speech, music, and video outputs using only open-source AI models and free infrastructure. It highlights agentic orchestration (LangGraph-style), cost-efficient design, and smart integration of HuggingFace models, demonstrating autonomy, resourcefulness, and innovation in multimodal AI workflows.

## Features

* **Text → Image** (Stable Diffusion)
* **Text → Speech** (TTS with fallback WAV generator)
* **Text → Music** (MusicGen with safe fallback)
* **Video Assembly** (image + audio → MP4)
* **Agentic Orchestration** (LangGraph-style planner + modular tools)
* **RAG mode** (retrieval + grounded generation → speech/image output)
* **100% Open Source**, no paid APIs

## Installation

```bash
# Clone repo
git clone https://github.com/yourname/multimodal-agent-poc.git
cd multimodal-agent-poc

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

## GPU Setup (CUDA Acceleration)

Install PyTorch with CUDA (example for CUDA 12.6):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

* Models automatically detect GPU (torch.cuda.is_available()).
* Using CUDA acceleration ensures fast and efficient generation.
* Falls back to CPU if GPU is not available.

## Install FFmpeg

**Windows**: Download [builds](https://www.gyan.dev/ffmpeg/builds/) and add to PATH.

**Linux/Mac**:

```bash
sudo apt-get install ffmpeg    # Ubuntu/Debian
brew install ffmpeg            # MacOS
```

## Run Modes

### 1. Orchestrator (Linear Pipeline)

```bash
python -m src.orchestrator
```

* Fixed sequence: Plan → Image → Speech → Music → Mix → Video.
* Produces:
  * `sample_image.png`
  * `sample_speech.wav`
  * `sample_music.wav`
  * `final_audio.wav`
  * `final_video.mp4`
* Simple, deterministic, robust with fallbacks.

**Default Prompt** (in orchestrator.py):
"A calm blue mountain landscape at sunset with birds flying"

### 2. Agentic Orchestration (LangGraph Workflow)

```bash
python -m src.agent.run_graph
```

* Executes workflow defined in `langgraph/graph.json`.
* Nodes: planner → retriever → generator → tools.
* Produces:
  * `rag_image.png`
  * `rag_speech.wav`
* Demonstrates **agentic behavior**: planning, branching, tool selection.

**Default Prompt** (in run_graph.py):
"Generate a futuristic city skyline with flying cars"

### 3. RAG Mode (Retrieval + Generation)

```bash
python -m src.agent.rag
```

* Adds **retrieval step** (FAISS + embeddings) → grounds generation in docs.
* Reduces hallucination, ensures factual outputs.
* Produces grounded text → speech + image.

**Default Prompt** (in rag.py):
"Explain how transformers work in deep learning"

## Pipeline Comparison

| Mode | Flow | Strengths |
|------|------|-----------|
| **Orchestrator** | Linear fixed steps → image, speech, music, video | Fast, reliable, easy demo |
| **RAG** | Retrieve docs → ground generator → tools (speech/image) | Knowledge-aware, factual |
| **LangGraph** | Graph-based nodes: planner, retriever, generator, tool execution | Agentic orchestration, branching, retries |

## Project Structure

```
multimodal-agent-poc/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── src/
│   ├── orchestrator.py
│   ├── tools/ (image_gen, tts, music_gen, speech_to_text, video)
│   ├── utils/ (logger, config)
│   └── agent/ (langgraph, run_graph, rag)
│
├── examples/
│   ├── outputs/
│   └── manifest.json
```

## Example Runs

| Mode | Sample Prompt | Location in Code | Outputs |
|------|---------------|------------------|---------|
| **Orchestrator** | "A calm blue mountain landscape at sunset with birds flying" | orchestrator.py | sample_image.png, sample_speech.wav, sample_music.wav, final_video.mp4 |
| **LangGraph** | "Generate a futuristic city skyline with flying cars" | run_graph.py | rag_image.png, rag_speech.wav |
| **RAG** | "Explain how transformers work in deep learning" | rag.py | rag_image.png, rag_speech.wav |
