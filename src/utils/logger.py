import json
from loguru import logger
from pathlib import Path
from src.utils.config import PROJECT_ROOT

LOG_FILE = PROJECT_ROOT / "examples" / "pipeline.log"
logger.add(LOG_FILE, rotation="1 MB")

def log_info(msg: str):
    logger.info(msg)

def write_manifest(data: dict, path: Path = None):
    path = path or (PROJECT_ROOT / "examples" / "manifest.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    log_info(f"Wrote manifest to {path}")