"""Central configuration with environment-variable overrides."""

import os
import subprocess
from dataclasses import dataclass, field


def detect_ollama_host() -> str:
    """Find a reachable Ollama server.

    Checks OLLAMA_HOST, then localhost, then the WSL2 default gateway
    (the Windows host, where Ollama typically runs).
    """
    import requests

    candidates = []
    if os.environ.get("OLLAMA_HOST"):
        host = os.environ["OLLAMA_HOST"]
        if not host.startswith("http"):
            host = f"http://{host}"
        candidates.append(host)
    candidates.append("http://localhost:11434")
    try:
        gateway = subprocess.check_output(
            ["sh", "-c", "ip route show default | awk '{print $3}'"], text=True
        ).strip()
        if gateway:
            candidates.append(f"http://{gateway}:11434")
    except subprocess.CalledProcessError:
        pass

    for base in candidates:
        try:
            requests.get(f"{base}/api/tags", timeout=3)
            return base
        except requests.RequestException:
            continue
    raise RuntimeError(
        f"No Ollama server reachable (tried {candidates}). "
        "Set OLLAMA_HOST or start Ollama."
    )


@dataclass
class Config:
    # Prompt generation
    ollama_model: str = os.environ.get("IR_OLLAMA_MODEL", "qwen3.6:latest")
    prompt_count: int = 50
    prompts_per_call: int = 5  # small batches keep quality up and topics varied
    temperature: float = 1.0
    prompts_dir: str = "prompts"

    # Image generation
    image_model: str = os.environ.get("IR_IMAGE_MODEL", "black-forest-labs/FLUX.1-dev")
    width: int = 1920
    height: int = 1088  # must be a multiple of 16
    steps: int = 28
    guidance: float = 3.5
    outputs_dir: str = "outputs"

    ollama_host: str = field(default="", repr=False)

    def resolve_ollama(self) -> str:
        if not self.ollama_host:
            self.ollama_host = detect_ollama_host()
        return self.ollama_host
