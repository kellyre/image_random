"""Generate diverse, detailed image prompts with a local LLM via Ollama."""

import json
import random
import time
from datetime import datetime
from pathlib import Path

import requests

from .config import Config
from .topics import sample_style, sample_topics

SYSTEM = """\
You are a world-class prompt writer for the FLUX.1 text-to-image model.
You write vivid, concrete, self-contained image prompts. Each prompt:
- is a single paragraph of 60-120 words
- describes exactly one scene with specific subjects, setting, lighting,
  color palette, atmosphere, composition, and camera or medium details
- is written as a description of the image, never as an instruction or
  a story, and never mentions text, words, or signage content
- uses landscape (wide) composition
"""

USER_TEMPLATE = """\
Write {n} image prompts. Use each of these topics for exactly one prompt,
treating the topic as loose inspiration you expand with your own specifics:
{topics}

Render all of them in this overall style: {style}

Return JSON: {{"prompts": ["...", "..."]}}
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "prompts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["prompts"],
}


def _call_ollama(cfg: Config, topics: list[str], style: str) -> list[str]:
    body = {
        "model": cfg.ollama_model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    n=len(topics),
                    topics="\n".join(f"- {t}" for t in topics),
                    style=style,
                ),
            },
        ],
        "stream": False,
        "think": False,
        "format": RESPONSE_SCHEMA,
        "options": {"temperature": cfg.temperature},
    }
    resp = requests.post(f"{cfg.resolve_ollama()}/api/chat", json=body, timeout=600)
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    prompts = json.loads(content)["prompts"]
    return [p.strip() for p in prompts if isinstance(p, str) and len(p.strip()) > 40]


def unload_ollama_model(cfg: Config) -> None:
    """Free the LLM's VRAM so the image model can use the whole GPU."""
    try:
        requests.post(
            f"{cfg.resolve_ollama()}/api/generate",
            json={"model": cfg.ollama_model, "keep_alive": 0},
            timeout=30,
        )
        print(f"Asked Ollama to unload {cfg.ollama_model}")
    except requests.RequestException as e:
        print(f"Warning: could not unload Ollama model: {e}")


def generate_prompts(cfg: Config, count: int | None = None, seed: int | None = None) -> Path:
    """Generate prompts in small batches and write them to a JSONL file."""
    count = count or cfg.prompt_count
    rng = random.Random(seed)
    out_dir = Path(cfg.prompts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"prompts_{datetime.now():%Y%m%d_%H%M%S}.jsonl"

    collected: list[dict] = []
    attempts = 0
    while len(collected) < count and attempts < count * 3:
        attempts += 1
        n = min(cfg.prompts_per_call, count - len(collected))
        topics = sample_topics(n, rng)
        style = sample_style(rng)
        t0 = time.time()
        try:
            batch = _call_ollama(cfg, topics, style)
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            print(f"Batch failed ({e}), retrying...")
            continue
        for prompt, topic in zip(batch, topics):
            collected.append({"prompt": prompt, "topic": topic, "style": style})
        print(
            f"  {len(collected)}/{count} prompts "
            f"(batch of {len(batch)} in {time.time() - t0:.1f}s)"
        )

    collected = collected[:count]
    with out_path.open("w") as f:
        for i, rec in enumerate(collected):
            rec["id"] = i
            rec["model"] = cfg.ollama_model
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(collected)} prompts to {out_path}")

    unload_ollama_model(cfg)
    return out_path


def load_prompts(path: str | Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]
