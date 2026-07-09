"""Generate diverse, detailed image prompts with a local LLM via Ollama."""

import json
import random
import time
from datetime import datetime
from pathlib import Path

import requests

from .config import Config
from .topics import sample_domains, sample_style

SYSTEM = """\
You are a world-class prompt writer for the FLUX.1 text-to-image model.
You write vivid, concrete, self-contained image prompts. Each prompt:
- is a single paragraph of 200-350 words describing ONE complex scene with
  several things happening at once: a clear main subject plus secondary
  actions and characters, and distinct foreground, midground, and background
  elements that each get specific detail
- favors imaginative, fantastical, or surreal premises, but renders every
  surface with physically real, tactile texture: weathered wood grain, worn
  metal, wet stone, fabric weave, skin, fur, dust, condensation - as if a
  real photograph of an impossible thing
- specifies lighting, color palette, atmosphere, weather, composition, and
  camera details (lens, depth of field, vantage point)
- is written as a description of the image, never as an instruction or a
  story, and never mentions text, words, or signage content
- uses landscape (wide) composition
"""

USER_TEMPLATE = """\
Write {n} image prompts. For each of these broad domains, invent one fresh,
specific scene concept (a short phrase) and then write the full prompt for it:
{domains}

Render all of them in this overall style: {style}

Novelty is critical. Here are concepts that were already used - every new
concept must be clearly different from ALL of these (different subjects,
settings, and central ideas, not just rewordings):
{avoid}

Return JSON: {{"items": [{{"concept": "...", "prompt": "..."}}, ...]}}
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string"},
                    "prompt": {"type": "string"},
                },
                "required": ["concept", "prompt"],
            },
        },
    },
    "required": ["items"],
}

MAX_AVOID = 400  # most recent concepts fed back to the LLM


def load_previous_concepts(cfg: Config) -> list[str]:
    """Collect concepts from every prompt file generated so far."""
    concepts: list[str] = []
    for path in sorted(Path(cfg.prompts_dir).glob("*.jsonl")):
        try:
            with path.open() as f:
                for line in f:
                    rec = json.loads(line)
                    topic = rec.get("topic")
                    if topic:
                        concepts.append(topic)
        except (OSError, json.JSONDecodeError):
            continue
    return concepts


def _call_ollama(
    cfg: Config, domains: list[str], style: str, avoid: list[str]
) -> list[dict]:
    avoid_text = (
        "\n".join(f"- {c}" for c in avoid[-MAX_AVOID:]) if avoid else "(none yet)"
    )
    body = {
        "model": cfg.ollama_model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    n=len(domains),
                    domains="\n".join(f"- {d}" for d in domains),
                    style=style,
                    avoid=avoid_text,
                ),
            },
        ],
        "stream": False,
        "think": False,
        "format": RESPONSE_SCHEMA,
        "options": {"temperature": cfg.temperature},
    }
    resp = requests.post(f"{cfg.resolve_ollama()}/api/chat", json=body, timeout=1200)
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    items = json.loads(content)["items"]
    return [
        it
        for it in items
        if isinstance(it, dict) and len(it.get("prompt", "")) > 100 and it.get("concept")
    ]


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

    avoid = load_previous_concepts(cfg)
    if avoid:
        print(f"Avoiding {len(avoid)} previously used concepts")

    collected: list[dict] = []
    attempts = 0
    while len(collected) < count and attempts < count * 3:
        attempts += 1
        n = min(cfg.prompts_per_call, count - len(collected))
        domains = sample_domains(n, rng)
        style = sample_style(rng)
        t0 = time.time()
        try:
            batch = _call_ollama(cfg, domains, style, avoid)
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            print(f"Batch failed ({e}), retrying...")
            continue
        for item, domain in zip(batch, domains):
            collected.append(
                {
                    "prompt": item["prompt"].strip(),
                    "topic": item["concept"].strip(),
                    "domain": domain,
                    "style": style,
                }
            )
            avoid.append(item["concept"].strip())
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
