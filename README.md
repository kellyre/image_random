# image_random

Generate batches of long, detailed, wildly varied image prompts with a local
LLM (via Ollama), then render them as high-resolution landscape images with
FLUX.1-dev.

## How it works

1. **Prompt generation** — the program calls Ollama (default model
   `qwen3.6`) in small batches. Each batch is seeded with random broad
   domains (30 of them) and a photographic style; the LLM invents a fresh,
   specific concept per domain. Every concept ever generated (read from all
   files in `prompts/`) is fed back as an avoid-list, so new prompts don't
   repeat earlier themes — keep the `prompts/` folder to keep that memory.
   Prompts are 200–350 word single-paragraph descriptions of complex scenes
   (multiple simultaneous actions, distinct foreground/midground/background)
   that favor fantastical premises rendered with physically real textures.
   Saved to `prompts/prompts_<timestamp>.jsonl`.
2. **VRAM handoff** — after prompting, Ollama is asked to unload its model
   (`keep_alive: 0`) so FLUX gets the whole GPU.
3. **Image generation** — FLUX.1-dev, NF4-quantized with bitsandbytes so the
   whole pipeline (~10GB) fits on a 24GB GPU, renders each prompt at
   1920×1088 (~45–60s per image). Every PNG embeds its prompt, seed, and
   settings in its metadata, and a `manifest.jsonl` is written alongside.
   Pass `--no-quantize` for full bf16 if you have >24GB of VRAM (on a 24GB
   card bf16 spills into shared memory and is ~15x slower).

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
# torch must match your driver's CUDA version (check `nvidia-smi`), e.g.:
venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

Requirements:
- NVIDIA GPU with 24GB VRAM (tested on an RTX 4090 under WSL2)
- Ollama running locally or on the Windows host (auto-detected in WSL2)
- Hugging Face token with access to `black-forest-labs/FLUX.1-dev`
  (`hf auth login`) — only needed for the initial model download; cached
  models load from disk without a valid token

## Usage

```bash
# Full pipeline: 50 prompts, then 50 images
venv/bin/python run.py all --count 50

# Prompts only
venv/bin/python run.py prompts --count 50 --llm qwen3.6:latest

# Images from an existing prompt file
venv/bin/python run.py images --prompts prompts/prompts_20260707_220000.jsonl

# Quick test: 3 prompts, smaller/faster images
venv/bin/python run.py all --count 3 --width 1344 --height 768 --steps 20
```

Useful flags: `--width/--height` (multiples of 16), `--steps` (default 28),
`--guidance` (default 3.5), `--limit N` (render only the first N prompts),
`--seed` / `--base-seed` for reproducibility, `--max-tokens` (T5 budget,
default 1024 — FLUX was trained at 512, so longer prompts are fully read
but may introduce artifacts; set 512 for exact trained behavior).

Environment overrides: `IR_OLLAMA_MODEL`, `IR_IMAGE_MODEL`, `OLLAMA_HOST`.

At 1920×1088 / 28 steps, expect roughly 45–60 seconds per image on an
RTX 4090; a batch of 50 takes about 40–50 minutes.
