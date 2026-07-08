"""Generate images from prompts with FLUX.1-dev via diffusers."""

import json
import random
import time
from datetime import datetime
from pathlib import Path

from .config import Config


def load_pipeline(cfg: Config):
    import torch
    from diffusers import FluxPipeline

    print(f"Loading {cfg.image_model} (bf16, CPU offload)...")
    pipe = FluxPipeline.from_pretrained(cfg.image_model, torch_dtype=torch.bfloat16)
    # The bf16 transformer alone is ~24GB; offloading keeps each component on
    # the GPU only while it runs, so the whole pipeline fits in 24GB VRAM.
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    return pipe


def generate_images(
    cfg: Config,
    prompts: list[dict],
    out_dir: str | Path | None = None,
    base_seed: int | None = None,
) -> Path:
    import torch

    out_dir = Path(out_dir or cfg.outputs_dir) / f"{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"

    pipe = load_pipeline(cfg)
    rng = random.Random(base_seed)

    for i, rec in enumerate(prompts):
        seed = rng.randrange(2**32)
        t0 = time.time()
        image = pipe(
            prompt=rec["prompt"],
            width=cfg.width,
            height=cfg.height,
            num_inference_steps=cfg.steps,
            guidance_scale=cfg.guidance,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(seed),
        ).images[0]

        name = f"{rec.get('id', i):03d}_{seed}"
        meta = {
            **rec,
            "seed": seed,
            "width": cfg.width,
            "height": cfg.height,
            "steps": cfg.steps,
            "guidance": cfg.guidance,
            "image_model": cfg.image_model,
            "file": f"{name}.png",
            "seconds": round(time.time() - t0, 1),
        }

        from PIL.PngImagePlugin import PngInfo

        pnginfo = PngInfo()
        pnginfo.add_text("parameters", json.dumps(meta))
        image.save(out_dir / f"{name}.png", pnginfo=pnginfo)
        with manifest.open("a") as f:
            f.write(json.dumps(meta) + "\n")
        print(f"  [{i + 1}/{len(prompts)}] {name}.png ({meta['seconds']}s)")

    print(f"Images written to {out_dir}")
    return out_dir
