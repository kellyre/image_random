"""Generate images from prompts with FLUX.1-dev via diffusers."""

import json
import random
import time
from datetime import datetime
from pathlib import Path

from .config import Config


def _resolve_model_path(model_id: str) -> str:
    """Return the local snapshot path if the model is fully cached.

    Loading from a local path skips all Hub API calls, which both avoids
    needing a valid HF token for cached models and works around diffusers
    calling the Hub for sharded checkpoints even when they are cached.
    """
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_id, local_files_only=True)
    except Exception:
        return model_id  # not cached; let from_pretrained download it


def load_pipeline(cfg: Config):
    import torch
    from diffusers import FluxPipeline

    model_path = _resolve_model_path(cfg.image_model)
    print(f"Loading {cfg.image_model} from {model_path}...")
    if cfg.quantize:
        # The bf16 transformer alone is ~24GB, which overflows a 24GB card and
        # makes the Windows driver spill into shared memory (~15x slower).
        # NF4 weights bring the whole pipeline to ~10GB so it fits on-GPU.
        from diffusers import BitsAndBytesConfig as DiffusersBnbConfig
        from diffusers import FluxTransformer2DModel
        from transformers import BitsAndBytesConfig as TransformersBnbConfig
        from transformers import T5EncoderModel

        transformer = FluxTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            quantization_config=DiffusersBnbConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            torch_dtype=torch.bfloat16,
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            model_path,
            subfolder="text_encoder_2",
            quantization_config=TransformersBnbConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            torch_dtype=torch.bfloat16,
        )
        pipe = FluxPipeline.from_pretrained(
            model_path,
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            torch_dtype=torch.bfloat16,
        )
        pipe.enable_model_cpu_offload()
    else:
        pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()

    pipe.vae.enable_tiling()
    return pipe


def encode_prompt(pipe, prompt: str, max_tokens: int):
    """Encode a prompt, allowing T5 sequences beyond diffusers' 512 cap.

    The pipeline's __call__ rejects max_sequence_length > 512, but
    encode_prompt itself does not, so long prompts are pre-encoded here and
    passed in as embeddings. Prompts that fit in 512 tokens use exactly the
    trained sequence length; only longer ones extend it.
    """
    n_tokens = pipe.tokenizer_2(prompt, return_tensors="pt", truncation=False).input_ids.shape[1]
    seq_len = 512 if n_tokens <= 512 else min(max_tokens, n_tokens)
    if n_tokens > max_tokens:
        print(f"  Warning: prompt is {n_tokens} T5 tokens, truncating to {max_tokens}")
    prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt, prompt_2=prompt, max_sequence_length=seq_len
    )
    return prompt_embeds, pooled_prompt_embeds


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
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            pipe, rec["prompt"], cfg.max_prompt_tokens
        )
        image = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            width=cfg.width,
            height=cfg.height,
            num_inference_steps=cfg.steps,
            guidance_scale=cfg.guidance,
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
            "max_prompt_tokens": cfg.max_prompt_tokens,
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
