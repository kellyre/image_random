"""Command-line interface.

Usage:
    python run.py prompts [--count 50]
    python run.py images --prompts prompts/prompts_XXXX.jsonl [--limit N]
    python run.py all [--count 50]
"""

import argparse
from pathlib import Path

from .config import Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image_random",
        description="Generate random prompts with a local LLM, then render them with FLUX.1-dev",
    )
    defaults = Config()
    sub = parser.add_subparsers(dest="command", required=True)

    p_prompts = sub.add_parser("prompts", help="generate prompts only")
    p_images = sub.add_parser("images", help="render images from an existing prompt file")
    p_all = sub.add_parser("all", help="generate prompts, then render them")

    for p in (p_prompts, p_all):
        p.add_argument("--count", type=int, default=defaults.prompt_count)
        p.add_argument("--llm", default=defaults.ollama_model)
        p.add_argument("--seed", type=int, default=None, help="seed for topic sampling")

    p_images.add_argument("--prompts", required=True, help="path to a prompts .jsonl file")
    for p in (p_images, p_all):
        p.add_argument("--limit", type=int, default=None, help="only render the first N prompts")
        p.add_argument("--width", type=int, default=defaults.width)
        p.add_argument("--height", type=int, default=defaults.height)
        p.add_argument("--steps", type=int, default=defaults.steps)
        p.add_argument("--guidance", type=float, default=defaults.guidance)
        p.add_argument("--base-seed", type=int, default=None, help="seed for image seeds")
        p.add_argument(
            "--no-quantize",
            action="store_true",
            help="run the transformer in bf16 (needs >24GB VRAM to be fast)",
        )

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = Config()

    if args.command in ("prompts", "all"):
        cfg.ollama_model = args.llm
    if args.command in ("images", "all"):
        if args.width % 16 or args.height % 16:
            raise SystemExit("--width and --height must be multiples of 16")
        cfg.width, cfg.height = args.width, args.height
        cfg.steps, cfg.guidance = args.steps, args.guidance
        cfg.quantize = not args.no_quantize

    from .prompt_gen import generate_prompts, load_prompts

    if args.command == "prompts":
        generate_prompts(cfg, count=args.count, seed=args.seed)
        return

    if args.command == "all":
        prompts_path = generate_prompts(cfg, count=args.count, seed=args.seed)
    else:
        prompts_path = Path(args.prompts)

    prompts = load_prompts(prompts_path)
    if args.limit:
        prompts = prompts[: args.limit]

    from .image_gen import generate_images

    generate_images(cfg, prompts, base_seed=args.base_seed)


if __name__ == "__main__":
    main()
