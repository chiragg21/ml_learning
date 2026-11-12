from __future__ import annotations

import argparse

from scripts.train import load_config
from src.generator.generator import Generator


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample instead of greedy decoding (makes --temperature actually do something).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Only sample from the top-k most likely tokens (requires --sample).",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    generator = Generator(config)

    output = generator.generate_text(
        args.text,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.sample,
        top_k=args.top_k,
    )

    print(output)


if __name__ == "__main__":
    main()
