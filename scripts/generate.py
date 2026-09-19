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

    args = parser.parse_args()

    config = load_config(args.config)
    generator = Generator(config)

    output = generator.generate_text(
        args.text,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print(output)


if __name__ == "__main__":
    main()
