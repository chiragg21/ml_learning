from __future__ import annotations

import argparse

from scripts.train import load_config
from src.generator.generator import Generator

EXIT_TOKENS = {"/exit", "/quit", "/q"}


def _run_repl(generator: Generator, args: argparse.Namespace) -> None:

    print(
        f"Interactive mode - enter a sentence to translate, "
        f"or type one of {sorted(EXIT_TOKENS)} to quit.",
        flush=True,
    )

    while True:

        try:
            text = input("> ").strip()
        except EOFError:
            break

        if not text:
            continue

        if text.lower() in EXIT_TOKENS:
            break

        output = generator.generate_text(
            text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.sample,
            top_k=args.top_k,
        )

        print(output, flush=True)


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--text",
        default=None,
        help="Translate this one sentence and exit. Omit to enter interactive mode.",
    )
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

    # Loading the model/tokenizer is the slow part - do it once and
    # reuse the same Generator across every sentence in interactive
    # mode, rather than paying that cost per sentence.
    generator = Generator(config)

    if args.text is None:
        _run_repl(generator, args)
        return

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
