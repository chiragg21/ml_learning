from __future__ import annotations

import argparse
from pathlib import Path


def download_wikitext(data_dir: Path) -> None:

    from datasets import load_dataset

    if data_dir.exists() and any(data_dir.iterdir()):
        print(f"{data_dir} already populated - skipping download.")
        return

    data_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
    )

    for split in ["train", "validation", "test"]:

        with open(data_dir / f"{split}.txt", "w", encoding="utf-8") as file:

            for row in dataset[split]:
                file.write(row["text"] + "\n")

    print(f"Wrote wikitext-2 to {data_dir}")


def download_translation(data_dir: Path) -> None:

    from datasets import load_dataset

    if data_dir.exists() and any(data_dir.iterdir()):
        print(f"{data_dir} already populated - skipping download.")
        return

    data_dir.mkdir(parents=True, exist_ok=True)

    # Helsinki-NLP/opus-100: parallel English<->French sentences, one of
    # the 100 language pairs in the OPUS-100 MT benchmark. Ships its own
    # train/validation/test splits.
    dataset = load_dataset("Helsinki-NLP/opus-100", "en-fr")

    for split in ["train", "validation", "test"]:

        en_path = data_dir / f"{split}.en.txt"
        fr_path = data_dir / f"{split}.fr.txt"

        with open(en_path, "w", encoding="utf-8") as en_file, \
             open(fr_path, "w", encoding="utf-8") as fr_file:

            for row in dataset[split]:

                pair = row["translation"]

                # One sentence per line, line-aligned across the two
                # files - collapse any embedded newlines so alignment
                # can't drift.
                en_file.write(pair["en"].replace("\n", " ").strip() + "\n")
                fr_file.write(pair["fr"].replace("\n", " ").strip() + "\n")

    print(f"Wrote OPUS-100 en-fr to {data_dir}")


DOWNLOADERS = {
    "wikitext": (download_wikitext, "data/raw/wikitext-2"),
    "translation": (download_translation, "data/raw/opus100_en_fr"),
}


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Download a raw data source for one of the configs under configs/."
    )

    parser.add_argument(
        "--dataset",
        choices=sorted(DOWNLOADERS),
        default="translation",
        help="Which data source to download (default: translation, used by configs/translation_v0.py).",
    )

    args = parser.parse_args()

    downloader, default_dir = DOWNLOADERS[args.dataset]
    downloader(Path(default_dir))


if __name__ == "__main__":
    main()
