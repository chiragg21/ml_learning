from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from scripts.train import load_config
from src.data.factory import DataFactory
from src.data.seq2seq import build_datasets, make_collate_fn
from src.evaluation.evaluator import Evaluator
from src.inference.base import LoadedExperiment
from src.quantization.quantize import (
    count_quantized_layers,
    model_size_bytes,
    quantize_model,
)


def compare_quantization(
    config,
    bits: int = 8,
    group_size: int | None = None,
    *,
    experiment_manager=None,
    device: str = "cpu",
) -> dict[str, float]:
    """
    Loads a trained experiment, quantizes a copy of its model and
    evaluates both on the validation set, so the size saved and the
    accuracy given up can be read side by side.
    """

    loaded = LoadedExperiment(
        config,
        experiment_manager=experiment_manager,
        device=device,
    )

    _, val_pairs, _ = DataFactory.build(config)

    _, validation_dataset = build_datasets(config, loaded.tokenizer, [], val_pairs)

    if validation_dataset is None:
        raise ValueError("This config has no validation split to evaluate on.")

    loader = DataLoader(
        validation_dataset,
        batch_size=config.training.batch_size,
        collate_fn=make_collate_fn(loaded.tokenizer.pad_token_id),
    )

    quantized = quantize_model(loaded.model, bits=bits, group_size=group_size).to(device)

    def evaluate(model) -> dict[str, float]:
        return Evaluator(
            model=model,
            device=device,
            pad_token_id=loaded.tokenizer.pad_token_id,
        ).evaluate(loader)

    float_metrics = evaluate(loaded.model)
    quantized_metrics = evaluate(quantized)

    float_size = model_size_bytes(loaded.model)
    quantized_size = model_size_bytes(quantized)

    results = {
        "float_bytes": float_size,
        "quantized_bytes": quantized_size,
        "size_ratio": float_size / quantized_size,
        "quantized_layers": count_quantized_layers(quantized),
        **{f"float_{key}": value for key, value in float_metrics.items()},
        **{f"quantized_{key}": value for key, value in quantized_metrics.items()},
    }

    for key, value in results.items():
        print(f"{key}: {value}", flush=True)

    return results


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--bits", type=int, default=8, choices=(4, 8))
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    compare_quantization(
        load_config(args.config),
        bits=args.bits,
        group_size=args.group_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
