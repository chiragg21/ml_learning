from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class CheckpointManager:

    def save(
        self,
        path: str | Path,
        *,
        model,
        optimizer,
        scheduler=None,
        epoch: int = 0,
        global_step: int = 0,
        metrics: dict[str, float] | None = None,
    ) -> None:

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),

            "optimizer_state_dict": (
                optimizer.state_dict()
                if optimizer is not None
                else None
            ),

            "scheduler_state_dict": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),

            "epoch": epoch,
            "global_step": global_step,
            "metrics": metrics or {},
        }

        torch.save(
            checkpoint,
            path,
        )

    def load(
        self,
        path: str | Path,
        *,
        model,
        optimizer=None,
        scheduler=None,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any]:

        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if (
            optimizer is not None
            and checkpoint["optimizer_state_dict"] is not None
        ):
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        if (
            scheduler is not None
            and checkpoint["scheduler_state_dict"] is not None
        ):
            scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        return {
            "epoch": checkpoint["epoch"],
            "global_step": checkpoint["global_step"],
            "metrics": checkpoint["metrics"],
        }