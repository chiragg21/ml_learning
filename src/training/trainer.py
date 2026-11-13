from __future__ import annotations

import math
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset

from artifacts.manager import ExperimentManager
from src.data.seq2seq import BucketBatchSampler, make_collate_fn
from src.evaluation.evaluator import Evaluator
from src.models.factory import ModelFactory
from src.training.checkpoint import CheckpointManager


class Trainer:

    def __init__(
        self,
        config,
        train_dataset,
        tokenizer,
        validation_dataset=None,
        *,
        experiment_manager=None,
        mlflow_logger=None,
    ):

        self.config = config
        self.tokenizer = tokenizer

        # =============================================
        # Experiment
        # =============================================

        self.experiment = (
            experiment_manager
            or ExperimentManager(config)
        )

        self.experiment.save_config()

        # =============================================
        # Runtime
        # =============================================

        self.device = self._resolve_device()

        self._set_seed(
            config.runtime.seed
        )

        # =============================================
        # Model
        # =============================================

        self.model = ModelFactory.create(
            config,
            tokenizer=tokenizer,
        )

        self.model.to(
            self.device
        )

        if config.training.gradient_checkpointing:

            if not hasattr(self.model, "set_gradient_checkpointing"):
                raise ValueError(
                    "training.gradient_checkpointing needs a model with "
                    "set_gradient_checkpointing (transformer_v2)."
                )

            self.model.set_gradient_checkpointing(True)

        if config.runtime.compile_model:

            self.model = torch.compile(
                self.model
            )

        # =============================================
        # Data
        # =============================================

        collate_fn = make_collate_fn(
            tokenizer.pad_token_id
        )

        # pin_memory speeds up host->device transfer and is only
        # meaningful when that transfer target is a GPU.
        pin_memory = self.device.type == "cuda"

        persistent_workers = config.runtime.num_workers > 0

        # Workers load the next batches while the model computes on the
        # current one; prefetch_factor sets how many each keeps ready.
        worker_options = (
            {
                "prefetch_factor": config.runtime.extra.get("prefetch_factor", 2),
                "persistent_workers": persistent_workers,
            }
            if persistent_workers
            else {}
        )

        # A stream can't be shuffled or length-bucketed by index; it
        # shuffles itself (see StreamingSeq2SeqDataset).
        is_stream = isinstance(train_dataset, IterableDataset)

        if config.data.extra.get("bucket_by_length"):

            if is_stream:
                raise ValueError(
                    "data.extra: bucket_by_length can't be used with streaming."
                )

            sampler_options = {
                "batch_sampler": BucketBatchSampler(
                    train_dataset.lengths,
                    config.training.batch_size,
                    seed=config.runtime.seed,
                ),
            }

        else:

            sampler_options = {
                "batch_size": config.training.batch_size,
                "shuffle": not is_stream,
            }

        self.train_loader = DataLoader(
            train_dataset,
            num_workers=config.runtime.num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            **sampler_options,
            **worker_options,
        )

        self.validation_loader = None

        if validation_dataset is not None:

            self.validation_loader = DataLoader(
                validation_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=config.runtime.num_workers,
                collate_fn=collate_fn,
                pin_memory=pin_memory,
                **worker_options,
            )

        # =============================================
        # Training
        # =============================================

        self.optimizer = (
            self._create_optimizer()
        )

        self.scheduler = (
            self._create_scheduler()
        )

        self.checkpoint_manager = (
            CheckpointManager()
        )

        # =============================================
        # Mixed precision
        # =============================================

        training_config = config.training

        self.mixed_precision = (
            training_config.mixed_precision
            and training_config.precision in ("fp16", "bf16")
        )

        # fp16 autocast isn't practically supported on CPU (no native
        # fp16 compute); fall back to bf16 there, which CPU does support.
        self.amp_dtype = (
            torch.float16
            if training_config.precision == "fp16"
            and self.device.type == "cuda"
            else torch.bfloat16
        )

        # GradScaler is a no-op when enabled=False or dtype != fp16, so
        # it's always safe to route through it regardless of settings.
        self.scaler = torch.amp.GradScaler(
            device=self.device.type if self.device.type == "cuda" else "cpu",
            enabled=(
                self.mixed_precision
                and self.amp_dtype == torch.float16
            ),
        )

        self.evaluator = Evaluator(
            model=self.model,
            device=self.device,
            pad_token_id=tokenizer.pad_token_id,
            mixed_precision=self.mixed_precision,
            amp_dtype=self.amp_dtype,
        )

        self.global_step = 0

        # =============================================
        # MLflow
        # =============================================

        self.mlflow_logger = mlflow_logger

    # ==================================================
    # Setup helpers
    # ==================================================

    def _resolve_device(self) -> torch.device:

        device = self.config.runtime.device

        if device == "auto":

            return torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        return torch.device(device)

    @staticmethod
    def _set_seed(seed: int) -> None:

        random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _create_optimizer(self) -> torch.optim.Optimizer:

        training_config = self.config.training
        name = training_config.optimizer.lower()

        if name == "adamw":

            return torch.optim.AdamW(
                self.model.parameters(),
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay,
            )

        if name == "adam":

            return torch.optim.Adam(
                self.model.parameters(),
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay,
            )

        if name == "sgd":

            return torch.optim.SGD(
                self.model.parameters(),
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay,
            )

        raise ValueError(
            f"Unknown optimizer: {training_config.optimizer}"
        )

    def _create_scheduler(self):

        scheduler = self.config.training.scheduler

        if scheduler is None:
            return None

        if scheduler == "linear_warmup":

            warmup_steps = max(
                self.config.training.warmup_steps,
                1,
            )

            def lr_lambda(step: int) -> float:
                return min(1.0, step / warmup_steps)

            return torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda,
            )

        if scheduler == "cosine_warmup":

            # v1's roadmap item: linear warmup up to warmup_steps, then
            # cosine decay from the peak LR down to 0 over the rest of
            # training. The cosine shape (vs. linear/step decay) spends
            # more steps near the peak LR and tapers off smoothly at the
            # end, which is the de facto standard LR schedule for
            # training transformers from scratch.
            warmup_steps = max(
                self.config.training.warmup_steps,
                1,
            )

            total_steps = max(
                (
                    len(self.train_loader)
                    * self.config.training.epochs
                )
                // self.config.training.gradient_accumulation_steps,
                warmup_steps + 1,
            )

            def lr_lambda(step: int) -> float:

                if step < warmup_steps:
                    return step / warmup_steps

                progress = (
                    (step - warmup_steps)
                    / (total_steps - warmup_steps)
                )

                progress = min(progress, 1.0)

                return 0.5 * (1.0 + math.cos(math.pi * progress))

            return torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda,
            )

        raise ValueError(
            f"Unknown scheduler: {scheduler}"
        )

    # ==================================================
    # Training loop
    # ==================================================

    def train(self) -> dict[str, float]:

        training_config = self.config.training

        if self.mlflow_logger is not None:

            self.mlflow_logger.log_experiment_config(
                self.config,
                self.experiment.experiment_hash,
            )

        final_metrics: dict[str, float] = {}

        print(
            f"Starting training: {training_config.epochs} epochs, "
            f"{len(self.train_loader)} batches/epoch, device={self.device}",
            flush=True,
        )

        for epoch in range(training_config.epochs):

            train_metrics = self._train_epoch(epoch)
            final_metrics = train_metrics

            print(
                f"[epoch {epoch + 1}/{training_config.epochs}] "
                f"train: {train_metrics}",
                flush=True,
            )

            if self.validation_loader is not None:

                validation_metrics = self.evaluator.evaluate(
                    self.validation_loader
                )

                final_metrics = {
                    **train_metrics,
                    **{
                        f"val_{key}": value
                        for key, value in validation_metrics.items()
                    },
                }

                print(
                    f"[epoch {epoch + 1}/{training_config.epochs}] "
                    f"validation: {validation_metrics}",
                    flush=True,
                )

                if self.mlflow_logger is not None:

                    self.mlflow_logger.log_metrics(
                        {
                            f"val_{key}": value
                            for key, value in validation_metrics.items()
                        },
                        step=self.global_step,
                    )

        self.model.save_model(
            self.experiment.model_path
        )

        self.experiment.mark_complete(
            global_step=self.global_step,
            metrics=final_metrics,
        )

        return final_metrics

    def _train_epoch(self, epoch: int) -> dict[str, float]:

        training_config = self.config.training

        self.model.train()

        total_loss = 0.0
        total_batches = 0

        num_batches = len(self.train_loader)

        # Print ~10 progress lines per epoch so long runs never look
        # stalled - this loop otherwise produces no output between
        # epochs, which has been mistaken for a hang more than once.
        log_every = max(1, num_batches // 10)

        self.optimizer.zero_grad()

        for batch_index, batch in enumerate(self.train_loader):

            loss = self._train_step(batch)

            scaled_loss = (
                loss / training_config.gradient_accumulation_steps
            )

            self.scaler.scale(scaled_loss).backward()

            is_accumulation_boundary = (
                (batch_index + 1)
                % training_config.gradient_accumulation_steps
                == 0
            )

            if is_accumulation_boundary:

                if training_config.gradient_clip_norm is not None:

                    self.scaler.unscale_(self.optimizer)

                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        training_config.gradient_clip_norm,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.optimizer.zero_grad()

                self.global_step += 1

                should_checkpoint = (
                    training_config.save_every_n_steps is not None
                    and self.global_step
                    % training_config.save_every_n_steps
                    == 0
                )

                if should_checkpoint:
                    self._save_checkpoint(epoch)

                if self.mlflow_logger is not None:

                    self.mlflow_logger.log_metric(
                        "train_loss",
                        loss.item(),
                        step=self.global_step,
                    )

            total_loss += loss.item()
            total_batches += 1

            if (batch_index + 1) % log_every == 0:

                print(
                    f"  epoch {epoch + 1} "
                    f"batch {batch_index + 1}/{num_batches} "
                    f"avg_loss={total_loss / total_batches:.4f}",
                    flush=True,
                )

        return {
            "train_loss": total_loss / max(total_batches, 1)
        }

    def _train_step(self, batch: dict) -> torch.Tensor:

        pad_token_id = self.tokenizer.pad_token_id

        # non_blocking only overlaps the copy with compute when the
        # source is pinned memory, which the loaders do for CUDA.
        non_blocking = self.device.type == "cuda"

        batch = {
            key: value.to(self.device, non_blocking=non_blocking)
            for key, value in batch.items()
        }

        source_ids = batch["source_ids"]
        target_ids = batch["target_ids"]

        # Teacher forcing: decoder sees all but the last target token,
        # and is scored against all but the first (the shifted labels).
        decoder_input_ids = target_ids[:, :-1]
        labels = target_ids[:, 1:]

        source_key_padding_mask = (
            source_ids == pad_token_id
        )

        target_key_padding_mask = (
            decoder_input_ids == pad_token_id
        )

        # Packed rows (see PackedSeq2SeqDataset) add segment ids so
        # examples sharing a row can't attend to each other.
        segment_kwargs = {}

        if "source_segment_ids" in batch:

            target_segment_ids = batch["target_segment_ids"]

            segment_kwargs = {
                "source_segment_ids": batch["source_segment_ids"],
                "target_segment_ids": target_segment_ids[:, :-1],
            }

            # Where the label belongs to the next example, the input's
            # "next token" is that example's <BOS> - not a prediction
            # worth learning.
            labels = labels.masked_fill(
                target_segment_ids[:, :-1] != target_segment_ids[:, 1:],
                pad_token_id,
            )

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.mixed_precision,
        ):

            output = self.model(
                source_ids=source_ids,
                target_ids=decoder_input_ids,
                source_key_padding_mask=source_key_padding_mask,
                target_key_padding_mask=target_key_padding_mask,
                **segment_kwargs,
            )

            vocab_size = output.logits.size(-1)

            loss = nn.functional.cross_entropy(
                output.logits.reshape(-1, vocab_size),
                labels.reshape(-1),
                ignore_index=pad_token_id,
            )

        return loss

    def _save_checkpoint(self, epoch: int) -> None:

        self.checkpoint_manager.save(
            self.experiment.checkpoint_path(
                f"step_{self.global_step}"
            ),
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            global_step=self.global_step,
            keep_last_n=self.config.training.save_total_limit,
        )
