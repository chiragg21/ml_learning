from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
import torch
from torch import nn


##Encoder

class BaseEncoderLayer(nn.Module, ABC):
    def __init__(
            self,
            config,
        ):
        super().__init__()


    @abstractmethod
    def forward(self):
        pass


class BaseEncoder(nn.Module, ABC):
    def __init__(
            self,
            config,
        ):
        super().__init__()

    @abstractmethod
    def forward(self):
        pass

## Decoder

class BaseDecoderLayer(nn.Module, ABC):
    def __init__(
            self,
            config
    ):
        super().__init__()

    @abstractmethod
    def forward(self):
        pass

class BaseDecoder(nn.Module, ABC):
    def __init__(
            self,
            config,
        ):
        super().__init__()

    @abstractmethod
    def forward(self):
        pass


## tranformer

class BaseTransformer(nn.Module, ABC):

    def __init__(
            self,
            config,
        ):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        source_ids: torch.Tensor,
        target_ids: torch.Tensor,
        **kwargs,
    ) -> Any:
        pass


        # --------------------------------------------------
    # Model persistence
    # --------------------------------------------------

    def save_model(
        self,
        path: str | Path,
    ) -> None:

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            self.state_dict(),
            path,
        )

    def load_model(
        self,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> None:

        path = Path(path)

        state_dict = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )

        self.load_state_dict(state_dict)

    # --------------------------------------------------
    # Utility
    # --------------------------------------------------

    def count_parameters(
        self,
        trainable_only: bool = True,
    ) -> int:

        parameters = self.parameters()

        if trainable_only:
            return sum(
                parameter.numel()
                for parameter in parameters
                if parameter.requires_grad
            )

        return sum(
            parameter.numel()
            for parameter in parameters
        )