from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

"""
Post-training weight quantization: store a trained model's Linear
weights as 8-bit or 4-bit integers instead of 32-bit floats, trading a
little accuracy for 4x (int8) or ~8x (int4) smaller weights.

Scheme: symmetric, weight-only, per-group.
  - Each row of a weight matrix (one output unit) is split into groups
    of `group_size` consecutive input weights (default: the whole row is
    one group, i.e. per-output-channel quantization).
  - Each group gets one float scale = max|w| / qmax, where qmax is 127
    (int8) or 7 (int4), and every weight becomes round(w / scale) in
    [-qmax, qmax].
  - At run time the integers are multiplied back by their scale
    (dequantized) and the layer runs as an ordinary float matmul.
    Activations stay float, so this saves *storage* and memory traffic;
    it does not make the arithmetic itself integer (real int8 speedups
    need integer kernels, which plain PyTorch doesn't give us).

Smaller groups follow the weights more closely (so int4 usually wants
group_size 32-128) at the cost of storing more scales. Int4 values are
packed two per byte.

What is deliberately NOT quantized (`skip`): the tied embedding/output
matrix (it is also the lookup table for the input and errors there hit
every token twice), and the small routing layers of v3 (MoE router,
depth router), whose tiny weights decide *which* computation happens
and are the most quantization-sensitive part of the model.

A quantized model is for inference only: its integer buffers have no
gradients. Save it with state_dict() and restore by calling
quantize_model() with the same settings on a freshly built model first,
then load_state_dict().
"""

DEFAULT_SKIP = ("llm_head", "router", "depth_router")


class QuantizedLinear(nn.Module):

    def __init__(
        self,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        in_features: int,
        out_features: int,
        bits: int,
        group_size: int,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size

        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales)

    # ==================================================
    # Quantize / dequantize
    # ==================================================

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int = 8,
        group_size: int | None = None,
    ) -> "QuantizedLinear":

        if bits not in (4, 8):
            raise ValueError("bits must be 4 or 8")

        if linear.bias is not None:
            raise ValueError("QuantizedLinear supports bias-free layers only")

        out_features, in_features = linear.weight.shape

        group_size = group_size or in_features

        if in_features % group_size != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by "
                f"group_size ({group_size})"
            )

        if bits == 4 and group_size % 2 != 0:
            raise ValueError("int4 needs an even group_size to pack pairs")

        qmax = 2 ** (bits - 1) - 1

        weight = linear.weight.detach().float()
        grouped = weight.view(out_features, in_features // group_size, group_size)

        scales = (grouped.abs().amax(dim=-1) / qmax).clamp(min=1e-8)

        quantized = torch.round(grouped / scales[..., None]).clamp(-qmax, qmax)
        quantized = quantized.view(out_features, in_features).to(torch.int8)

        if bits == 4:
            quantized = cls._pack_int4(quantized)

        return cls(quantized, scales, in_features, out_features, bits, group_size)

    @staticmethod
    def _pack_int4(values: torch.Tensor) -> torch.Tensor:

        # shift [-7, 7] to [1, 15] so each value fits an unsigned nibble
        unsigned = (values + 8).to(torch.uint8)

        return unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)

    @staticmethod
    def _unpack_int4(packed: torch.Tensor) -> torch.Tensor:

        low = (packed & 0x0F).to(torch.int8) - 8
        high = (packed >> 4).to(torch.int8) - 8

        return torch.stack([low, high], dim=-1).flatten(start_dim=1)

    def dequantize(self) -> torch.Tensor:

        values = (
            self._unpack_int4(self.qweight)
            if self.bits == 4
            else self.qweight
        )

        grouped = values.view(
            self.out_features,
            self.in_features // self.group_size,
            self.group_size,
        ).float()

        return (grouped * self.scales[..., None]).view(
            self.out_features,
            self.in_features,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return F.linear(x, self.dequantize().to(x.dtype))

    def extra_repr(self) -> str:

        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, group_size={self.group_size}"
        )


# ==================================================
# Whole-model helpers
# ==================================================

def quantize_model(
    model: nn.Module,
    bits: int = 8,
    group_size: int | None = None,
    skip: tuple[str, ...] = DEFAULT_SKIP,
) -> nn.Module:
    """
    Returns a quantized *copy* of model (the original is untouched, so
    float and quantized versions can be compared). Every bias-free
    nn.Linear whose qualified name contains none of `skip` is replaced
    by a QuantizedLinear.
    """

    quantized = copy.deepcopy(model).eval()

    # Collect first: replacing modules while iterating named_modules()
    # would mutate the thing being walked.
    targets = [
        (name, module)
        for name, module in quantized.named_modules()
        if isinstance(module, nn.Linear)
        and not any(pattern in name for pattern in skip)
    ]

    for name, linear in targets:

        parent_name, _, child_name = name.rpartition(".")

        parent = quantized.get_submodule(parent_name) if parent_name else quantized

        setattr(
            parent,
            child_name,
            QuantizedLinear.from_linear(linear, bits=bits, group_size=group_size),
        )

    for parameter in quantized.parameters():
        parameter.requires_grad_(False)

    return quantized


def model_size_bytes(model: nn.Module) -> int:
    """
    Bytes held by parameters and buffers. Tensors that share storage
    (the tied embedding/output matrix) are counted once.
    """

    seen: set[int] = set()
    total = 0

    for tensor in [*model.parameters(), *model.buffers()]:

        key = tensor.untyped_storage().data_ptr()

        if key in seen:
            continue

        seen.add(key)
        total += tensor.untyped_storage().nbytes()

    return total


def count_quantized_layers(model: nn.Module) -> int:

    return sum(isinstance(m, QuantizedLinear) for m in model.modules())
