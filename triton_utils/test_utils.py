import torch
import enum
import functools
import transformer_engine

from triton_utils import layer
from typing import Callable


class Provider(enum.Enum):
    CUDA = enum.auto()
    TORCH = enum.auto()
    TRITON = enum.auto()


def cast_tensor(
    t: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool = True,
) -> torch.Tensor:
    t = t.to(
        device=device,
        dtype=dtype,
    ).requires_grad_(requires_grad)
    t.retain_grad()
    return t


def get_callable_by_provider(provider: Provider) -> Callable:
    match provider:
        case Provider.CUDA:
            return functools.partial(
                transformer_engine.pytorch.attention.apply_rotary_pos_emb,
                fused=True,
            )
        case Provider.TORCH:
            return functools.partial(
                transformer_engine.pytorch.attention.apply_rotary_pos_emb,
                fused=False,
            )
        case Provider.TRITON:
            return layer.TritonRotaryPositionEmbedding.apply
        case _:
            raise NotImplementedError

