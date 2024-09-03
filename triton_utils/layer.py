import torch
import triton
import triton.language as tl
import transformer_engine

from loguru import logger
from triton_utils import kernel
from typing import Union

class TritonRotaryPositionEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,
        tensor_format: str = "sbhd",
    ) -> torch.Tensor:
        match tensor_format:
            case "sbhd":
                pass
            case "bshd":
                t = t.transpose(0, 1).contiguous()
            case _:
                raise NotImplementedError(
                    "Only formats `sbhd` or `bshd` are supported for input tensor `t` "
                    f"when fused is False, got {tensor_format}."
                )

        max_seq_len = freqs.shape[0]
        cur_seq_len = t.shape[0]

        assert cur_seq_len <= max_seq_len, (
            f"Rotary Embeddings only supported up to {max_seq_len} sequence length!"
        )
        freqs = freqs[:cur_seq_len]

        n_elements = t.numel()
        grid = lambda meta: (triton.cdiv(t.numel(), meta["t_dimension"]),)
        output = t.clone()
        t_dimension = t.shape[3]
        t_stride = t.shape[1] * t.shape[2]

        kernel.rope_forward[grid](
            t_ptr=t,
            freqs_ptr=freqs,
            output_ptr=output,
            t_stride=t_stride,
            t_dimension=t_dimension,
            n_elements=n_elements,
            NUM_SEQUENCE=cur_seq_len,
            BLOCK_SIZE=freqs.shape[-1],
        )

        if tensor_format == "bshd":
            output = output.transpose(0, 1).contiguous()

        ctx.save_for_backward(freqs)
        ctx.tensor_format=tensor_format

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        freqs, = ctx.saved_tensors
        match ctx.tensor_format:
            case "sbhd":
                pass
            case "bshd":
                grad_output = grad_output.transpose(0, 1)
            case _:
                raise NotImplementedError(
                    "Only formats `sbhd` or `bshd` are supported for input tensor `t` "
                    f"when fused is False, got {ctx.tensor_format}."
                )


        grad_output = grad_output.contiguous()
        grad_input = grad_output.clone().detach()
        t_stride = grad_output.shape[1] * grad_output.shape[2]
        t_dimension = grad_output.shape[3]
        n_elements = grad_output.numel()
        cur_seq_len = grad_output.shape[0]
        grid = lambda meta: (triton.cdiv(n_elements, meta["t_dimension"]),)
        kernel.rope_backward[grid](
            t_ptr=grad_output,
            freqs_ptr=freqs,
            output_ptr=grad_input,
            t_stride=t_stride,
            t_dimension=t_dimension,
            n_elements=n_elements,
            NUM_SEQUENCE=cur_seq_len,
            BLOCK_SIZE=freqs.shape[-1],
        )

        if ctx.tensor_format == "bshd":
            grad_input = grad_input.transpose(0, 1).contiguous()
        return grad_input, None, None, None
