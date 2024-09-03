import torch
import triton
import triton.language as tl
import transformer_engine

@triton.jit
def rope_forward(
    t_ptr,
    freqs_ptr,
    output_ptr,
    t_stride,
    t_dimension,
    n_elements,
    NUM_SEQUENCE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    t_start = pid * t_dimension

    t_left_offsets = t_start + tl.arange(0, BLOCK_SIZE // 2)
    t_right_offsets = t_start + tl.arange(BLOCK_SIZE // 2, BLOCK_SIZE)

    t_left_masks = (t_left_offsets < n_elements)
    t_right_masks = (t_right_offsets < n_elements)

    freqs_start = (pid // t_stride) * BLOCK_SIZE
    freqs_left_offsets = freqs_start + tl.arange(0, BLOCK_SIZE // 2)
    freqs_right_offsets = freqs_start + tl.arange(BLOCK_SIZE // 2, BLOCK_SIZE)

    freqs_left_masks = (freqs_left_offsets < NUM_SEQUENCE * BLOCK_SIZE)
    freqs_right_masks = (freqs_right_offsets < NUM_SEQUENCE * BLOCK_SIZE)

    t_left = tl.load(t_ptr + t_left_offsets, mask=t_left_masks)
    t_right = tl.load(t_ptr + t_right_offsets, mask=t_right_masks)
    freqs_left = tl.load(freqs_ptr + freqs_left_offsets, mask=freqs_left_masks)
    freqs_right = tl.load(freqs_ptr + freqs_right_offsets, mask=freqs_right_masks)

    cos_freqs_left = tl.cos(freqs_left)
    cos_freqs_right = tl.cos(freqs_right)
    sin_freqs_left = tl.sin(freqs_left)
    sin_freqs_right = tl.sin(freqs_right)

    output = t_left * cos_freqs_left - t_right * sin_freqs_left
    tl.store(output_ptr + t_left_offsets, output, mask=t_left_masks)

    output = t_left * sin_freqs_right + t_right * cos_freqs_right
    tl.store(output_ptr + t_right_offsets, output, mask=t_right_masks)


@triton.jit
def rope_backward(
    t_ptr,
    freqs_ptr,
    output_ptr,
    t_stride,
    t_dimension,
    n_elements,
    NUM_SEQUENCE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    t_start = pid * t_dimension

    t_left_offsets = t_start + tl.arange(0, BLOCK_SIZE // 2)
    t_right_offsets = t_start + tl.arange(BLOCK_SIZE // 2, BLOCK_SIZE)

    t_left_masks = (t_left_offsets < n_elements)
    t_right_masks = (t_right_offsets < n_elements)

    freqs_start = (pid // t_stride) * BLOCK_SIZE
    freqs_left_offsets = freqs_start + tl.arange(0, BLOCK_SIZE // 2)
    freqs_right_offsets = freqs_start + tl.arange(BLOCK_SIZE // 2, BLOCK_SIZE)

    freqs_left_masks = (freqs_left_offsets < NUM_SEQUENCE * BLOCK_SIZE)
    freqs_right_masks = (freqs_right_offsets < NUM_SEQUENCE * BLOCK_SIZE)

    t_left = tl.load(t_ptr + t_left_offsets, mask=t_left_masks)
    t_right = tl.load(t_ptr + t_right_offsets, mask=t_right_masks)
    freqs_left = tl.load(freqs_ptr + freqs_left_offsets, mask=freqs_left_masks)
    freqs_right = tl.load(freqs_ptr + freqs_right_offsets, mask=freqs_right_masks)

    cos_freqs_left = tl.cos(freqs_left)
    cos_freqs_right = tl.cos(freqs_right)
    sin_freqs_left = tl.sin(freqs_left)
    sin_freqs_right = tl.sin(freqs_right)

    output = cos_freqs_left * t_left + sin_freqs_right *  t_right
    tl.store(output_ptr + t_left_offsets, output, mask=t_left_masks)

    output = cos_freqs_right * t_right - t_left * sin_freqs_left
    tl.store(output_ptr + t_right_offsets, output, mask=t_right_masks)
