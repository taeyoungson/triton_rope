import triton
import torch
import transformer_engine
from loguru import logger
from triton_utils import layer
from triton_utils import test_utils


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["dimension_size"],
        x_vals=[2**i for i in range(4, 14, 1)],
        line_arg="provider",
        line_vals=[test_utils.Provider.TRITON, test_utils.Provider.TORCH, test_utils.Provider.CUDA],
        line_names=["Triton", "Torch", "Cuda"],
        styles=[('blue', '-'), ('green', '-'), ('orange', '-')],
        ylabel="ms",
        plot_name="RoPE performance",
        args={
            "sequence_size": 2048,
            "batch_size": 2,
            "head_size": 64,
            "rotary_percent": 0.5,
            "tensor_format": "sbhd",
        },
    ))
def benchmark(
    dimension_size: int,
    sequence_size: int,
    batch_size: int,
    head_size: int,
    provider: test_utils.Provider,
    mode: str = "forward",
    rotary_percent: float = 1.0,
    tensor_format: str ="sbhd",
    device: torch.device = torch.device("cuda"),
):
    t = torch.rand(
        (sequence_size, batch_size, head_size, dimension_size),
        dtype=torch.float32,
        device=device,
        requires_grad=True
    )
    quantiles = [0.5, 0.2, 0.8]

    if tensor_format == "bshd":
        t = t.transpose(0, 1).contiguous()

    rotary_pos_emb = transformer_engine.pytorch.attention.RotaryPositionEmbedding(
        dimension_size,
        rotary_percent,
    )
    emb = rotary_pos_emb(sequence_size)

    # forward pass
    if mode == 'forward':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: test_utils.get_callable_by_provider(provider)(t, emb, tensor_format), quantiles=quantiles)

    if mode == 'backward':
        y = test_utils.get_callable_by_provider(provider)(t, emb, tensor_format)
        dy = torch.randn_like(y)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: y.backward(dy, retain_graph=True), quantiles=quantiles, grad_to_none=[t])

    dur = lambda ms: ms
    return dur(ms), dur(max_ms), dur(min_ms)

benchmark.run(save_path="./results", print_data=True)
