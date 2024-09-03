import torch
import transformer_engine
from triton_utils import layer
from triton_utils import test_utils

from typing import Union
from absl.testing import absltest
from absl.testing import parameterized

def _loss_fn(output: torch.Tensor) -> torch.Tensor:
    t = torch.ones_like(output)
    return torch.sum(output * t)

class TritonLayerTest(parameterized.TestCase):
    @parameterized.parameters(
        (torch.randn((2048, 2, 64, 128)), torch.randn((2048, 1, 1, 128))),
        (torch.randn((512, 2, 64, 256)), torch.randn((512, 1, 1, 128))),
        (torch.randn((1024, 2, 64, 512)), torch.randn((1024, 1, 1, 512))),
        (torch.randn((4096, 2, 64, 384)), torch.randn((4096, 1, 1, 256))),
        (torch.randn((2, 2048, 64, 128)), torch.randn((2048, 1, 1, 128)), "bshd"),
        (torch.randn((2, 512, 64, 256)), torch.randn((512, 1, 1, 128)), "bshd"),
        (torch.randn((2, 1024, 64, 512)), torch.randn((1024, 1, 1, 512)), "bshd"),
        (torch.randn((2, 4096, 64, 384)), torch.randn((4096, 1, 1, 256)), "bshd"),
    )
    def test_triton_apply_rotary_pos_emb(
        self,
        t: torch.Tensor,
        freqs: torch.Tensor,
        tensor_format: str = "sbhd",
        cu_seqlens: Union[torch.Tensor, None] = None,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float32,
    ):
        t = test_utils.cast_tensor(t, device=device, dtype=dtype)
        freqs = test_utils.cast_tensor(freqs, device=device, dtype=dtype)

        expected = test_utils.get_callable_by_provider(provider=test_utils.Provider.TORCH)(
            t=t,
            freqs=freqs,
            tensor_format=tensor_format,
            cu_seqlens=cu_seqlens,
        )
        loss_expected = _loss_fn(expected)
        loss_expected.backward()
        expected_grad = t.grad.detach().clone()

        t.grad = None
        actual = test_utils.get_callable_by_provider(provider=test_utils.Provider.TRITON)(
            t,
            freqs,
            tensor_format,
        )

        loss_actual = _loss_fn(actual)
        loss_actual.backward()
        actual_grad = t.grad.detach().clone()

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_grad, expected_grad)

if __name__ == "__main__":
    absltest.main()
