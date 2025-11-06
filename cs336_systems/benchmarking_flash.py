import torch
from torch_util import get_device
# from torch.utils import benchmark
import timeit
import math
import numpy as np
import statistics
import triton
from cs336_systems.flashattention import FlashAttentionPytorch,FlashAttentionTriton


batch_size = 4
d_model_list = [16, 32, 64, 128]
context_length_list = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
torch.manual_seed(42)


def test_timing_flash_each(flash_fun, fun_name='pytorch', batch_size=1, context_length=128, d_model=64,
                           dtype=torch.float32):
    q, k, v = torch.randn(3, batch_size, context_length, d_model, device=get_device(),
                          dtype=dtype, requires_grad=True)
    def flash_forward_only():
        return flash_fun(q, k, v, True)

    def flash_forward_backward():
        o = flash_fun(q, k, v, True)
        loss = o.sum()
        loss.backward()

    r = triton.testing.do_bench(flash_forward_only, rep=1000, warmup=200, return_mode='all')
    r_len, r_median, r_mean, r_min, r_max = len(r), statistics.median(r), statistics.mean(r), min(r), max(r)
    res_str = (
        f'fun_name={fun_name}, batch_size={batch_size}, context_length={context_length}, '
        f'd_model={d_model}, dtype={dtype}, device={get_device()}, test_count={r_len}, '
        f'median_time={r_median:.6g}ms, mean_time={r_mean:.6g}ms, min_time={r_min:.6g}ms, max_time={r_max:.6g}ms'
    )
    print(res_str)


def test_timing_flash_forward_backward():
    flash_pytorch = (FlashAttentionPytorch.apply, 'pytorch')
    flash_pytorch2 = (torch.compile(FlashAttentionPytorch.apply), 'pytorch_compile')
    flash_triton = (FlashAttentionTriton.apply, 'triton')
    # test_timing_flash_each(flash_pytorch[0], flash_pytorch[1])
    # test_timing_flash_each(flash_pytorch2[0], flash_pytorch2[1])
    # test_timing_flash_each(flash_triton[0], flash_triton[1])
    # test_timing_flash_each(flash_triton[0], flash_triton[1], 1, 256)
    for fun, fun_name in [flash_pytorch, flash_pytorch2, flash_triton][2:3]:
        for context_length in context_length_list:
            for d_model in d_model_list[:1]:
                for dt in [torch.float32, torch.bfloat16][:1]:
                    if fun_name == 'triton' and dt == torch.bfloat16:
                        continue
                    test_timing_flash_each(fun, fun_name, 1, context_length, d_model, dt)
        print()
    print()


if __name__ == '__main__':
    test_timing_flash_forward_backward()
    pass


