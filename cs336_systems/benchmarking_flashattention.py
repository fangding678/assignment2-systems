import torch
from torch import Tensor
from cs336_basics.model import BasicsTransformerLM, softmax
from cs336_basics.model import scaled_dot_product_attention
from einops import einsum
from jaxtyping import Float, Bool, Int
from torch_util import get_device, precision_context
# from torch.utils import benchmark
import timeit
import math
import triton
import triton.language as tl


params_dic = {
    'small': {'d_model': 768, 'd_ff': 3072, 'num_layers': 12, 'num_heads': 12},
    'medium': {'d_model': 1024, 'd_ff': 4096, 'num_layers': 24, 'num_heads': 16},
    'large': {'d_model': 1280, 'd_ff': 5120, 'num_layers': 36, 'num_heads': 20},
    'xl': {'d_model': 1600, 'd_ff': 6400, 'num_layers': 48, 'num_heads': 25},
    '2.7B': {'d_model': 2560, 'd_ff': 10240, 'num_layers': 32, 'num_heads': 32},
}
vocab_size = 10000
batch_size = 4
context_length = 256
rope_theta = 10000

d_model_list = [16, 32, 64, 128]
context_length_list = [256, 1024, 4096, 8192, 16384]


def benchmark_each(d_model, context_length, dt, jit, bs=8, warm_num=10, step=100):
    qkv = torch.randn((batch_size, context_length, d_model * 3), dtype=dt, device=get_device())
    Q, K, V = torch.chunk(qkv, 3, dim=-1)
    if jit:
        atten = torch.compile(scaled_dot_product_attention)
    else:
        atten = scaled_dot_product_attention
    for _ in range(warm_num):
        atten(Q, K, V)
    torch.cuda.synchronize()
    st = timeit.default_timer()
    for _ in range(step):
        atten(Q, K, V)
    torch.cuda.synchronize()
    et = timeit.default_timer()
    res_str = (
                f'jit={jit:<1},dt={str(dt):<13},d_model={d_model:.6g},context_length={context_length:.6g},' 
                f'step={step:<3},mtime={et-st:.6g}s'
            )
    print(res_str)
    return f'{jit},{str(dt)},{d_model},{context_length},{step},{et-st:.6g}s,' 


def test_compile_performance():
    start_ts = timeit.default_timer()
    res_list = ['jit,dtype,d_model,context_length,step,time']
    for jit in [False, True]:
        for dt in [torch.float32, torch.bfloat16, torch.float16]:
            for d_model in d_model_list:
                for context_length in context_length_list:
                    res = benchmark_each(d_model, context_length, dt, jit)
                    res_list.append(res)
    end_ts = timeit.default_timer()
    print('\n'.join(res_list))
    print(f'test_compile_performance total cost {end_ts - start_ts:.6g}s')


if __name__ == '__main__':
    test_compile_performance()
    pass


