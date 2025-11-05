import torch
from torch import Tensor
from cs336_basics.model import BasicsTransformerLM, softmax
from einops import einsum
from jaxtyping import Float, Bool, Int
from torch_util import get_device, precision_context
# from torch.utils import benchmark
import timeit
from typing import Callable
from torch.cuda import nvtx
import math
from contextlib import contextmanager, nullcontext
from cs336_basics.model import scaled_dot_product_attention


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


def run_test01(d_model=16, context_length=256, is_back=True, step=50):
    bs = 8
    x = torch.randn((bs, context_length, d_model), device=get_device())
    proj_qkv = torch.nn.Linear(d_model, 3 * d_model, device=get_device())
    # print(f'torch.cuda.is_available()={torch.cuda.is_available()}')

    optimizer = torch.optim.Adam(proj_qkv.parameters(), lr=1e-4) if is_back else None

    def run():
        for t in range(step):
            if optimizer:
                optimizer.zero_grad()
            else:
                proj_qkv.zero_grad()
            Q, K, V = torch.chunk(proj_qkv(x), 3, dim=-1)
            res = scaled_dot_product_attention(Q, K, V)
            if optimizer:
                res.mean().backward()
                optimizer.step()

    return run


def benchmarking_fun(fun: Callable, warm_num=3, trial_num=5):
    for _ in range(warm_num):
        fun()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    st = timeit.default_timer()
    for _ in range(trial_num):
        fun()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    et = timeit.default_timer()
    return (et - st) / trial_num


def test_flashattention_benchmark(step=30, warm_num=3, trial_num=5, is_back=True):
    res_dic = {}
    res_print = ['d_model,context_length,is_back,step,warm_num,trial_num,mtime']
    for d_model in d_model_list:
        for context_length in context_length_list:
            context_length //= 2
            fun = run_test01(d_model, context_length, is_back=is_back, step=step)
            mtime = benchmarking_fun(fun, warm_num, trial_num)
            res_str = (
                f'd_model={d_model:.6g}, context_length={context_length:.6g}, is_back={is_back:<2}, step={step:<3},' 
                f'warm_num={warm_num:<3}, trial_num={trial_num:<3}, mtime={mtime:.6g}s'
            )
            print(res_str)
            res_dic[f'{d_model}_{context_length}'] = res_dic
            res_print.append(f'{d_model},{context_length},{is_back},{step},{warm_num},{trial_num},{mtime}')
        print('-' * 50)
    print('=' * 100)
    print('\n'.join(res_print))
    return res_dic


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
    # test_flashattention_benchmark(step=50, is_back=False)
    test_flashattention_benchmark(step=500, is_back=True)
    # test_compile_performance()
    pass


