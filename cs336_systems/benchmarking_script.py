import torch
from torch import Tensor
from cs336_basics.model import BasicsTransformerLM, softmax
from einops import einsum
from jaxtyping import Float, Bool, Int
from torch_util import get_device
# from torch.utils import benchmark
import timeit
from typing import Callable
from torch.cuda import nvtx
import math


@nvtx.range('scaled dot product attention')
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = K.shape[-1]
    with nvtx.range('computing attention scores'):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range('computing softmax'):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range('final matmul'):
        res = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    return res


# cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
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


def run_model(mtype='small', step=1, is_back=True, is_opt=True):
    p_dic = params_dic[mtype]
    d_model, d_ff, num_layers, num_heads = p_dic['d_model'], p_dic['d_ff'], p_dic['num_layers'], p_dic['num_heads']
    global vocab_size, batch_size, context_length, rope_theta
    if not torch.cuda.is_available():
        vocab_size //= 4
        context_length //= 4
        d_model //= 4
        num_layers //= 4
        print(mtype, 'no_cuda', vocab_size, context_length, d_model, num_layers)

    with nvtx.range('define_model'):
        model = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta).to(get_device())
    with nvtx.range('define_input'):
        x = torch.randint(0, vocab_size, (batch_size, context_length), device=get_device())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) if is_opt else None

    def run(is_back=True):
        for t in range(step):
            if t > 10:
                torch.cuda.cudart().cudaProfilerStart()
            nvtx.range_push(f'step_{t}')
            if is_opt and optimizer:
                optimizer.zero_grad()
            else:
                model.zero_grad()

            with nvtx.range('forward'):
                y = model(x).mean()

            with nvtx.range('backward'):
                if is_back:
                    y.backward()

            with nvtx.range('optimizer'):
                if is_opt and optimizer:
                    optimizer.step()

            nvtx.range_pop()

    return run


def benchmarking_fun(fun: Callable, warm_num=3, trial_num=5, is_back=True):
    for _ in range(warm_num):
        fun(is_back)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    st = timeit.default_timer()
    for _ in range(trial_num):
        fun(is_back)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    et = timeit.default_timer()
    return (et - st) / trial_num


def test_model_benchmark(mtype='small', step=3, warm_num=3, trial_num=5, is_back=True, is_opt=True):
    fun = run_model(mtype, step, is_back, is_opt)
    mtime = benchmarking_fun(fun, warm_num, trial_num, is_back)
    res_str = (
        f'model_type={mtype:<8}, step={step:<3}, warm_num={warm_num:<3}, trial_num={trial_num:<3}, '
        f'is_back={is_back:<2}, is_opt={is_opt:<2}, mtime={mtime:.6g}s'
    )
    print(res_str)
    return res_str


def benchmarking_test01():
    res1_list = [test_model_benchmark(is_back=t1, is_opt=t2) for t1 in [True, False] for t2 in [True, False]]
    # res2_list = [test_model_benchmark(is_opt=t) for t in [True, False]]
    res3_list = [test_model_benchmark(step=s) for s in range(3, 31, 3)]
    res4_list = [test_model_benchmark(warm_num=w) for w in [0, 1, 3, 5]]
    res5_list = [test_model_benchmark(mtype=m) for m in ['small', 'medium', 'large', 'xl', '2.7B'][:2]]


def profiling_test01():
    test_model_benchmark(mtype='medium', step=50, warm_num=0, trial_num=1)


if __name__ == '__main__':
    # benchmarking_test01()
    profiling_test01()


