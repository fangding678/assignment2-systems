import torch
from cs336_basics.model import BasicsTransformerLM
from torch_util import get_device
# from torch.utils import benchmark
import timeit
from typing import Callable


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


def run_model(mtype='small', step=1, is_back=True):
    p_dic = params_dic[mtype]
    d_model, d_ff, num_layers, num_heads = p_dic['d_model'], p_dic['d_ff'], p_dic['num_layers'], p_dic['num_heads']
    model = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta).to(get_device())
    x = torch.randint(0, vocab_size, (batch_size, context_length), device=get_device())

    def run():
        for _ in range(step):
            y = model(x)
            if is_back:
                y.mean().backward()

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


def test_model_benckmarking(mtype='small', step=3, warm_num=3, trial_num=5, is_back=True):
    test_result = []
    fun = run_model(mtype, step, is_back)
    mtime = benchmarking_fun(fun, warm_num, trial_num)
    test_result.append(((mtype, step, warm_num, trial_num, is_back), mtime))
    res_str = f'model_type={mtype}, step={step}, warm_num={warm_num}, trial_num={trial_num}, is_back={is_back}, mtime={mtime}'
    print(res_str)
    return res_str


if __name__ == '__main__':
    res1_list = []
    for is_back in [True, False]:
        res1_list.append(test_model_benckmarking(is_back=is_back))
    print(res1_list)

    res2_list = []
    for step in range(3, 31, 3):
        res2_list.append(test_model_benckmarking(is_back=is_back))
    print(res2_list)

    res3_list = []
    for w_num in [0, 1, 3, 5]:
        res3_list.append(test_model_benckmarking(warm_num=w_num))
    print(res3_list)

    res4_list = []
    for mtype in ['small', 'medium', 'large', 'xl', '2.7B']:
        res4_list.append(test_model_benckmarking(mtype=mtype))
    print(res4_list)

