import timeit
import torch
from cs336_systems.torch_util import *
from tests.adapters import (
    ddp_bucketed_on_after_backward,
    ddp_bucketed_on_train_batch_start,
    get_ddp_bucketed,
)
from cs336_basics.model import BasicsTransformerLM


params_dic = {
    'small': {'d_model': 768, 'd_ff': 3072, 'num_layers': 12, 'num_heads': 12},
    'medium': {'d_model': 1024, 'd_ff': 4096, 'num_layers': 24, 'num_heads': 16},
    'large': {'d_model': 1280, 'd_ff': 5120, 'num_layers': 36, 'num_heads': 20},
    'xl': {'d_model': 1600, 'd_ff': 6400, 'num_layers': 48, 'num_heads': 25},
    '2.7B': {'d_model': 2560, 'd_ff': 10240, 'num_layers': 32, 'num_heads': 32},
}
model_type = 'small'

d_model = params_dic[model_type]['d_model']
num_layers = params_dic[model_type]['num_layers']
num_heads = params_dic[model_type]['num_heads']
d_ff = params_dic[model_type]['d_ff']
vocab_size = 10000
batch_size = 4
context_length = 256
rope_theta = 10000

rank = 2
world_size = 2
base_model = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers,\
                                 num_heads, d_ff, rope_theta).to(get_device())

def worker(rank, world_size, model, bucket_size):
    setup(rank, world_size)
    ddp_model = get_ddp_bucketed(model, bucket_size)
    try:
        x = torch.randn((batch_size, context_length, d_model), device=get_device())
        optimzer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)
        for _ in range(5):
            optimzer.zero_grad()
            y = ddp_model(x)
            loss = torch.square(y).mean()
            loss.backward()
            ddp_model.finish_gradient_synchronization()
            optimzer.step()
    except Exception as e:
        print(f'except in rank={rank}')
        print(e)
    finally:
        cleanup()

def benchmark_each(bucket_size):
    mp.spawn(
        fn=worker,
        args=(world_size, base_model, bucket_size),
        nprocs=world_size,
        join=True,
    )

def benchmark_all():
    for bucket_size in [1, 10, 100, 1000][:1]:
        print(f'\nbucket_size={bucket_size}MB, start')
        st = timeit.default_timer()
        benchmark_each(bucket_size)
        et = timeit.default_timer()
        print(f'bucket_size={bucket_size}MB, cost={(et-st):.6g}s')
    pass

if __name__ == '__main__':
    benchmark_all()



