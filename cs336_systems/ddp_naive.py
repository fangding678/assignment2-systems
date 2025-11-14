import torch

from torch_util import *


def generate_data():
    torch.random.manual_seed(0)
    batch_size = 128
    num_dim = 1024
    data = torch.randn(batch_size, num_dim)
    return data


def data_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_steps: int):
    setup(rank, world_size)
    batch_size, num_dim = data.shape
    local_bs = (batch_size + world_size - 1) // world_size
    start_bs_index = local_bs * rank
    end_bs_index = min(local_bs * (rank + 1), batch_size)
    data_split = data[start_bs_index: end_bs_index].to(get_device(rank))
    params = [get_init_params(num_dim, num_dim, rank) for _ in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)  # Each rank has own optimizer state

    for step in range(num_steps):
        optimizer.zero_grad()
        x = data_split
        for param in params:
            x = F.relu(x @ param)
        loss = x.square().mean()
        loss.backward()
        for param in params:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.SUM, async_op=False)
            param.grad.data.div_(world_size)
        optimizer.step()
        # for param in params:
        #     if rank == 0:
        #         print(f'step={step}, rank={rank}', param)

    if rank == 0:
        non_parallel_data = data
        non_parallel_params = [get_init_params(num_dim, num_dim, rank) for _ in range(num_layers)]
        non_parallel_optimizer = torch.optim.AdamW(non_parallel_params, lr=1e-3)
        for step in range(num_steps):
            non_parallel_optimizer.zero_grad()
            non_parallel_x = non_parallel_data
            for non_parallel_param in non_parallel_params:
                non_parallel_x = F.relu(non_parallel_x @ non_parallel_param)
            loss = non_parallel_x.square().mean()
            loss.backward()
            non_parallel_optimizer.step()

        for non_parallel_param, param in zip(non_parallel_params, params):
            print()
            # print(type(non_parallel_param))
            print(non_parallel_param)
            print(param)
            print(non_parallel_param.data == param.data)
            assert torch.allclose(non_parallel_param.data, param.data, rtol=1e4, atol=1e5)
        print('#' * 20, f'{rank} test ending', '#' * 20)

    cleanup()


def test_naive_ddp():
    data = generate_data()
    spawn(data_parallelism_main, world_size=4, data=data, num_layers=2, num_steps=2)


if __name__ == '__main__':
    test_naive_ddp()
    pass


