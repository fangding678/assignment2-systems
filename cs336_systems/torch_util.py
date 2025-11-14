import torch
from contextlib import contextmanager, nullcontext
import timeit
from typing import List, Callable
import torch.multiprocessing as mp
import torch.nn.functional as F
import os, sys
import torch.distributed as dist
from inspect import isfunction
import math
from copy import deepcopy


def get_device(index: int = 0) -> torch.device:
    """Try to use the GPU if possible, otherwise, use CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")


@contextmanager
def precision_context(device_type='cuda', dtype=torch.float16):
    print('\n' + '-' * 50)
    print(f'torch.cuda.is_available() = {torch.cuda.is_available()}')
    print('precision_context begin...', device_type, dtype)
    st = timeit.default_timer()
    with torch.autocast(device_type=device_type, dtype=dtype):
        yield
    et = timeit.default_timer()
    print(f'precision_context time={et - st:.6g}s')
    print('precision_context ending...', device_type, dtype)
    print()


def get_gpu_info():
    if torch.cuda.is_available():
        gpu_prop = torch.cuda.get_device_properties(0)
        print(f"GPU 型号: {gpu_prop.name}")
        print(f"SM 数量: {gpu_prop.multi_processor_count}")
    else:
        print("未检测到 GPU")


def render_duration(duration: float) -> str:
    if duration < 1e-3:
        return f"{duration * 1e6:.2f}us"
    if duration < 1:
        return f"{duration * 1e3:.2f}ms"
    return f"{duration:.2f}s"


class DisableDistributed:
    """Context manager that temporarily disables distributed functions (replaces with no-ops)"""
    def __enter__(self):
        self.old_functions = {}
        for name in dir(dist):
            value = getattr(dist, name, None)
            if isfunction(value):
                self.old_functions[name] = value
                setattr(dist, name, lambda *args, **kwargs: None)

    def __exit__(self, exc_type, exc_value, traceback):
        for name in self.old_functions:
            setattr(dist, name, self.old_functions[name])


def get_init_params(num_inputs: int, num_outputs: int, rank: int) -> torch.nn.Parameter:
    torch.random.manual_seed(0)  # For reproducibility
    return torch.nn.Parameter(torch.randn(num_inputs, num_outputs, device=get_device(rank)) / math.sqrt(num_outputs))


def setup(rank: int, world_size: int):
    # Specify where master lives (rank 0), used to coordinate (actual data goes through NCCL)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "15623"
    if torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def spawn(func: Callable, world_size: int, *args, **kwargs):
    # Note: assume kwargs are in the same order as what main needs
    if sys.gettrace():
        # If we're being traced, run the function directly, since we can't trace through mp.spawn
        with DisableDistributed():
            args = (0, world_size,) + args + tuple(kwargs.values())
            func(*args)
    else:
        args = (world_size,) + args + tuple(kwargs.values())
        mp.spawn(func, args=args, nprocs=world_size, join=True)


