import torch
from contextlib import contextmanager, nullcontext
import timeit


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



