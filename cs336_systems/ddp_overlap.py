import torch
from cs336_systems.torch_util import *


class DDPIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float = 16):
        super().__init__()
        self.handles = []
        self.module = module
        self.bucket_size = 1024 * 1024 * bucket_size_mb
        self.cur_size = 0
        self.grad_buckets = []
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(
                    lambda x=param: self._sync_gradient(x)
                )

    def _sync_gradient(self, p: torch.Tensor):
        if p.grad is not None:
            self.cur_size += p.grad.numel() * p.grad.element_size()
            self.grad_buckets.append(p.grad)
            if self.cur_size > self.bucket_size:
                self._sync_gradient_in_buckets()

    def _sync_gradient_in_buckets(self):
        if not self.grad_buckets:
            return
        flatten_grad = torch._utils._flatten_dense_tensors(self.grad_buckets)
        dist_op = dist.ReduceOp.SUM if dist.get_backend() == dist.Backend.GLOO else dist.ReduceOp.AVG
        handle = dist.all_reduce(flatten_grad, op=dist_op, async_op=True)
        self.handles.append((handle, flatten_grad, list(self.grad_buckets)))
        self.cur_size = 0
        self.grad_buckets.clear()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        self._sync_gradient_in_buckets()
        for handle, flatten_grad, grad_buckets in self.handles:
            handle.wait()
            unflatten_grad = torch._utils._unflatten_dense_tensors(flatten_grad, grad_buckets)
            for grad, sync_grad in zip(grad_buckets, unflatten_grad):
                grad.copy_(sync_grad)
                if dist.get_backend() == dist.Backend.GLOO:
                    world_size = dist.get_world_size()
                    grad /= world_size
        self.handles.clear()

