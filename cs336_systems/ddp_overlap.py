import torch
from cs336_system.torch_util import *


class DDPIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.handles = []
        pass

    def forward(self, *inputs, **kwargs):
        pass

    def finish_gradient_synchronization(self):
        pass
