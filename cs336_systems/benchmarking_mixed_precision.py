import torch
from torch import nn
from torch_util import get_device


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        print('input data', x.dtype)
        x = self.relu(self.fc1(x))
        print('fc1 data', x.dtype)
        x = self.ln(x)
        print('ln data', x.dtype)
        x = self.fc2(x)
        print('fc2 data', x.dtype)
        return x


def mix_precision_test01():
    in_dim = 32
    out_dim = 32

    model = ToyModel(in_dim, out_dim).to(get_device())
    x = torch.randn(16, 32, device=get_device())
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)

    print(f'\ntorch.cuda.is_available() = {torch.cuda.is_available()}\n')

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        y = model(x)
        print('y', y.dtype)
        print()
        y.mean().backward()
        print(type(model.state_dict()))
        for k, v in model.state_dict().items():
            print(k, v.shape, v.dtype)
        print()
        optimizer.step()
        for k, v in optimizer.state_dict()['state'].items():
            try:
                print(k, v.shape, v.dtype)
            except:
                print(k)
                for k1, v1 in v.items():
                    print(k1, v1.shape, v1.dtype)


if __name__ == '__main__':
    mix_precision_test01()
    pass




