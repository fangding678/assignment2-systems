import torch

print(torch.cuda.is_available())


t1 = torch.cdist(torch.randn(10, 2), torch.randn(10, 2))
print(t1.shape)


a = torch.randn(1024, 512, 1280, device='cuda:0')
b = torch.randn(1024, 1280, 512, device='cuda:0')
c = torch.bmm(a, b)
print(c.shape)

