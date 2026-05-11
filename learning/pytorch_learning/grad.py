import torch

a = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)
c = a * b

c.backward()

print(a.grad)
print(b.grad)
