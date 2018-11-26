#!/usr/bin/env python3

import os
import torch

os.system("clear")

print("Construct a 5x3 matrix, uninitialized:")
x = torch.empty(5, 3)
print(x)
print()

print("Construct a randomly initialized matrix:")
x = torch.rand(5, 3)
print(x)
print()

print("Construct a matrix filled zeros and of dtype long:")
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
print()

print("Construct a tensor directly from data:")
x = torch.tensor([5.5, 3])
print(x)
print()

print("Construct a matrix filled withones and of dtype double:")
x = x.new_ones(5, 3, dtype=torch.double)
print(x)
print()

print("Create a randomly initialized tensor based on an existing tensor:")
x = torch.randn_like(x, dtype=torch.float)
print(x)
print()

print("Get the size of a tensor:")
print(x.size())
print()

print("Perform addition using + operator:")
y = torch.rand(5, 3)
print(x + y)
print()

print("Perform addition using add method:")
print(torch.add(x, y))
print()

print("Perform addition, providing an output tensor:")
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
print()

print("Perform addition in-place:")
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
print()

print("Perform numpy-like indexing:")
print(x[:, 1])
print()

print("Perform reshaping:")
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())
print()

print("Get the scalar value of a one-element tensor:")
x = torch.randn(1)
print(x)
print(x.item())
print()

print("Create a numpy view of a tensor:")
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
print()

print("Confirm that changing the original tensor changes the numpy view:")
a.add_(1)
print(a)
print(b)
print()

print("Create a tensor view of a numpy array:")
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
print()

print("Move a tensor to GPU:")
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on G
    print("y:")
    print(y)
    print("x (on CPU):")
    print(x)
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    print("x (transferred to GPU)::")
    print(x)
    z = x + y
    print("z = x + y:")
    print(z)
    print("z transferred to CPU")
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
print()
