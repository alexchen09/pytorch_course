import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss
from torch import nn

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
outputs = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss(reduction='sum')
result = loss(inputs, outputs)

loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs, outputs)

print(result)
print(result_mse)

x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)