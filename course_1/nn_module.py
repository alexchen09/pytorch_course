from torch import nn
import torch

class Tudui(nn.Module):
  def __init__(self):
    super(Tudui, self).__init__()
  
  def forward(self, x):
    return x + 1

tudui = Tudui()
x = torch.tensor(1.0)
output = tudui(x)
print(output)