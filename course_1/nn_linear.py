import torch
from torch import nn
from torch.nn import ReLU
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Sigmoid
from torch.nn import Linear

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, download=True, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.linear1 = Linear(in_features=3*32*32, out_features=10)
  
  def forward(self, x):
    output = self.linear1(x)
    return output

net = Net()

for data in dataloader:
  imgs, targets = data
  print(imgs.shape)
  # output = torch.reshape(imgs, (1, 1, 1, -1))
  output = torch.flatten(imgs)
  print(output.shape)
  output = net(output)
  print(output.shape)
  break