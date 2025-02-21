import torch
from torch import nn
from torch.nn import Module, Conv2d, MaxPool2d, Linear, Flatten, Sequential
from torch.utils.data import DataLoader

import torchvision

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    # self.conv1 = Conv2d(3, 32, 5, padding=2)
    # self.maxpool1 = MaxPool2d(2)
    # self.conv2 = Conv2d(32, 32, 5, padding=2)
    # self.maxpool2 = MaxPool2d(2)
    # self.conv3 = Conv2d(32, 64, 5, padding=2)
    # self.maxpool3 = MaxPool2d(2)
    # self.flatten = Flatten()
    # self.linear1 = Linear(64*4*4, 64)
    # self.linear2 = Linear(64, 10)
    self.model = Sequential(
      Conv2d(3, 32, 5, padding=2),
      MaxPool2d(2),
      Conv2d(32, 32, 5, padding=2),
      MaxPool2d(2),
      Conv2d(32, 64, 5, padding=2),
      MaxPool2d(2),
      Flatten(),
      Linear(64*4*4, 64),
      Linear(64, 10)
    )
    
  def forward(self, x):
    x = self.model(x)
    return x
  
  # def forward(self, x):
  #   x = self.conv1(x)
  #   x = self.maxpool1(x)
  #   x = self.conv2(x)
  #   x = self.maxpool2(x)
  #   x = self.conv3(x)
  #   x = self.maxpool3(x)
  #   x = self.flatten(x)
  #   x = self.linear1(x)
  #   output = self.linear2(x)
  #   return output

net = Net()
for data in dataloader:
  imgs, targets = data
  output = net(imgs)
  print(output.shape)
  break