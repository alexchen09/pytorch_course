import torch
from torch import nn
from torch.nn import Module, Conv2d, MaxPool2d, Linear, Flatten, Sequential
from torch.utils.data import DataLoader

import torchvision

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=1)

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
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

loss = nn.CrossEntropyLoss()
net = Net()
for data in dataloader:
  imgs, targets = data
  output = net(imgs)
  result_loss = loss(output, targets)
  result_loss.backward()
  print(result_loss)