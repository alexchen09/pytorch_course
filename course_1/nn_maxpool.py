import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter



dataset = torchvision.datasets.CIFAR10("./dataset", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

  def forward(self, x):
    output = self.maxpool1(x)
    return output

net = Net()

writer = SummaryWriter("./logs_maxpool")
step = 0

for data in dataloader:
  imgs, target = data
  writer.add_images("input", imgs, step)
  output = net(imgs)
  writer.add_images("output", output, step)
  step += 1

writer.close()