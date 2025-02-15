import torch
from torch import nn
from torch.nn import ReLU
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Sigmoid

input = torch.tensor([[1, -0.5], [-1, 3]])

input = torch.reshape(input, (1, 1, 2, 2))
print(input.shape)

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.relu1 = ReLU()
    self.sigmoid1 = Sigmoid()
    
  def forward(self, input):
    output = self.sigmoid1(input)
    return output
  
net = Net()
writer = SummaryWriter("./logs_sigmoid")
step = 0
for data in dataloader:
  imgs, target = data
  writer.add_images("input", imgs, global_step=step)
  output = net(imgs)
  writer.add_images("output", output, step)
  step += 1

writer.close()