import torchvison

train_data = torchvision.datasets.ImageNet("./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                           download=True)

test_data = torchvision.datasets.ImageNet("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                          download=True)

#show length of dataset
train_data_size = len(train_data)
test_data_size = len(test_data)
print("Train data size: ", train_data_size)
print("Test data size: ", test_data_size)

#load dataset with Dataloader
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

#build model
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.model = nn.Sequential(
      nn.Conv2d(3, 32, 5, 1, padding=2),
      nn.MaxPool2d(2),
      nn.Conv2d(32, 32, 5, padding=2),
      nn.MaxPool2d(2),
      nn.Conv2d(32, 64, 5, padding=2),
      nn.MaxPool2d(2),
      nn.Flatten(),
      nn.Linear(64*4*4, 64),
      nn.Linear(64, 1000)
    )
  
  def forward(self, x):
    x = self.model(x)
    return x