import torchvision

import torchvision

train_data = torchvision.datasets.ImageNet("./dataset", split="train", transform=torchvision.transforms.ToTensor())
