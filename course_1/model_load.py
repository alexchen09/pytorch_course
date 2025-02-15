import torch
import torchvision
from model_save import *

# method 1: model structure and weights
model = torch.load("vgg16_true.pth")
print(model)

# method 2: model weights
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_state_dict.pth"))
# model = torch.load("vgg16_state_dict.pth")
print(vgg16)