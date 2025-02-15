import torch
import torchvision

#method 1: model structure and weights
vgg16 = torchvision.models.vgg16(pretrained=True)
torch.save(vgg16, "vgg16_true.pth")

# method 2: model weights
torch.save(vgg16.state_dict(), "vgg16_state_dict.pth")