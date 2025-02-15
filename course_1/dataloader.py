import torchvision

from torch.utils.data import DataLoader

test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_loader = DataLoader(test_data, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

img, target = test_data[0]
print(img.shape)
print(target)