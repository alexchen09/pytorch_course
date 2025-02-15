from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path1 = "dataset/train/ants_image/0013035.jpg"
# image_path2 = "pytorch/dataset/train/ants_image/5650366_e22b7e1065.jpg"
img_PIL1 = Image.open(image_path1)
img_array1 = np.array(img_PIL1)
print(img_array1.shape)
print(type(img_array1))
writer.add_image("train", img_array1, 2, dataformats="HWC")

for i in range(10):
  writer.add_scalar("y=2x", i * 3, i)

writer.close()


