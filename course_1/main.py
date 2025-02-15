from torch.utils.data import Dataset
from PIL import Image

image_path = "dataset/train/ants/0013035.jpg"
image = Image.open(image_path)
image.show()