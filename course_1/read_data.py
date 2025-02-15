from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self, root_dir, image_dir, label):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.label = label
        self.path = os.path.join(self.root_dir, self.image_dir)
        self.images = os.listdir(self.path)
    
    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.path, image_name)
        img = Image.open(image_path)
        return img, self.label
    
    def __len__(self):
        return len(self.images)

    def create_label_txt(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        for image_name in self.images:
            txt_filename = os.path.splitext(image_name)[0] + ".txt"
            txt_path = os.path.join(output_dir, txt_filename)

            with open(txt_path, "w") as f:
                f.write(self.label)


root_dir = "dataset/train"

ants_image_dir = "ants_image"
ants_label = "ant"
ants_dataset = MyData(root_dir, ants_image_dir, ants_label)
ants_output_dir = "dataset/train/ants_label"  
ants_dataset.create_label_txt(ants_output_dir)


bees_image_dir = "bees_image"
bees_label = "bee"
bees_dataset = MyData(root_dir, bees_image_dir, bees_label)
bees_output_dir = "dataset/train/bees_label"  
bees_dataset.create_label_txt(bees_output_dir)


