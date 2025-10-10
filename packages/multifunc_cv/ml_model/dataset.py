import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import random

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.class_to_images = {}

        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)

            if not os.path.isdir(class_path):
                continue

            image_files = [
                os.path.join(class_path, img)
                for img in os.listdir(class_path)
                if img.endswith(('.jpg', '.jpeg', '.png'))
            ]

            self.class_to_images[class_name] = image_files

        self.classes = list(self.class_to_images.keys())

        print(f"Loaded {len(self.classes)} classes:")
        for class_name, images in self.class_to_images.items():
            print(f"  {class_name}: {len(images)} images")

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        same_class = random.randint(0, 1)

        if same_class:
            class_name = random.choice(self.classes)

            if len(self.class_to_images[class_name]) < 2:
                class1, class2 = random.sample(self.classes, 2)
                img1_path = random.choice(self.class_to_images[class1])
                img2_path = random.choice(self.class_to_images[class2])
                label = 1
            else:
                img1_path, img2_path = random.sample(self.class_to_images[class_name], 2)
                label = 0
        else:
            class1, class2 = random.sample(self.classes, 2)
            img1_path = random.choice(self.class_to_images[class1])
            img2_path = random.choice(self.class_to_images[class2])
            label = 1

        img1 = Image.open(img1_path).convert('L')
        img2 = Image.open(img2_path).convert('L')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)