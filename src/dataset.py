import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CTDataset(Dataset):
    def __init__(self, root_dir, image_paths, labels, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = image_paths
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, img_path
