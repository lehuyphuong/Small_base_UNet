import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class OxfordPetsDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, size=(128, 128)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.image_list = sorted([
            f for f in os.listdir(image_dir) if f.endswith('.jpg')
        ])
        self.mask_list = sorted([
            f for f in os.listdir(mask_dir) if f.endswith('.png')
        ])

        # Ensure both lists are aligned by filename prefix
        self.image_list = [f for f in self.image_list if f.replace('.jpg', '.png') in self.mask_list]
        self.mask_list = [f.replace('.jpg', '.png') for f in self.image_list]

        self.size = size
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_list[idx])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        image = image.resize(self.size)
        mask = mask.resize(self.size, Image.NEAREST)

        mask_np = np.array(mask)
        mask_np = np.where(mask_np == 1, 1, 0)  # Binary: pet = 1, else = 0

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(mask_np, dtype=torch.long)
    
def get_loaders(image_dir, mask_dir, batch_size=2, size=(128, 128)):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = OxfordPetsDataset(image_dir, mask_dir, transform, size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def compute_metrics(preds, labels):
    with torch.no_grad():
        preds = (preds > 0.5).float()
        pixel_acc = (preds == labels).float().mean().item()

        intersection = (preds * labels).sum().float()
        union = (preds + labels).sum().float() - intersection
        dice = (2. * intersection) / (preds.sum() + labels.sum() + 1e-6)
        iou = intersection / (union + 1e-6)

        return pixel_acc, iou.item(), dice.item()
    
