import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ThyroidSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.mask_transform = mask_transform

        self.images = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        base_name = os.path.splitext(img_name)[0]
        mask_name = base_name + "_mask.png"
        mask_path = os.path.join(self.masks_dir, mask_name)

        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = transforms.ToTensor()(mask)

        mask = (mask > 0).float()

        return image, mask
