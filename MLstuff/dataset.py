from torch.utils.data import Dataset
from torch.utils.data import Dataset
import h5py
import cv2
import numpy as np
from pathlib import Path
import torch

class DensityDataset(Dataset):
    def __init__(self, transform=None):
        self.image_dir = Path(r"C:/Users/isakh/Desktop/bee_image_processing/processed_images/JPG_images")
        self.gt_dir = Path(r"C:/Users/isakh/Desktop/bee_image_processing/processed_images/GT_images")
        self.image_paths = sorted(list(self.image_dir.glob("*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        gt_path = self.gt_dir / f"GT_{img_path.stem}.h5"

        # Load image
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # CHW format

        # Load density map
        with h5py.File(gt_path, 'r') as f:
            density = np.array(f['density']).astype(np.float32)

        img_tensor = torch.from_numpy(img)
        density_tensor = torch.from_numpy(density).unsqueeze(0)

        return img_tensor, density_tensor