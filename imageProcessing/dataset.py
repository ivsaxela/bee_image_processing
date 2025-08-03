from torch.utils.data import Dataset
import h5py
import cv2
import numpy as np
from pathlib import Path
import torch

class DensityDataset(Dataset):
    def __init__(self, transform=None):
        ROOT = Path(__file__).resolve().parent.parent
        self.image_dir = ROOT / "processed_images" / "JPG_images"
        self.gt_dir = ROOT / "processed_images" / "GT_images"
        self.image_paths = sorted(self.image_dir.glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        gt_path = self.gt_dir / f"GT_{img_path.stem}.h5"

        # Load image and normalize to [0, 1]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img_tensor = torch.from_numpy(img)

        # Get original image dimensions
        input_h, input_w = img.shape[1], img.shape[2]

        # Load ground truth density map
        with h5py.File(gt_path, 'r') as f:
            density = np.array(f['density']).astype(np.float32)

        # Resize density map to match model output (1/8 of image dimensions)
        output_w = input_w // 8
        output_h = input_h // 8
        density_resized = cv2.resize(density, (output_w, output_h), interpolation=cv2.INTER_CUBIC)

        # Normalize to preserve total count
        if density.sum() > 0:
            density_resized *= (density.sum() / density_resized.sum())

        density_tensor = torch.from_numpy(density_resized).unsqueeze(0)  # Shape: [1, H, W]

        return img_tensor, density_tensor