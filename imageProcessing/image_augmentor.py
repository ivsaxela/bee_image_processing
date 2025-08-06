import cv2
import numpy as np
import h5py
from pathlib import Path
import random
from scipy.ndimage import gaussian_filter, rotate

# --- Config ---
ROOT = Path(__file__).resolve().parent.parent
image_dir = ROOT / "processed_images" / "JPG_images"
gt_dir = ROOT / "processed_images" / "GT_images"
aug_img_dir = ROOT / "processed_images" / "AUG_images"
aug_gt_dir = ROOT / "processed_images" / "AUG_GT"
aug_img_dir.mkdir(parents=True, exist_ok=True)
aug_gt_dir.mkdir(parents=True, exist_ok=True)

AUGS_PER_IMAGE = 19  # not including original
MAX_ROTATE_DEG = 30

def apply_augmentations(img, dmap):
    # Horizontal flip
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
        dmap = cv2.flip(dmap, 1)

    # Vertical flip
    if random.random() < 0.5:
        img = cv2.flip(img, 0)
        dmap = cv2.flip(dmap, 0)

    # Rotation
    angle = random.uniform(-MAX_ROTATE_DEG, MAX_ROTATE_DEG)
    img = rotate(img, angle, reshape=False, mode='reflect')
    dmap = rotate(dmap, angle, reshape=False, mode='reflect')

    # Brightness and contrast
    alpha = random.uniform(0.8, 1.2)
    beta = random.uniform(-10, 10)
    img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

    return img, dmap

# --- Main loop ---
image_paths = sorted(image_dir.glob("*.jpg"))
print(f"Found {len(image_paths)} images. Creating {AUGS_PER_IMAGE + 1} total versions each...")

for img_path in image_paths:
    base_name = img_path.stem
    gt_path = gt_dir / f"GT_{base_name}.h5"

    if not gt_path.exists():
        print(f"Missing GT for {base_name}")
        continue

    img = cv2.imread(str(img_path))
    with h5py.File(gt_path, 'r') as f:
        dmap = np.array(f['density'])

    # Save original as _aug0
    orig_img_path = aug_img_dir / f"{base_name}_aug0.jpg"
    orig_gt_path = aug_gt_dir / f"GT_{base_name}_aug0.h5"
    cv2.imwrite(str(orig_img_path), img)
    with h5py.File(orig_gt_path, 'w') as hf:
        hf['density'] = dmap.astype(np.float32)

    for i in range(1, AUGS_PER_IMAGE + 1):
        aug_img, aug_dmap = apply_augmentations(img.copy(), dmap.copy())

        aug_img_path = aug_img_dir / f"{base_name}_aug{i}.jpg"
        aug_gt_path = aug_gt_dir / f"GT_{base_name}_aug{i}.h5"

        cv2.imwrite(str(aug_img_path), aug_img)
        with h5py.File(aug_gt_path, 'w') as hf:
            hf['density'] = aug_dmap.astype(np.float32)

print("Augmentation complete.")