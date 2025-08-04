import cv2
import numpy as np
import h5py
from pathlib import Path
import os

# -------- Configuration --------
ROOT = Path(__file__).resolve().parent.parent
image_dir = ROOT / "processed_images" / "JPG_images"
gt_dir = ROOT / "processed_images" / "GT_images"
output_dir = ROOT / "processed_images" / "overlaid_heatmaps"
output_dir.mkdir(parents=True, exist_ok=True)
opacity = 0.5  # 50%
# -------------------------------

image_paths = sorted(image_dir.glob("*.jpg"))

for image_path in image_paths:
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load image: {image_path.name}")
        continue

    # Load corresponding .h5 density map
    h5_path = gt_dir / f"GT_{image_path.stem}.h5"
    if not h5_path.exists():
        print(f"Missing density map: {h5_path.name}")
        continue

    with h5py.File(h5_path, 'r') as hf:
        density = np.array(hf['density'])

    # Check if image and density map sizes match
    if (img.shape[0], img.shape[1]) != density.shape:
        print(f"Size mismatch: {image_path.name} | image: {img.shape[:2]} vs density: {density.shape}")
        continue

    # Normalize density map for visualization
    density_normalized = (density / np.max(density + 1e-6) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(density_normalized, cv2.COLORMAP_JET)

    # Overlay heatmap
    overlayed = cv2.addWeighted(img, 1 - opacity, heatmap, opacity, 0)

    # Save overlayed image
    out_path = output_dir / image_path.name
    cv2.imwrite(str(out_path), overlayed)
    print(f"Saved overlay: {out_path.name}")

print("Overlay generation complete.")