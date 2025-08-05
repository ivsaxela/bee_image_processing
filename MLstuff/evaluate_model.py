import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DensityDataset
from model import CSRNet
import numpy as np

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = CSRNet().to(device)
model.load_state_dict(torch.load("csrnet_trained.pth", map_location=device))
model.eval()

# Load dataset (already sorted internally using pathlib)
dataset = DensityDataset(root_dir='path_to_images')  # Replace with your actual image directory
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Metrics
mae = 0.0
mse = 0.0

# Evaluation loop
for i, (img, gt_map) in enumerate(dataloader):
    img = img.to(device)
    gt_map = gt_map.to(device)

    with torch.no_grad():
        pred_map = model(img)

    pred_count = pred_map.sum().item()
    gt_count = gt_map.sum().item()

    filename = dataset.image_files[i].name  # pathlib.Path.name gives filename only

    print(f"Image {i} ({filename}): GT count = {gt_count:.2f}, Pred count = {pred_count:.2f}")

    mae += abs(pred_count - gt_count)
    mse += (pred_count - gt_count) ** 2

# Final metrics
mae /= len(dataset)
rmse = np.sqrt(mse / len(dataset))

print(f"\nEvaluation Results on {len(dataset)} images:")
print(f"MAE  = {mae:.2f}")
print(f"RMSE = {rmse:.2f}")