import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DensityDataset
from model import CSRNet
import numpy as np

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CSRNet().to(device)
model.load_state_dict(torch.load("csrnet_trained.pth", map_location=device))
model.eval()

# Dataset and DataLoader
dataset = DensityDataset()
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Metrics
mae = 0.0
mse = 0.0

for i, (img, gt_map) in enumerate(dataloader):
    img = img.to(device)
    gt_map = gt_map.to(device)

    with torch.no_grad():
        pred_map = model(img)

    pred_count = pred_map.sum().item()
    gt_count = gt_map.sum().item()

    # Add this sanity check print:
    print(f"Image {i}: GT count = {gt_count:.2f}, Pred count = {pred_count:.2f}")

    mae += abs(pred_count - gt_count)
    mse += (pred_count - gt_count) ** 2

mae /= len(dataset)
rmse = np.sqrt(mse / len(dataset))

print(f"Evaluation Results on {len(dataset)} images:")
print(f"MAE  = {mae:.2f}")
print(f"RMSE = {rmse:.2f}")