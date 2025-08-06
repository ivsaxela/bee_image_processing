import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataset import DensityDataset
from model import CSRNet
import os
from pathlib import Path

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset and split into train/test
dataset = DensityDataset()
total_len = len(dataset)
train_len = int(0.8 * total_len)
test_len = total_len - train_len
train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Preview GT counts for the first few training samples
for i in range(min(24, len(train_dataset))):
    img, gt_density = train_dataset[i]
    print(f"Train Image {i}: GT bee count = {gt_density.sum().item():.2f}")

print("Beginning training...")

# ----- Hybrid Loss Definition -----
class HybridLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, use_mae=True):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.use_mae = use_mae

    def forward(self, pred, gt):
        pixel_loss = F.l1_loss(pred, gt) if self.use_mae else F.mse_loss(pred, gt)
        pred_count = pred.sum()
        gt_count = gt.sum()
        percent_count_loss = torch.abs(pred_count - gt_count) / (gt_count + 1e-6)
        total = self.alpha * pixel_loss + self.beta * percent_count_loss
        return total, pixel_loss.detach(), percent_count_loss.detach()
# ----------------------------------

# Model, loss, optimizer
model = CSRNet().to(device)
criterion = HybridLoss(alpha=1.0, beta=0.1, use_mae=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Prepare checkpoint saving
epochs = 20
model_dir = Path("checkpoints")
model_dir.mkdir(exist_ok=True)
saved_models = []
best_loss = float('inf')
best_model_path = None

# Training loop
for epoch in range(epochs):
    model.train()
    train_total_loss = 0.0
    train_pixel_loss = 0.0
    train_count_loss = 0.0

    for imgs, gt_maps in train_loader:
        imgs = imgs.to(device)
        gt_maps = gt_maps.to(device)

        preds = model(imgs)
        loss, pixel_loss, count_loss = criterion(preds, gt_maps)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        train_total_loss += loss.item()
        train_pixel_loss += pixel_loss.item()
        train_count_loss += count_loss.item()

    model.eval()
    test_pixel_loss = 0.0
    test_count_loss = 0.0

    with torch.no_grad():
        for imgs, gt_maps in test_loader:
            imgs = imgs.to(device)
            gt_maps = gt_maps.to(device)

            preds = model(imgs)
            _, pixel_loss, count_loss = criterion(preds, gt_maps)

            test_pixel_loss += pixel_loss.item()
            test_count_loss += count_loss.item()

    num_train = len(train_loader)
    num_test = len(test_loader)

    avg_train_total = train_total_loss / num_train
    avg_train_pixel = train_pixel_loss / num_train
    avg_train_count = train_count_loss / num_train

    avg_test_pixel = test_pixel_loss / num_test
    avg_test_count = test_count_loss / num_test

    print(f"[Epoch {epoch + 1}/{epochs}]")
    print(f"  Train — Total Loss: {avg_train_total:.4f} | Pixel Loss: {avg_train_pixel:.4f} | %Count Loss: {avg_train_count * 100:.2f}%")
    print(f"  Test  — Pixel Loss: {avg_test_pixel:.4f} | %Count Loss: {avg_test_count * 100:.2f}%")

    # Save checkpoint for this epoch
    epoch_model_path = model_dir / f"model_epoch{epoch+1}_loss{avg_test_count*100:.2f}.pth"
    torch.save(model.state_dict(), epoch_model_path)
    saved_models.append((epoch_model_path, avg_test_count))

    if avg_train_count < best_loss:
        best_loss = avg_train_count
        best_model_path = epoch_model_path

# Cleanup: keep only the best model
print(f"\nBest model: {best_model_path.name} (chosen by training %Count Loss: {best_loss * 100:.2f}%)")
for path, _ in saved_models:
    if path != best_model_path:
        try:
            path.unlink()
            print(f"Deleted {path.name}")
        except Exception as e:
            print(f"Failed to delete {path.name}: {e}")

# Rename best model
final_path = Path("csrnet_trained.pth")
best_model_path.rename(final_path)
print(f"Best model saved to: {final_path}")
