import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import DensityDataset
from model import CSRNet

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset and DataLoader
dataset = DensityDataset()
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # You can set shuffle=False if desired

# Preview dataset counts
for i in range(24):
    img, gt_density = dataset[i]
    print(f"Image {i}: GT bee count from density map = {gt_density.sum().item():.2f}")

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
        percent_count_loss = torch.abs(pred_count - gt_count) / (gt_count + 1e-6)  # avoid div-by-zero

        total = self.alpha * pixel_loss + self.beta * percent_count_loss
        return total, pixel_loss.detach(), percent_count_loss.detach()
# ----------------------------------

# Model, loss, optimizer
model = CSRNet().to(device)
criterion = HybridLoss(alpha=1.0, beta=0.1, use_mae=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Training loop
epochs = 30
for epoch in range(epochs):
    model.train()
    epoch_total_loss = 0.0
    epoch_pixel_loss = 0.0
    epoch_count_loss = 0.0

    num_batches = len(dataloader)

    for imgs, gt_maps in dataloader:
        imgs = imgs.to(device)
        gt_maps = gt_maps.to(device)

        preds = model(imgs)
        loss, pixel_loss, count_loss = criterion(preds, gt_maps)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        epoch_total_loss += loss.item()
        epoch_pixel_loss += pixel_loss.item()
        epoch_count_loss += count_loss.item()

    avg_total_loss = epoch_total_loss / num_batches
    avg_pixel_loss = epoch_pixel_loss / num_batches
    avg_count_loss = epoch_count_loss / num_batches

    # Print with percentage count loss (2 decimals)
    print(f"[Epoch {epoch + 1}/{epochs}] Total Loss: {avg_total_loss:.4f} | "
          f"Pixel Loss: {avg_pixel_loss:.4f} | Count Loss: {avg_count_loss*100:.4f}%")
torch.save(model.state_dict(), "csrnet_trained.pth")
print("Model saved to csrnet_trained.pth")