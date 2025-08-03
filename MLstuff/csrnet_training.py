import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DensityDataset
from model import CSRNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset and DataLoader
dataset = DensityDataset()
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

for i in range(5):
    img, gt_density = dataset[i]
    print(f"Image {i}: GT bee count from density map = {gt_density.sum().item():.2f}")

# Model, loss, optimizer
model = CSRNet().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for imgs, gt_maps in dataloader:
        imgs = imgs.to(device)
        gt_maps = gt_maps.to(device)

        preds = model(imgs)
        loss = criterion(preds, gt_maps)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping here
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

# Save trained model
torch.save(model.state_dict(), "csrnet_trained.pth")
print("Model saved to csrnet_trained.pth")