import torch.nn as nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights

class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()

        # Load pretrained VGG16 model
        vgg = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)

        # Use only the first 33 layers of VGG16 as the frontend
        self.frontend = nn.Sequential(*list(vgg.features.children())[:33])

        # CSRNet backend layers
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Final output layer for density map
        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x