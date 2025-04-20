import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.refinement = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up_sample(x)
        x = torch.cat([x, skip], dim=1)
        return self.refinement(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.in_conv = ConvBlock(n_channels, 64)
        self.enc1 = Encoder(64, 128)
        self.enc2 = Encoder(128, 256)
        self.enc3 = Encoder(256, 512)
        self.enc4 = Encoder(512, 1024)

        self.dec1 = Decoder(1024 + 512, 512)
        self.dec2 = Decoder(512 + 256, 256)
        self.dec3 = Decoder(256 + 128, 128)
        self.dec4 = Decoder(128 + 64, 64)

        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        d1 = self.dec1(x5, x4)
        d2 = self.dec2(d1, x3)
        d3 = self.dec3(d2, x2)
        d4 = self.dec4(d3, x1)

        return torch.sigmoid(self.out_conv(d4))

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = UNet(3, 1).to(device)