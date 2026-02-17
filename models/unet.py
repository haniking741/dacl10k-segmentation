"""
U-Net Architecture for Semantic Segmentation
Includes both standard and lightweight versions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        # Use bilinear upsampling (faster) or transposed conv
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatch (input size not divisible by 16)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net architecture
    
    Args:
        n_channels: Number of input channels (3 for RGB)
        n_classes: Number of output classes
        bilinear: Use bilinear upsampling (faster) instead of transposed conv
    """
    
    def __init__(self, n_channels=3, n_classes=20, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNetLite(nn.Module):
    """
    Lightweight U-Net for CPU training
    Fewer filters, shallower architecture
    ~4x faster than standard U-Net
    """
    
    def __init__(self, n_channels=3, n_classes=20, bilinear=True):
        super(UNetLite, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder (fewer filters: 32->64->128->256)
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down3 = Down(128, 256 // factor)
        
        # Decoder
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Decoder with skip connections
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(model_type='unet', n_classes=20, device='cpu'):
    """
    Factory function to create model
    
    Args:
        model_type: 'unet' (standard) or 'unet_lite' (lightweight for CPU)
        n_classes: Number of classes
        device: 'cpu' or 'cuda'
    
    Returns:
        model: PyTorch model
    """
    if model_type == 'unet':
        model = UNet(n_channels=3, n_classes=n_classes, bilinear=True)
        print(f"📐 Created standard U-Net")
    elif model_type == 'unet_lite':
        model = UNetLite(n_channels=3, n_classes=n_classes, bilinear=True)
        print(f"⚡ Created lightweight U-Net (CPU-optimized)")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    n_params = count_parameters(model)
    print(f"📊 Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    model = model.to(device)
    return model


if __name__ == "__main__":
    # Test the models
    print("Testing U-Net models...\n")
    
    # Test standard U-Net
    print("=" * 50)
    print("STANDARD U-NET")
    print("=" * 50)
    model = get_model('unet', n_classes=20, device='cpu')
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print()
    
    # Test lightweight U-Net
    print("=" * 50)
    print("LIGHTWEIGHT U-NET")
    print("=" * 50)
    model_lite = get_model('unet_lite', n_classes=20, device='cpu')
    with torch.no_grad():
        out_lite = model_lite(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out_lite.shape}")
    print()
    
    print("✅ Models working correctly!")