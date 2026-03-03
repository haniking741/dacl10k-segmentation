"""
DeepLabV3+ Implementation for Multi-label Semantic Segmentation
Uses pretrained ResNet backbone from torchvision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ for multi-label segmentation
    
    Args:
        n_classes: Number of output classes
        backbone: 'resnet50' or 'resnet101'
        pretrained: Use ImageNet pretrained weights
    """
    
    def __init__(self, n_classes=19, backbone='resnet50', pretrained=True):
        super(DeepLabV3Plus, self).__init__()
        
        self.n_classes = n_classes
        
        # Load pretrained DeepLabV3 from torchvision
        if backbone == 'resnet50':
            self.model = deeplabv3_resnet50(pretrained=pretrained, progress=True)
        elif backbone == 'resnet101':
            self.model = deeplabv3_resnet101(pretrained=pretrained, progress=True)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Replace classifier to match our number of classes
        # Original has 21 classes (PASCAL VOC), we need n_classes
        in_channels = self.model.classifier[4].in_channels
        
        self.model.classifier[4] = nn.Conv2d(
            in_channels, 
            n_classes, 
            kernel_size=1
        )
        
        # Also update auxiliary classifier if it exists
        if hasattr(self.model, 'aux_classifier'):
            aux_in_channels = self.model.aux_classifier[4].in_channels
            self.model.aux_classifier[4] = nn.Conv2d(
                aux_in_channels,
                n_classes,
                kernel_size=1
            )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Output tensor [B, n_classes, H, W]
        """
        input_shape = x.shape[-2:]  # H, W
        
        # Forward through model
        output = self.model(x)
        
        # DeepLabV3 returns a dict with 'out' and optionally 'aux'
        # We only need 'out' for inference
        if isinstance(output, dict):
            output = output['out']
        
        # Resize to input size if needed
        if output.shape[-2:] != input_shape:
            output = F.interpolate(
                output, 
                size=input_shape, 
                mode='bilinear', 
                align_corners=False
            )
        
        return output


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_deeplabv3(n_classes=19, backbone='resnet50', pretrained=True, device='cpu'):
    """
    Factory function to create DeepLabV3+ model
    
    Args:
        n_classes: Number of output classes
        backbone: 'resnet50' (faster) or 'resnet101' (better)
        pretrained: Use ImageNet pretrained weights
        device: Device to load model on
        
    Returns:
        DeepLabV3+ model
    """
    print(f"📐 Creating DeepLabV3+ with {backbone} backbone")
    
    model = DeepLabV3Plus(
        n_classes=n_classes,
        backbone=backbone,
        pretrained=pretrained
    )
    
    n_params = count_parameters(model)
    print(f"📊 Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    model = model.to(device)
    
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing DeepLabV3+ model...\n")
    
    model = get_deeplabv3(n_classes=19, backbone='resnet50', device='cpu')
    
    # Test forward pass
    x = torch.randn(2, 3, 384, 384)
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        out = model(x)
    
    print(f"Output shape: {out.shape}")
    print(f"Expected: [2, 19, 384, 384]")
    
    if out.shape == torch.Size([2, 19, 384, 384]):
        print("\n✅ Model working correctly!")
    else:
        print("\n❌ Model output shape incorrect!")