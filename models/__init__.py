fixed_init = '''"""
Models package
"""

from .unet import UNet, UNetLite, get_model as get_unet_model
from .deeplabv3 import DeepLabV3Plus, get_model as get_deeplabv3_model

def get_model(model_type='unet', n_classes=19, device='cpu'):
    """
    Factory function to create any model
    
    Args:
        model_type: 'unet', 'unet_lite', 'deeplabv3_resnet50', 'deeplabv3_resnet101'
        n_classes: Number of output classes
        device: Device to load model on
        
    Returns:
        Model instance
    """
    
    if model_type == 'unet':
        return get_unet_model('unet', n_classes, device)
    
    elif model_type == 'unet_lite':
        return get_unet_model('unet_lite', n_classes, device)
    
    elif model_type in ['deeplabv3_resnet50', 'deeplabv3_resnet101']:
        # Use the deeplabv3 factory (which expects model_type, n_classes, device)
        return get_deeplabv3_model(model_type, n_classes, device)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


__all__ = [
    'UNet',
    'UNetLite', 
    'DeepLabV3Plus',
    'get_model',
    'get_unet_model',
    'get_deeplabv3_model',
]
'''

with open("/kaggle/working/dacl10k-segmentation/models/__init__.py", "w") as f:
    f.write(fixed_init)

print("✅ models/__init__.py fixed")
