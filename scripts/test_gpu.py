import torch
import torch_directml

print("available:", torch_directml.is_available())

device = None
try:
    device = torch_directml.device()
    print("device:", device)
except Exception as e:
    print("device error:", e)

if device is not None:
    x = torch.randn(100, 100, device=device)
    y = torch.randn(100, 100, device=device)
    z = x @ y
    print("compute ok:", z.shape)
