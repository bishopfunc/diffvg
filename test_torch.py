import torch
print(torch.cuda.is_available())  # True expected
print(torch.version.cuda)         # Should be 12.1
print(torch.__version__)  # 1.10.2+cu102 -> 1.13.0
print(torch.cuda.get_device_name(0))  # NVIDIA GeForce RTX 4090