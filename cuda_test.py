import torch
print("CUDA verfügbar:", torch.cuda.is_available())
print("CUDA Version (torch):", torch.version.cuda)
print("Gerät:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
