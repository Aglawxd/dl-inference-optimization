import torch

print("CUDA available ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name: ", torch.cuda.get_device_name(0))
    print("CUDA version: ", torch.version.cuda)
    print('GPUs available: ' , torch.cuda.device_count())
else:
    print("No NVIDIA GPU available - check connection")
