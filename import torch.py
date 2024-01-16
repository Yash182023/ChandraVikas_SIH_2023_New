import torch

if torch.cuda.is_available():
    print("CUDA is available on your system.")
    # Additional code for GPU-related operations can go here
    cuda_device_count = torch.cuda.device_count()
    print(f"Number of CUDA devices available: {cuda_device_count}")
else:
    print("CUDA is not available on your system. You may want to run this code on a GPU-enabled machine or check your CUDA installation.")
