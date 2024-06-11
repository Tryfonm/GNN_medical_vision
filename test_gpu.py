import torch

# Check if a GPU is available
if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(device)
    cuda_version = torch.version.cuda

    print(f"GPU Model: {gpu_name}")
    print(f"CUDA Version: {cuda_version}")
else:
    print("GPU not available, using CPU")
    device = torch.device('cpu')

# Create a tensor
tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

# Move the tensor to the GPU (if available) and perform an operation
tensor = tensor.to(device)
result = torch.matmul(tensor, tensor)

print("Tensor:")
print(tensor)
print("Result of matrix multiplication on GPU:")
print(result)
