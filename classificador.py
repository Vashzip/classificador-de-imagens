import torch

print("Versão do PyTorch:", torch.__version__)

x = torch.rand(3, 3)
print("Tensor aleatório:\n", x)

a = torch.tensor([2.0, 3.0])
b = torch.tensor([4.0, 5.0])
print("Soma de tensores:", a + b)