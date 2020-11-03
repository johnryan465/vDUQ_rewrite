import torch
import gpytorch
import typing


class GP_DK(gpytorch.Module):
    def __init__(self, embedding : torch.nn.Module, gp :  gpytorch.Module ) -> None:
        super(GP_DK, self).__init__()
        self.embedding = embedding
        self.gp = gp

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.gp(self.embedding(x))
