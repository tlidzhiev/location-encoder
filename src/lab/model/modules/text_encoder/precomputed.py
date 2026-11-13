import torch
from torch import nn


class PrecomputedTextEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projection = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, text_emb: torch.Tensor) -> torch.Tensor:
        return self.projection(text_emb)
