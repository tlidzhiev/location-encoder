import torch
from torch import nn

from ...utils import get_activation, initialize_weights


class ResBlock(nn.Module):
    def __init__(self, dim: int, activation: str = 'relu') -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            get_activation(activation),
            nn.Dropout(),
            nn.Linear(dim, dim),
            get_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Space2Vec(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        activation: str = 'relu',
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.class_emb = nn.Linear(hidden_dim, output_dim, bias=False)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            get_activation(activation),
            *(ResBlock(hidden_dim, activation) for _ in range(num_layers)),
        )
        initialize_weights(self, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.class_emb(self.backbone(x))
