import math

import numpy as np
import torch
from torch import nn


def cal_freq_list(frequency_num: int, max_radius: float, min_radius: float) -> np.ndarray:
    log_timescale_increment = math.log(float(max_radius) / float(min_radius)) / (frequency_num - 1)
    timescales = min_radius * np.exp(
        np.arange(frequency_num).astype(float) * log_timescale_increment
    )
    return 1.0 / timescales


class GridPositionEncoder(nn.Module):
    freq_list: torch.Tensor

    def __init__(
        self,
        min_radius: float,
        max_radius: float,
        frequency_num: int,
        coord_dim: int = 2,
    ) -> None:
        super().__init__()
        self.coord_dim = coord_dim
        self.frequency_num = frequency_num
        self.output_dim = 2 * coord_dim * frequency_num
        self.register_buffer(
            'freq_list',
            torch.from_numpy(cal_freq_list(frequency_num, max_radius, min_radius)),
            persistent=False,
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        angles = coords[:, :, None] * self.freq_list.to(coords.dtype)  # (B, coord_dim, F)
        embed = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, coord_dim, F, 2)
        return embed.flatten(1)
