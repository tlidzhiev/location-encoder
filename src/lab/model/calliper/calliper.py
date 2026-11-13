import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..base import BaseModel
from ..modules.text_encoder import PrecomputedTextEncoder


class Calliper(BaseModel):
    def __init__(
        self,
        positional_encoding: nn.Module,
        location_encoder: nn.Module,
        text_encoder: PrecomputedTextEncoder,
    ) -> None:
        super().__init__()
        assert positional_encoding.output_dim == location_encoder.input_dim, (
            f'Positional encoding size ({positional_encoding.output_dim}) != '
            f'location encoder input size ({location_encoder.input_dim})'
        )
        assert text_encoder.output_dim == location_encoder.output_dim, (
            f'Text encoder size ({text_encoder.output_dim}) != '
            f'location encoder output size ({location_encoder.output_dim})'
        )

        # double precision only for the positional encoding: sin / cos of the coordinates in
        # meters. The rest runs in float32 (and in AMP).
        self.positional_encoding = positional_encoding.double()
        self.location_encoder = location_encoder
        self.text_encoder = text_encoder
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_location(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Coordinates (B, 2) -> location embeddings (B, location_encoder.output_dim).
        """
        return self.location_encoder(self.positional_encoding(coords.double()).float())

    def encode_text(self, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Precomputed text embeddings (B, text_encoder.input_dim) -> (B, text_encoder.output_dim).
        """
        # the precomputed embeddings are stored in float16
        return self.text_encoder(text_emb.float())

    def forward(
        self,
        coords: torch.Tensor,
        text_emb: torch.Tensor,
        **batch: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        location_emb = F.normalize(self.encode_location(coords), dim=1)
        outputs = {'location_emb': location_emb}

        text_emb = F.normalize(self.encode_text(text_emb), dim=1)
        # CLIP: the temperature is capped at 1 / 100
        logit_scale = self.logit_scale.clamp(max=np.log(100)).exp()
        logits_per_text = logit_scale * text_emb @ location_emb.t()
        outputs['text_emb'] = text_emb
        outputs['logits_per_text'] = logits_per_text
        outputs['logits_per_location'] = logits_per_text.t()
        return outputs
