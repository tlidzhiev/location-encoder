import torch
import torch.nn.functional as F

from .base import BaseLoss


def same_text_mask(text_id: torch.Tensor) -> torch.Tensor:
    """
    (B, B) mask of the off-diagonal pairs with the same text: they are not negatives.
    """
    same = text_id[:, None] == text_id[None, :]
    return same & ~torch.eye(len(text_id), dtype=torch.bool, device=text_id.device)


class ClipLoss(BaseLoss):
    """
    Symmetric contrastive (InfoNCE) loss over text-location pairs of a batch.

    The pairs of different POIs with the same text (a chain: equal 'text_id') are excluded from
    the negatives: their text embeddings are identical, so neither can be told apart.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loss_names = ['loss']

    def forward(
        self,
        logits_per_text: torch.Tensor,
        logits_per_location: torch.Tensor,
        text_id: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Average of the text->location and location->text cross-entropy, under the 'loss' key.

        Parameters
        ----------
        logits_per_text : torch.Tensor
            (B, B) logits, the row is a text and the column is a location.
        logits_per_location : torch.Tensor
            Transposed 'logits_per_text'.
        text_id : torch.Tensor or None, optional
            (B,) ids of the distinct texts; if given, the pairs with the same text are masked.
        """
        if text_id is not None:
            mask = same_text_mask(text_id)
            logits_per_text = logits_per_text.masked_fill(mask, float('-inf'))
            logits_per_location = logits_per_location.masked_fill(mask, float('-inf'))

        labels = torch.arange(logits_per_text.shape[0], device=logits_per_text.device)
        loss = (
            F.cross_entropy(logits_per_text, labels) + F.cross_entropy(logits_per_location, labels)
        ) / 2
        return {'loss': loss}
