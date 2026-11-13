from typing import Any

import torch

from .base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer for the contrastive text-location models.
    """

    def process_batch(self, batch: dict[str, Any], part: str) -> dict[str, Any]:
        """
        Run a batch through the model, compute loss and metrics, and backpropagate the loss.
        """
        is_train = part == 'train'
        metric_tracker = self.train_metric_tracker if is_train else self.eval_metric_tracker

        batch = self.prepare_batch(batch, is_train=is_train)

        with self.autocast():
            outputs = self.model(**batch)
            batch.update(outputs)

            all_losses = self.criterion(**batch)
            batch.update(all_losses)

        if is_train:
            self.backward(batch['loss'])  # Sum of all losses is always called loss

        # Update metrics for each loss (in case of multiple losses)
        for loss_name in self.criterion.loss_names:
            metric_tracker.update(loss_name, batch[loss_name].detach())

        for met in self.metrics['train' if is_train else 'inference']:
            if is_train:
                metric_tracker.update(met.name, met(**batch))
            else:
                # Accumulated over the whole partition, computed in '_evaluate_part'
                met.update(**batch)

        return batch

    @torch.no_grad()
    def _log_batch(self, batch: dict[str, Any], part: str) -> None:
        """
        Log data from a batch (nothing is logged in this example).
        """
