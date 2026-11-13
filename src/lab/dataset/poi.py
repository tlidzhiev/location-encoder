import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset

from lab.utils.io import get_root

DEFAULT_EMBEDDING = 'Qwen__Qwen3-Embedding-0.6B'


class POIDataset(Dataset):
    """
    POI dataset: coordinates of a place and the precomputed embedding of its text description.

    Reads data/<city>/processed/<split>.csv and the embeddings written by
    scripts/precompute_text_embeddings.py to data/<city>/processed/text_emb/<embedding>.safetensors.
    Everything is held in memory: the embedding matrix has one row per distinct text, and each POI
    only stores the row id.

    An element is a dict:
        'coords'   : (2,) float64, projected x, y (meters)
        'text_emb' : (dim,) float16, L2-normalized text embedding
        'text_id'  : () int64, id of the distinct text; equal ids mean equal texts, so the pairs
                     with the same 'text_id' are false negatives of the contrastive loss
    The key names must stay consistent across the dataset, collate_fn, loss and model.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        city: str = 'london',
        split: Literal['train', 'val'] = 'train',
        embedding: str = DEFAULT_EMBEDDING,
        limit: int | None = None,
        shuffle_index: bool = False,
        instance_transforms: dict[str, Callable] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        root : Path or str or None, optional
            Processed data directory relative to the project root. If None,
            'data/{city}/processed'.
        city : str, optional
            City folder under data/; used only if 'root' is None.
        split : {'train', 'val'}, optional
            Dataset split.
        embedding : str, optional
            Name of the precomputed embeddings file in '{root}/text_emb', without the extension
            (the model name with '/' replaced by '__', plus '_d<dim>' if truncated).
        limit : int or None, optional
            If not None, keep only the first 'limit' elements.
        shuffle_index : bool, optional
            Shuffle the elements with a fixed seed (42), before 'limit'.
        instance_transforms : dict[str, Callable] or None, optional
            Transforms applied to an instance, by tensor name.
        """
        if root is None:
            root = get_root() / 'data' / city / 'processed'
        else:
            root = get_root() / root

        with open(root / f'{split}.csv', newline='', encoding='utf-8') as f:
            coords = np.array([(float(r['x']), float(r['y'])) for r in csv.DictReader(f)])

        stored = load_file(root / 'text_emb' / f'{embedding}.safetensors')
        text_id = stored[f'{split}_text_id']
        assert len(text_id) == len(coords), (
            f'{split}.csv has {len(coords)} rows, but the embeddings have {len(text_id)}: '
            'rerun scripts/precompute_text_embeddings.py'
        )

        rows = np.arange(len(coords))
        if shuffle_index:
            # Local generator: the global RNG state (the configured seed) stays untouched.
            rows = np.random.default_rng(42).permutation(rows)
        if limit is not None:
            rows = rows[:limit]

        self.coords = torch.from_numpy(coords[rows])
        self.text_id = text_id[torch.from_numpy(rows)]
        self.text_emb = stored['emb']  # (N_unique, dim), shared by all the rows
        self.instance_transforms = instance_transforms

    def __getitem__(self, index: int) -> dict[str, Any]:
        text_id = self.text_id[index]
        instance_data = {
            'coords': self.coords[index],
            'text_emb': self.text_emb[text_id],
            'text_id': text_id,
        }
        return self.preprocess_data(instance_data)

    def __len__(self) -> int:
        return len(self.coords)

    def preprocess_data(self, instance_data: dict[str, Any]) -> dict[str, Any]:
        """
        Apply instance transforms to the tensors by name.
        """
        if self.instance_transforms is not None:
            for name, transform in self.instance_transforms.items():
                if name in instance_data:
                    instance_data[name] = transform(instance_data[name])
        return instance_data
