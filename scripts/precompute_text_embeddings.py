"""
Precompute the text embeddings of the processed POI dataset for PrecomputedTextEncoder.

Reads  data/<city>/processed/train.csv and val.csv (the 'text' column)
Writes data/<city>/processed/text_emb/<model>.safetensors with tensors
       'emb'            (N_unique, dim) L2-normalized float16, one row per distinct text
       'train_text_id'  (len(train),)   int64, row of 'emb' for each train.csv row
       'val_text_id'    (len(val),)     int64, row of 'emb' for each val.csv row

Each distinct text of train and val is embedded once (many POIs of a chain share a text).
A POI's embedding is emb[text_id]; equal text_id means equal text (false negatives in the
contrastive loss).

Examples
  uv run scripts/precompute_text_embeddings.py
  uv run scripts/precompute_text_embeddings.py --model sentence-transformers/all-MiniLM-L6-v2
  uv run scripts/precompute_text_embeddings.py --truncate-dim 384 --device cuda
  uv run scripts/precompute_text_embeddings.py --dtype float32
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

log = logging.getLogger('poi')

# Strong open embedding model with sentence-transformers support: last-token pooling,
# 1024-d output that can be truncated (Matryoshka) to 32..1024 dimensions.
DEFAULT_MODEL_NAME = 'Qwen/Qwen3-Embedding-0.6B'

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'  # next to src/
SPLITS = ('train', 'val')
DTYPES = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}


def default_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


@torch.no_grad()
def precompute_text_embeddings(
    texts: list[str],
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 128,
    device: str | None = None,
    truncate_dim: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Embed texts with the frozen backbone loaded in 'dtype': (len(texts), dim), L2-normalized
    (in float32, before the cast), float16, on the CPU.
    """
    backbone = SentenceTransformer(model_name, device=device, model_kwargs={'dtype': dtype})
    backbone.eval()
    # similar lengths in a batch: less padding, much faster
    order = np.argsort([len(t) for t in texts], kind='stable')
    result = None  # allocated once the embedding size is known; filled in the input order
    for start in tqdm(range(0, len(texts), batch_size), desc='Embedding texts', unit='batch'):
        rows = order[start : start + batch_size]
        tokens = backbone.tokenize([texts[i] for i in rows])
        features = backbone(
            {
                'input_ids': tokens['input_ids'].to(backbone.device),
                'attention_mask': tokens['attention_mask'].to(backbone.device),
            }
        )
        embedding = features['sentence_embedding'][:, :truncate_dim]
        embedding = F.normalize(embedding.float(), p=2, dim=1).half().cpu()
        if result is None:
            result = torch.empty(len(texts), embedding.shape[1], dtype=torch.float16)
        result[torch.from_numpy(rows)] = embedding
    assert result is not None, 'no texts to embed'
    return result


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--city', default='london', help='City folder under data/')
    p.add_argument('--data-dir', type=Path, default=DATA_DIR)
    p.add_argument('--model', default=DEFAULT_MODEL_NAME, help='sentence-transformers model')
    p.add_argument('--truncate-dim', type=int, help='Keep the first N (Matryoshka) dimensions')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument(
        '--dtype',
        default='bfloat16',
        choices=sorted(DTYPES),
        help='Precision of the backbone (the embeddings are stored in float16 anyway)',
    )
    p.add_argument('--device', default=default_device())
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S'
    )
    processed = args.data_dir / args.city / 'processed'
    texts = {
        split: pd.read_csv(processed / f'{split}.csv', keep_default_na=False, usecols=['text'])[
            'text'
        ]
        for split in SPLITS
    }
    unique = pd.Index(pd.concat(texts.values()).unique())
    log.info(
        '%s rows, %d distinct texts | model %s on %s',
        ' + '.join(f'{len(t)} {s}' for s, t in texts.items()),
        len(unique),
        args.model,
        args.device,
    )

    embeddings = precompute_text_embeddings(
        unique.tolist(),
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        truncate_dim=args.truncate_dim,
        dtype=DTYPES[args.dtype],
    )

    tensors = {'emb': embeddings.contiguous()} | {
        f'{split}_text_id': torch.from_numpy(unique.get_indexer(t)).long()
        for split, t in texts.items()
    }
    out_dir = processed / 'text_emb'
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f'_d{args.truncate_dim}' if args.truncate_dim else ''
    out = out_dir / f'{args.model.replace("/", "__")}{suffix}.safetensors'
    save_file(
        tensors,
        out,
        metadata={
            'model': args.model,
            'dim': str(embeddings.shape[1]),
            'dtype': args.dtype,
            'source': str(processed),
        },
    )
    log.info('wrote %s: %s', out, {s: tuple(t.shape) for s, t in tensors.items()})


if __name__ == '__main__':
    main()
