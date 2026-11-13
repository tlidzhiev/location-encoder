"""
Filter the raw POI dataset of get_poi_data.py and split it into train / val for CaLLiPer.

Reads  data/<city>/raw/pois.csv
Writes data/<city>/processed/train.csv and val.csv

Filters (each one is logged with the number of removed rows; all can be switched off)
  1. points with many POIs  - a coordinate shared by more than --max-per-point POIs is dropped:
                              these are registered company addresses and virtual offices, where
                              the location says nothing about the POI
  2. low confidence         - Overture confidence below --min-confidence. Records known only to a
                              company registry (BrightQuery) all get exactly 0.5 ("unverified"),
                              so the default 0.51 removes them together with the weak matches
  3. registry-only records  - records whose only upstream dataset is in --registry-sources
                              (company registries such as BrightQuery, not real places)
  4. service categories     - coarse categories in --drop-coarse (businesses without a storefront)

Split
  Spatial: the area is cut into --block-size metre blocks and whole blocks go to val, so that
  neighbouring POIs (the same building, the same street) never end up in both parts.

Examples
  uv run scripts/process_poi_data.py
  uv run scripts/process_poi_data.py --city london --max-per-point 5 --val-fraction 0.05
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger('poi')

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'  # next to src/


def drop(df: pd.DataFrame, keep: pd.Series, reason: str) -> pd.DataFrame:
    log.info('%-58s -%7d rows -> %7d', reason, int((~keep).sum()), int(keep.sum()))
    return df[keep]


def filter_pois(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    log.info('%-58s %15s -> %7d', 'raw dataset', '', len(df))

    if args.max_per_point is not None:
        per_point = df.groupby(['lon', 'lat'])['name'].transform('size')
        df = drop(df, per_point <= args.max_per_point, f'points with > {args.max_per_point} POIs')

    if args.min_confidence > 0:
        # OSM rows have no confidence and are kept
        df = drop(
            df,
            df['confidence'].fillna(1.0) >= args.min_confidence,
            f'confidence < {args.min_confidence}',
        )

    if args.registry_sources:
        registry = set(args.registry_sources)
        only_registry = df['sources'].map(lambda s: bool(s) and set(s.split(';')) <= registry)
        df = drop(df, ~only_registry, f'only from {sorted(registry)}')

    if args.drop_coarse:
        df = drop(
            df, ~df['category_coarse'].isin(args.drop_coarse), f'coarse in {args.drop_coarse}'
        )

    return df.reset_index(drop=True)


def split_by_blocks(
    df: pd.DataFrame, val_fraction: float, block_size: float, seed: int, num_strata: int = 20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Whole spatial blocks go to val. Blocks are grouped by their POI count and every group gives
    val_fraction of its rows: the city centre has a few huge blocks, so a plain random choice of
    blocks would put the centre into val (or leave it out) by chance.
    """
    cells = zip(
        (df['x'] // block_size).astype(int), (df['y'] // block_size).astype(int), strict=True
    )
    block_ids = pd.Series(list(cells)).factorize()[0]
    sizes = np.bincount(block_ids)

    rng = np.random.default_rng(seed)
    val_blocks = []
    for stratum in np.array_split(np.argsort(sizes, kind='stable'), num_strata):
        stratum = rng.permutation(stratum)
        cumulative = np.cumsum(sizes[stratum])
        # the number of blocks whose total size is the closest to the target
        count = int(np.abs(cumulative - val_fraction * cumulative[-1]).argmin()) + 1
        val_blocks.append(stratum[:count])
    is_val = np.isin(block_ids, np.concatenate(val_blocks))
    log.info(
        'split: %d blocks of %g m, %d of them in val',
        len(sizes),
        block_size,
        len(np.unique(block_ids[is_val])),
    )
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


def summarize(name: str, df: pd.DataFrame) -> None:
    log.info(
        '%-5s %8d rows | %5d categories | %6.1f%% named | x %.0f..%.0f | y %.0f..%.0f',
        name,
        len(df),
        df['category'].nunique(),
        100 * (df['name'] != '').mean(),
        df['x'].min(),
        df['x'].max(),
        df['y'].min(),
        df['y'].max(),
    )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--city', default='london', help='City folder under data/')
    p.add_argument('--data-dir', type=Path, default=DATA_DIR)
    p.add_argument('--input', default='pois.csv', help='File in <data-dir>/<city>/raw')
    p.add_argument(
        '--max-per-point',
        type=int,
        default=9,
        help='Drop coordinates shared by more POIs than this (default: 9; use -1 to disable)',
    )
    p.add_argument(
        '--min-confidence',
        type=float,
        default=0.51,
        help='Minimum Overture confidence (0-1); 0.5 is the "unverified" value (default: 0.51)',
    )
    p.add_argument(
        '--registry-sources',
        nargs='*',
        default=[],
        help='Drop records that come only from these datasets, e.g. BrightQuery',
    )
    p.add_argument(
        '--drop-coarse',
        nargs='*',
        default=['home service'],
        help='Coarse categories to drop (default: "home service"; pass no value to keep all)',
    )
    p.add_argument('--val-fraction', type=float, default=0.1)
    p.add_argument('--block-size', type=float, default=1000, help='Split block size, metres')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(argv)
    if args.max_per_point < 0:
        args.max_per_point = None
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S'
    )
    city_dir = args.data_dir / args.city
    source = city_dir / 'raw' / args.input
    df = pd.read_csv(
        source, keep_default_na=False, na_values=[''], dtype={'source_id': str, 'sources': str}
    )
    # empty strings are real values in these columns, not missing data
    for column in ('name', 'text', 'category', 'category_coarse', 'sources'):
        df[column] = df[column].fillna('')
    if not {'x', 'y'} <= set(df.columns):
        raise SystemExit('x / y columns are missing: run get_poi_data.py with --projected-crs')

    df = filter_pois(df, args)
    train, val = split_by_blocks(df, args.val_fraction, args.block_size, args.seed)

    out_dir = city_dir / 'processed'
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / 'train.csv', index=False)
    val.to_csv(out_dir / 'val.csv', index=False)

    summarize('train', train)
    summarize('val', val)
    unseen = ~val['category'].isin(set(train['category']))
    log.info('val rows with a category absent from train: %d', int(unseen.sum()))
    log.info('wrote %s', out_dir)


if __name__ == '__main__':
    main()
