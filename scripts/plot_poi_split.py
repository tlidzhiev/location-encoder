"""
Draw a map of the train / val split made by process_poi_data.py.

Reads  data/<city>/processed/train.csv and val.csv
Writes data/<city>/processed/split_map.png

Left: the whole city. Right: a zoom on the centre with the split blocks (--block-size must be
the same as in process_poi_data.py).

Examples
  uv run scripts/plot_poi_split.py
  uv run scripts/plot_poi_split.py --boundary "London, UK" --zoom-km 12
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'  # next to src/

# Categorical slots 1 and 2 of the reference palette (validated, both pairs pass)
SURFACE = '#fcfcfb'
TRAIN_COLOR = '#2a78d6'
VAL_COLOR = '#eb6834'
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
GRID = '#e3e2dd'


def load_split(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ['x', 'y']
    train = pd.read_csv(processed_dir / 'train.csv', usecols=columns)
    val = pd.read_csv(processed_dir / 'val.csv', usecols=columns)
    return train, val


def load_boundary(query: str, cache_dir: Path):
    """City outline in the metric CRS of the data, or None if it cannot be fetched."""
    import osmnx as ox

    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache_dir)
    try:
        return ox.geocode_to_gdf(query).to_crs(27700).geometry.union_all().boundary
    except OSError as e:  # no network and nothing in the cache
        print(f'boundary skipped: {e}')
        return None


def draw_boundary(ax, boundary, origin: tuple[float, float]) -> None:
    if boundary is None:
        return
    for line in getattr(boundary, 'geoms', [boundary]):
        xs, ys = line.xy
        ax.plot(
            (np.asarray(xs) - origin[0]) / 1000,
            (np.asarray(ys) - origin[1]) / 1000,
            color=INK_SECONDARY,
            linewidth=0.6,
            alpha=0.6,
            zorder=1,
        )


def draw_points(ax, train, val, origin: tuple[float, float], size: float) -> None:
    # train first, the small val set on top so that it stays visible
    for frame, color, alpha, z in ((train, TRAIN_COLOR, 0.3, 2), (val, VAL_COLOR, 0.8, 3)):
        ax.scatter(
            (frame['x'] - origin[0]) / 1000,
            (frame['y'] - origin[1]) / 1000,
            s=size,
            color=color,
            alpha=alpha,
            linewidths=0,
            zorder=z,
            rasterized=True,
        )


def style_axes(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_aspect('equal')
    ax.set_title(title, loc='left', fontsize=11, color=INK, pad=8)
    ax.set_xlabel('км на восток от центра', fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel('км на север от центра', fontsize=9, color=INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=0)
    ax.grid(color=GRID, linewidth=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--city', default='london', help='City folder under data/')
    p.add_argument('--data-dir', type=Path, default=DATA_DIR)
    p.add_argument('--block-size', type=float, default=1000, help='Split block size, metres')
    p.add_argument('--zoom-km', type=float, default=12, help='Side of the zoomed square, km')
    p.add_argument('--boundary', help='Place name for the city outline, e.g. "London, UK"')
    p.add_argument('--out', default='split_map.png', help='File in <data-dir>/<city>/processed')
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    city_dir = args.data_dir / args.city
    train, val = load_split(city_dir / 'processed')
    both = pd.concat([train, val])
    origin = (float(both['x'].median()), float(both['y'].median()))
    boundary = load_boundary(args.boundary, city_dir / 'raw' / 'cache') if args.boundary else None

    val_blocks = len(
        set(
            zip(
                (val['x'] // args.block_size).astype(int),
                (val['y'] // args.block_size).astype(int),
                strict=True,
            )
        )
    )
    block_km = args.block_size / 1000

    fig, (ax_city, ax_zoom) = plt.subplots(
        1, 2, figsize=(15, 7.6), gridspec_kw={'width_ratios': [1.25, 1]}, facecolor=SURFACE
    )

    # whole city
    draw_boundary(ax_city, boundary, origin)
    draw_points(ax_city, train, val, origin, size=0.4)
    style_axes(ax_city, 'Весь город')
    half = args.zoom_km / 2
    ax_city.add_patch(
        Rectangle(
            (-half, -half),
            args.zoom_km,
            args.zoom_km,
            fill=False,
            edgecolor=INK,
            linewidth=1.0,
            zorder=4,
        )
    )

    # zoom on the centre
    zoom_train = train[
        ((train['x'] - origin[0]).abs() < half * 1000)
        & ((train['y'] - origin[1]).abs() < half * 1000)
    ]
    zoom_val = val[
        ((val['x'] - origin[0]).abs() < half * 1000) & ((val['y'] - origin[1]).abs() < half * 1000)
    ]
    draw_points(ax_zoom, zoom_train, zoom_val, origin, size=3)
    style_axes(
        ax_zoom, f'Центр, {args.zoom_km:g} × {args.zoom_km:g} км (сетка — блоки {block_km:g} км)'
    )
    ax_zoom.set_xlim(-half, half)
    ax_zoom.set_ylim(-half, half)
    # block edges are aligned to the metric CRS grid, not to the plot origin
    shift_x = (-origin[0] / 1000) % block_km
    shift_y = (-origin[1] / 1000) % block_km
    for t in np.arange(-half - block_km, half + block_km, block_km):
        ax_zoom.axvline(t + shift_x, color=GRID, linewidth=0.6, zorder=1.5)
        ax_zoom.axhline(t + shift_y, color=GRID, linewidth=0.6, zorder=1.5)

    handles = [
        Line2D(
            [],
            [],
            marker='o',
            linestyle='',
            markersize=7,
            color=TRAIN_COLOR,
            label=f'train — {len(train):,} POI'.replace(',', ' '),
        ),
        Line2D(
            [],
            [],
            marker='o',
            linestyle='',
            markersize=7,
            color=VAL_COLOR,
            label=f'val — {len(val):,} POI в {val_blocks} блоках'.replace(',', ' '),
        ),
    ]
    fig.legend(
        handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=10, labelcolor=INK
    )
    fig.suptitle(
        'Пространственное разбиение: в val уходят целые блоки, соседние POI не попадают в обе части',
        x=0.01,
        ha='left',
        fontsize=13,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))

    out = city_dir / 'processed' / args.out
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
