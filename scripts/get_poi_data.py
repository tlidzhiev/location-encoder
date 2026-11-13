"""
Build a city POI dataset of (coordinates, text) pairs for CaLLiPer / CaLLiPer+.

Sources
  * OpenStreetMap  - via OSMnx (Overpass API)
  * Overture Maps  - "place" type, via the official `overturemaps` package

Output (one row per POI)
  lon, lat            WGS84 coordinates (EPSG:4326)
  x, y                optional projected coordinates (--projected-crs)
  text                the description fed to the CaLLiPer text encoder
  name                POI name ("" if unnamed)
  category            fine-grained, human-readable category ("fast food", "art museum")
  category_coarse     coarse group (OSM key such as "amenity"/"shop", or Overture basic_category)
  source              "osm" or "overture"
  source_id           OSM "node/123" or Overture GERS id
  confidence          Overture confidence score 0-1 (empty for OSM)
  sources             upstream datasets of the record, ";"-separated: "osm" for OSM,
                      e.g. "Foursquare;meta" or "BrightQuery" (a company registry) for Overture

Everything is written to data/<city>/raw/ (data/ is next to src/): the output file and the
OSMnx HTTP cache (data/<city>/raw/cache/). <city> is the lowercased city name before the first
comma ("London, UK" -> london); with --bbox it is given by --name.

Install
  uv sync --group data

Examples
  # CaLLiPer+ text ("A place of <category>, a type of <coarse>, named <name>."), both sources
  # -> data/london/raw/pois.csv
  uv run scripts/get_poi_data.py --city "London, UK" --projected-crs EPSG:27700

  # Original CaLLiPer text (categories only), OSM only -> data/vienna/raw/vienna_pois.csv
  uv run scripts/get_poi_data.py --city "Vienna, Austria" --sources osm --text-mode calliper \\
      --out vienna_pois.csv

  # Skip geocoding and use an explicit bbox (minx miny maxx maxy) -> data/london/raw/pois.csv
  uv run scripts/get_poi_data.py --bbox -0.51 51.28 0.33 51.69 --name london

Notes
  * Overture removed the legacy `categories` field in the Sept-2026 release; this
    script reads `taxonomy.primary` / `basic_category` and falls back to
    `categories.primary` only for older releases.
  * Overture places do not come from OSM, so both sources overlap heavily
    (same shop, slightly different coordinates). By default, an Overture POI is
    dropped when an OSM POI with a similar name lies within --dedup-radius metres.
"""

import argparse
import logging
import random
import re
import sys
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

log = logging.getLogger('poi')

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'  # next to src/

# --------------------------------------------------------------------------- #
# Progress reporting
# --------------------------------------------------------------------------- #


def format_elapsed(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f'{hours}h{minutes:02d}m{secs:02d}s'
    return f'{minutes}m{secs:02d}s' if minutes else f'{secs}s'


@contextmanager
def heartbeat(message: str, interval: float) -> Generator[None]:
    """Log a line every `interval` seconds while a blocking call without progress runs."""
    stop = threading.Event()
    start = time.monotonic()

    def beat() -> None:
        while not stop.wait(interval):
            log.info(
                '%s - still running, %s elapsed', message, format_elapsed(time.monotonic() - start)
            )

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


class Stages:
    """Numbered stages with per-stage and total timings: "[3/8] ... done in 42s (total 2m10s)"."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.index = 0
        self.start = time.monotonic()

    @contextmanager
    def stage(self, name: str, heartbeat_every: float | None = 30) -> Generator[None]:
        self.index += 1
        label = f'[{self.index}/{self.total}] {name}'
        log.info('%s ...', label)
        began = time.monotonic()
        with heartbeat(label, heartbeat_every) if heartbeat_every else nullcontext():
            yield
        now = time.monotonic()
        log.info(
            '%s done in %s (total %s)',
            label,
            format_elapsed(now - began),
            format_elapsed(now - self.start),
        )


# --------------------------------------------------------------------------- #
# OSM configuration
# --------------------------------------------------------------------------- #

# Tag keys that define a POI, in priority order: when a feature has several
# (e.g. amenity=cafe + shop=coffee), the first key wins for its category.
OSM_POI_KEYS = [
    'amenity',
    'shop',
    'tourism',
    'leisure',
    'healthcare',
    'office',
    'craft',
    'historic',
    'sport',
    'railway',
    'public_transport',
    'aeroway',
    'emergency',
    'man_made',
]

# Only keep these values for "infrastructure-ish" keys; everything else under
# them is street furniture or network geometry rather than a place.
OSM_KEY_VALUE_WHITELIST = {
    'railway': {'station', 'halt', 'tram_stop', 'subway_entrance'},
    'public_transport': {'station'},
    'aeroway': {'aerodrome', 'terminal', 'helipad'},
    'emergency': {'ambulance_station'},
    'man_made': {'lighthouse', 'observatory', 'windmill', 'watermill', 'tower', 'works'},
}

# Values that are rarely meaningful "places" and mostly add noise.
OSM_EXCLUDE_VALUES = {
    'amenity': {
        'bench',
        'waste_basket',
        'waste_disposal',
        'recycling',
        'parking_space',
        'parking_entrance',
        'bicycle_parking',
        'motorcycle_parking',
        'vending_machine',
        'drinking_water',
        'telephone',
        'post_box',
        'clock',
        'grit_bin',
        'hunting_stand',
        'loading_dock',
        'letter_box',
        'water_point',
        'shelter',
        'bbq',
        'fountain',
        'toilets',
        'charging_station',
        'atm',
        'parcel_locker',
        'lounger',
        'feeding_place',
    },
    'leisure': {'picnic_table', 'slipway', 'firepit', 'bleachers', 'outdoor_seating'},
    'tourism': {'information'},
    'historic': {'boundary_stone', 'milestone'},
}

# Overpass rejects (HTTP 413) queries whose polygon filter is too large, and a detailed
# city boundary has ~10k vertices. The query polygon is simplified down to this many vertices.
OSM_QUERY_MAX_VERTICES = 1000
OSM_QUERY_MIN_TOLERANCE = 0.0005  # degrees, ~50 m
OSM_TILE_MIN_AREA = 1e-7  # square degrees; smaller tile slivers are skipped

GENERIC_VALUES = {'yes', 'no', 'user defined', 'unknown', 'other', 'none'}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def humanize(label: Any) -> str:
    """'fast_food' -> 'fast food', 'eat_and_drink' -> 'eat and drink'."""
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return ''
    s = str(label).strip().replace('_', ' ').replace(';', ', ')
    return re.sub(r'\s+', ' ', s).lower()


def slugify(name: str) -> str:
    """'London, UK' -> 'london', 'New York, USA' -> 'new_york'."""
    city = name.split(',')[0].strip().lower()
    return re.sub(r'\W+', '_', city).strip('_')


def clean_name(name: Any) -> str:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ''
    return re.sub(r'\s+', ' ', str(name)).strip()


def to_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Represent polygons/lines by a point guaranteed to lie on the feature."""
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    non_point = gdf.geometry.geom_type != 'Point'
    if non_point.any():
        gdf.loc[non_point, 'geometry'] = gdf.loc[non_point, 'geometry'].representative_point()
    return gdf


def get_area(city: str | None, bbox: list[float] | None):
    """Return (shapely polygon, (minx, miny, maxx, maxy)) in EPSG:4326."""
    if bbox:
        poly = box(*bbox)
        return poly, tuple(bbox)
    if city is None:
        raise ValueError('Either city or bbox is required')
    import osmnx as ox

    log.info('Geocoding %r with Nominatim ...', city)
    area = ox.geocode_to_gdf(city).to_crs(4326)
    poly = area.geometry.union_all() if hasattr(area.geometry, 'union_all') else area.unary_union
    return poly, tuple(area.total_bounds)


# --------------------------------------------------------------------------- #
# OpenStreetMap
# --------------------------------------------------------------------------- #


def simplify_for_query(polygon, max_vertices: int = OSM_QUERY_MAX_VERTICES):
    """
    Coarsen the polygon until it has at most max_vertices vertices. It is first buffered by the
    tolerance, so the result still covers the original; clip the results to the original.
    """
    tolerance = OSM_QUERY_MIN_TOLERANCE
    simplified = polygon
    while shapely.get_num_coordinates(simplified) > max_vertices:
        simplified = polygon.buffer(tolerance).simplify(tolerance, preserve_topology=True)
        tolerance *= 2
    return simplified


def make_tiles(polygon, grid: int) -> list:
    """Split the polygon into at most grid x grid pieces by a regular grid over its bounds."""
    if grid <= 1:
        return [polygon]
    minx, miny, maxx, maxy = polygon.bounds
    xs = np.linspace(minx, maxx, grid + 1)
    ys = np.linspace(miny, maxy, grid + 1)
    tiles = []
    for i in range(grid):
        for j in range(grid):
            part = polygon.intersection(box(xs[i], ys[j], xs[i + 1], ys[j + 1]))
            pieces = [
                g
                for g in getattr(part, 'geoms', [part])
                if g.geom_type in ('Polygon', 'MultiPolygon') and g.area > OSM_TILE_MIN_AREA
            ]
            if pieces:
                tiles.append(shapely.union_all(pieces))
    return tiles


def fetch_features(polygon, key: str, attempts: int = 3) -> gpd.GeoDataFrame | None:
    """OSM features with the tag key inside the polygon; None if there are none."""
    import osmnx as ox
    from osmnx._errors import InsufficientResponseError, ResponseStatusCodeError

    tags: dict[str, bool | str | list[str]] = {key: True}
    for attempt in range(1, attempts + 1):
        try:
            return ox.features_from_polygon(polygon, tags=tags)
        except InsufficientResponseError:
            return None
        except ResponseStatusCodeError as e:  # Overpass is busy / timed out
            if attempt == attempts:
                raise
            wait = 30 * attempt
            log.warning(
                'Overpass error (%s), retry %d/%d in %ds', str(e)[:80], attempt, attempts, wait
            )
            time.sleep(wait)
    return None


def fetch_osm(polygon, tiles: int) -> gpd.GeoDataFrame:
    import osmnx as ox

    ox.settings.requests_timeout = 600
    ox.settings.use_cache = True

    # One Overpass query per (tile, tag key); their union equals a single query for the whole
    # area. Small queries do not hit the server limits, give a progress bar with a meaningful
    # ETA, and a failed run resumes from the OSMnx cache instead of starting over.
    tile_polygons = make_tiles(simplify_for_query(polygon), tiles)
    jobs = [(t, key) for t in range(len(tile_polygons)) for key in OSM_POI_KEYS]
    random.Random(0).shuffle(jobs)  # dense and sparse tiles mixed: the ETA converges faster
    log.info(
        '%d tiles x %d tag keys = %d Overpass queries',
        len(tile_polygons),
        len(OSM_POI_KEYS),
        len(jobs),
    )

    frames = []
    num_features = 0
    with tqdm(jobs, desc='OSM queries', unit='query', postfix={'features': 0}) as bar:
        for t, key in bar:
            found = fetch_features(tile_polygons[t], key)
            if found is not None:
                frames.append(found.reset_index())
                num_features += len(found)
            bar.set_postfix(tile=t, key=key, features=num_features)
    if not frames:
        log.warning('OSM query returned nothing')
        return gpd.GeoDataFrame(geometry=[], crs=4326)

    gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True))
    # A feature is returned once per tag key and per tile it touches
    id_cols = ['element', 'id'] if 'element' in gdf.columns else ['element_type', 'osmid']
    return gpd.GeoDataFrame(gdf.drop_duplicates(subset=id_cols, ignore_index=True))


def osm_category(row: pd.Series) -> tuple[str, str] | None:
    """Pick (coarse_key, value) for an OSM feature, or None if it is not a POI."""
    for key in OSM_POI_KEYS:
        if key not in row.index:
            continue
        val = row[key]
        if not isinstance(val, str) or not val.strip():
            continue
        val = val.split(';')[0].strip()
        if key in OSM_KEY_VALUE_WHITELIST and val not in OSM_KEY_VALUE_WHITELIST[key]:
            continue
        if val in OSM_EXCLUDE_VALUES.get(key, set()):
            continue
        if val.lower() in GENERIC_VALUES:
            # e.g. shop=yes -> "shop", office=yes -> "office"
            return key, key
        return key, val
    return None


def process_osm(raw: gpd.GeoDataFrame, name_lang: str | None, polygon) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    raw = to_points(raw)
    raw = raw[raw.geometry.within(polygon)]

    cats = raw.apply(osm_category, axis=1)
    keep = cats.notna()
    raw, cats = raw[keep], cats[keep]

    name_cols = [
        c for c in ([f'name:{name_lang}'] if name_lang else []) + ['name', 'brand'] if c in raw
    ]
    names = pd.Series('', index=raw.index)
    for c in reversed(name_cols):  # earlier columns take precedence
        v = raw[c].map(clean_name)
        names = v.where(v != '', names)

    if 'element' in raw.columns:  # osmnx >= 2.0
        sid = raw['element'].astype(str) + '/' + raw['id'].astype(str)
    else:  # osmnx 1.x
        sid = raw['element_type'].astype(str) + '/' + raw['osmid'].astype(str)

    out = pd.DataFrame(
        {
            'lon': raw.geometry.x.values,
            'lat': raw.geometry.y.values,
            'name': names.values,
            'category': [humanize(v) for _, v in cats],
            'category_coarse': [humanize(k) for k, _ in cats],
            'source': 'osm',
            'source_id': sid.values,
            'confidence': np.nan,
            'sources': 'osm',
        }
    )
    log.info('OSM: %d POIs after filtering', len(out))
    return out


# --------------------------------------------------------------------------- #
# Overture Maps
# --------------------------------------------------------------------------- #


def fetch_overture(bbox, release: str | None) -> gpd.GeoDataFrame:
    from overturemaps import core

    if release is None:
        release = core.get_latest_release()
    log.info('Downloading Overture places (release %s) for bbox %s ...', release, bbox)
    gdf = gpd.GeoDataFrame(core.geodataframe('place', bbox=tuple(bbox), release=release))
    gdf.set_crs(4326, allow_override=True, inplace=True)
    return gdf


def _get(d: Any, key: str):
    return d.get(key) if isinstance(d, dict) else None


def overture_sources(sources: Any) -> str:
    """Upstream datasets of an Overture record, e.g. 'Foursquare;meta'."""
    if sources is None or (isinstance(sources, float) and np.isnan(sources)):
        return ''
    # 'Overture' entries only describe how the confidence was calculated
    datasets = {_get(s, 'dataset') for s in sources} - {None, 'Overture'}
    return ';'.join(sorted(datasets))


def overture_category(row: pd.Series, mode: str) -> tuple[str, str]:
    """Return (fine category, coarse category) from taxonomy / basic_category / categories."""
    tax = row.get('taxonomy')
    basic = row.get('basic_category')
    legacy = row.get('categories')

    primary = _get(tax, 'primary') or _get(legacy, 'primary')
    hierarchy = _get(tax, 'hierarchy')
    top = hierarchy[0] if hierarchy is not None and len(hierarchy) else None

    basic = basic if isinstance(basic, str) and basic else None
    fine = (basic or primary) if mode == 'basic' else (primary or basic)
    coarse = basic or top or primary
    return humanize(fine), humanize(coarse)


def process_overture(
    raw: gpd.GeoDataFrame, min_conf: float, name_lang: str | None, cat_mode: str, polygon
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    raw = to_points(raw)
    raw = raw[raw.geometry.within(polygon)]

    if 'confidence' in raw:
        raw = raw[raw['confidence'].fillna(0) >= min_conf]
    if 'operating_status' in raw:
        raw = raw[raw['operating_status'].isna() | (raw['operating_status'] == 'open')]

    def pick_name(n):
        if not isinstance(n, dict):
            return ''
        if name_lang:
            common = n.get('common')
            if isinstance(common, dict) and common.get(name_lang):
                return clean_name(common[name_lang])
            if isinstance(common, (list, np.ndarray)):  # list of (key, value) pairs
                for kv in common:
                    if isinstance(kv, (tuple, list)) and kv[0] == name_lang:
                        return clean_name(kv[1])
        return clean_name(n.get('primary'))

    cats = raw.apply(lambda r: overture_category(r, cat_mode), axis=1)
    out = pd.DataFrame(
        {
            'lon': raw.geometry.x.values,
            'lat': raw.geometry.y.values,
            'name': raw['names'].map(pick_name).values if 'names' in raw else '',
            'category': [c[0] for c in cats],
            'category_coarse': [c[1] for c in cats],
            'source': 'overture',
            'source_id': raw['id'].astype(str).values,
            'confidence': raw['confidence'].values if 'confidence' in raw else np.nan,
            'sources': raw['sources'].map(overture_sources).values if 'sources' in raw else '',
        }
    )
    out = out[out['category'] != '']
    log.info(
        'Overture: %d POIs after confidence>=%.2f / open-status / boundary filters',
        len(out),
        min_conf,
    )
    return out


# --------------------------------------------------------------------------- #
# Cross-source deduplication
# --------------------------------------------------------------------------- #


def _norm(s: str) -> str:
    return re.sub(r'[^\w\s]', '', s.lower()).strip()


def _similarity(a: str, b: str) -> float:
    try:
        from rapidfuzz import fuzz

        return fuzz.token_set_ratio(a, b)
    except ImportError:
        from difflib import SequenceMatcher

        return 100 * SequenceMatcher(None, a, b).ratio()


def dedup_overture_against_osm(
    osm: pd.DataFrame, ovt: pd.DataFrame, radius_m: float, name_thresh: float
) -> pd.DataFrame:
    """Drop Overture POIs that have a similarly named OSM POI within radius_m metres."""
    if osm.empty or ovt.empty or radius_m <= 0:
        return ovt
    a = gpd.GeoDataFrame(osm[['name']], geometry=gpd.points_from_xy(osm.lon, osm.lat), crs=4326)
    b = gpd.GeoDataFrame(ovt[['name']], geometry=gpd.points_from_xy(ovt.lon, ovt.lat), crs=4326)
    utm = a.estimate_utm_crs()
    a, b = a.to_crs(utm), b.to_crs(utm)
    a = a[a['name'] != '']
    b_named = b[b['name'] != '']
    if a.empty or b_named.empty:
        return ovt

    pairs = gpd.sjoin(
        b_named, a, predicate='dwithin', distance=radius_m, lsuffix='ovt', rsuffix='osm'
    )
    if pairs.empty:
        return ovt
    sim = [
        _similarity(_norm(x), _norm(y))
        for x, y in tqdm(
            zip(pairs['name_ovt'], pairs['name_osm']),
            total=len(pairs),
            desc='Name similarity',
            unit='pair',
            leave=False,
        )
    ]
    dup_idx = set(pairs.index[np.asarray(sim) >= name_thresh])
    log.info(
        'Dedup: removed %d Overture POIs matching an OSM POI (<=%gm, name sim>=%g)',
        len(dup_idx),
        radius_m,
        name_thresh,
    )
    return ovt.loc[~ovt.index.isin(dup_idx)]


# --------------------------------------------------------------------------- #
# Text construction
# --------------------------------------------------------------------------- #

TEXT_MODES = {
    # CaLLiPer+ (Liu et al., GIScience 2025, eq. 1): "A place of [t2], a type of [t1], named [n]."
    'calliper_plus': (
        'A place of {category}, a type of {category_coarse}, named {name}.',
        'A place of {category}, a type of {category_coarse}.',
    ),
    # original CaLLiPer (Wang et al., CEUS 2025): the two category levels only
    'calliper': (
        'A place of {category}, a type of {category_coarse}',
        'A place of {category}, a type of {category_coarse}',
    ),
    # short forms
    'category': ('{category}', '{category}'),
    'name_category': ('{name}, {category}', '{category}'),
    # sentence form, sometimes better for sentence encoders
    'sentence': ('{article} {category} named {name}.', '{article} {category}.'),
}


def build_text(df: pd.DataFrame, template: str, template_noname: str) -> pd.Series:
    def fmt(r):
        t = template if r['name'] else template_noname
        if r['category'] == r['category_coarse']:
            # Overture often has no finer level: avoid "A place of cafe, a type of cafe"
            t = t.replace(', a type of {category_coarse}', '')
        article = 'An' if r['category'][:1] in 'aeiou' else 'A'
        return t.format(
            name=r['name'],
            category=r['category'],
            category_coarse=r['category_coarse'],
            article=article,
        )

    return df.apply(fmt, axis=1)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def num_stages(args) -> int:
    """Stages of main(): area, [OSM download, process], [Overture download, process, dedup], text, write."""
    return 1 + 2 * ('osm' in args.sources) + 3 * ('overture' in args.sources) + 2


def build_dataset(args, stages: Stages) -> pd.DataFrame:
    with stages.stage('Resolving the area', heartbeat_every=None):
        polygon, bbox = get_area(args.city, args.bbox)

    parts = []
    osm = ovt = pd.DataFrame()
    if 'osm' in args.sources:
        with stages.stage('Downloading OSM (Overpass)', heartbeat_every=None):
            raw_osm = fetch_osm(polygon, args.tiles)
        with stages.stage('Processing OSM'):
            osm = process_osm(raw_osm, args.name_lang, polygon)
    if 'overture' in args.sources:
        # The download has no progress reporting: only the elapsed time can be shown
        with stages.stage('Downloading Overture places (no progress available)'):
            raw_ovt = fetch_overture(bbox, args.overture_release)
        with stages.stage('Processing Overture'):
            ovt = process_overture(
                raw_ovt,
                args.min_confidence,
                args.name_lang,
                args.overture_category,
                polygon,
            )
        with stages.stage('Deduplicating Overture against OSM'):
            ovt = dedup_overture_against_osm(osm, ovt, args.dedup_radius, args.dedup_name_sim)
    parts = [p for p in (osm, ovt) if not p.empty]
    if not parts:
        raise SystemExit('No POIs found - check the city name / bbox.')
    df = pd.concat(parts, ignore_index=True)

    if args.require_name:
        df = df[df['name'] != '']

    # Exact duplicates within a source (same coords + name + category)
    df = df.drop_duplicates(subset=['lon', 'lat', 'name', 'category'])

    with stages.stage('Building texts'):
        if args.text_template:
            tpl, tpl_noname = args.text_template, args.text_template_noname or '{category}'
        else:
            tpl, tpl_noname = TEXT_MODES[args.text_mode]
        df['text'] = build_text(df, tpl, tpl_noname)

        if args.projected_crs:
            pts = gpd.GeoSeries(gpd.points_from_xy(df.lon, df.lat), crs=4326).to_crs(
                args.projected_crs
            )
            df['x'], df['y'] = pts.x.values, pts.y.values

    df['lon'], df['lat'] = df['lon'].round(7), df['lat'].round(7)
    cols = (
        ['lon', 'lat']
        + (['x', 'y'] if args.projected_crs else [])
        + ['text', 'name', 'category', 'category_coarse', 'source', 'source_id']
        + ['confidence', 'sources']
    )
    return df[cols].reset_index(drop=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    area = p.add_mutually_exclusive_group(required=True)
    area.add_argument('--city', help='Place name for Nominatim, e.g. "London, UK"')
    area.add_argument(
        '--bbox',
        nargs=4,
        type=float,
        metavar=('MINX', 'MINY', 'MAXX', 'MAXY'),
        help='WGS84 bounding box instead of a city name',
    )
    p.add_argument(
        '--name',
        help='City folder name under data/ (default: derived from --city; required with --bbox)',
    )
    p.add_argument(
        '--data-dir', type=Path, default=DATA_DIR, help='Root of the data folders (default: data/)'
    )
    p.add_argument(
        '--out',
        default='pois.csv',
        help='Output .csv, .parquet or .geojson, relative to <data-dir>/<city>/raw',
    )
    p.add_argument(
        '--tiles',
        type=int,
        default=4,
        help='Download OSM in a TILES x TILES grid of small Overpass queries (1 = a single area)',
    )
    p.add_argument('--sources', nargs='+', choices=['osm', 'overture'], default=['osm', 'overture'])
    p.add_argument(
        '--text-mode',
        choices=list(TEXT_MODES),
        default='calliper_plus',
        help='Preset text format (default: calliper_plus, the template of the CaLLiPer+ paper)',
    )
    p.add_argument(
        '--text-template',
        help='Custom template, e.g. "{name} ({category})". '
        'Fields: {name} {category} {category_coarse} {article}',
    )
    p.add_argument('--text-template-noname', help='Template used when the POI has no name')
    p.add_argument('--require-name', action='store_true', help='Drop POIs without a name')
    p.add_argument('--name-lang', help="Prefer names in this language, e.g. 'en' (name:en)")
    p.add_argument('--overture-release', help='e.g. 2026-10-22.0 (default: latest)')
    p.add_argument(
        '--overture-category',
        choices=['primary', 'basic'],
        default='primary',
        help='Fine taxonomy.primary (~2000 labels) or basic_category (~300 labels)',
    )
    p.add_argument(
        '--min-confidence',
        type=float,
        default=0.0,
        help='Minimum Overture confidence score (0-1); the score is saved, so filter it later',
    )
    p.add_argument(
        '--dedup-radius',
        type=float,
        default=50,
        help='Metres; 0 disables OSM/Overture deduplication',
    )
    p.add_argument(
        '--dedup-name-sim',
        type=float,
        default=85,
        help='Name similarity (0-100) above which two nearby POIs are duplicates',
    )
    p.add_argument('--projected-crs', help='Also output x/y in this CRS, e.g. EPSG:27700')
    p.add_argument('-v', '--verbose', action='store_true')
    args = p.parse_args(argv)
    args.city_dir = slugify(args.name or args.city or '')
    if not args.city_dir:
        p.error('--name is required with --bbox')
    return args


def setup_osmnx(cache_dir: Path, verbose: bool) -> None:
    import osmnx as ox

    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache_dir)
    ox.settings.log_console = verbose  # Overpass requests, pauses and rate-limit waits


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
    )
    raw_dir = args.data_dir / args.city_dir / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / args.out
    setup_osmnx(raw_dir / 'cache', args.verbose)
    stages = Stages(num_stages(args))

    with logging_redirect_tqdm():  # log lines do not break the progress bars
        df = build_dataset(args, stages)
        with stages.stage('Writing the dataset', heartbeat_every=None):
            if out.suffix == '.parquet':
                df.to_parquet(out, index=False)
            elif out.suffix in ('.geojson', '.gpkg'):
                gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326).to_file(
                    out
                )
            else:
                df.to_csv(out, index=False)

    log.info('Wrote %d POIs to %s', len(df), out)
    log.info('By source: %s', df['source'].value_counts().to_dict())
    log.info(
        'Unique categories: %d | named: %.1f%%',
        df['category'].nunique(),
        100 * (df['name'] != '').mean(),
    )
    print(df.head(10).to_string(index=False))


if __name__ == '__main__':
    sys.exit(main())
