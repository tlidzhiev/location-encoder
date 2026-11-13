# Location Encoder

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#data">Data</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a>
</p>

## About

Location encoders for geospatial representation learning: positional encodings, location networks and CLIP-style contrastive pre-training. The included example aligns the locations of places (POIs) with precomputed embeddings of their text descriptions, in the spirit of CaLLiPer.

## Installation

### Requirements
- Python 3.15.2
- CUDA (optional, for GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/tlidzhiev/location-encoder.git
cd location-encoder

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.15.2
source .venv/bin/activate

# Install dependencies via uv
uv sync

# Install pre-commit
pre-commit install
```

The scripts that download and process the POI data need extra packages (geopandas, osmnx, ...):

```bash
uv sync --extra download
```

### CometML Configuration

CometML is used by default for experiment tracking. Create a `.comet.config` file in the project root:

```ini
[comet]
api_key=YOUR_API_KEY
workspace=YOUR_WORKSPACE
project_name=YOUR_PROJECT_NAME
```

The API key is only needed for online logging: `writer.mode` is `offline` by default.

## Data

Run the scripts in this order. Everything is stored under `data/<city>/`.

```bash
# 1. Download the POIs (OpenStreetMap, Overture Maps) -> data/<city>/raw/pois.csv
uv run scripts/get_poi_data.py

# 2. Filter them and make the spatial train / val split -> data/<city>/processed/{train,val}.csv
uv run scripts/process_poi_data.py

# 3. (optional) Draw a map of the split -> data/<city>/processed/split_map.png
uv run scripts/plot_poi_split.py

# 4. Embed every distinct text once with a frozen backbone -> data/<city>/processed/text_emb/
uv run scripts/precompute_text_embeddings.py
```

The text backbone (Qwen3-Embedding-0.6B by default) is never trained and never runs during training: the model only learns a projection of the precomputed embeddings. Run each script with `--help` for the options.

## How To Use

To run a script (training or inference), use the following command:

```bash
uv run <script> -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

- `<script>`: Use `train` for training or `inference` for inference
- `CONFIG_NAME`: A config file from `src/lab/configs` (`train` and `inference` by default)
- `HYDRA_CONFIG_ARGUMENTS`: Optional Hydra overrides

Runs are saved to `logs/{project_name}/{run_name}`, with checkpoints in the `checkpoints` subdirectory.

## Credits

The code of this project is based on the [PyTorch project template](https://github.com/Blinorot/pytorch_project_template).
