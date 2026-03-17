# Hardware Dataset Clustering Visualization

An interactive web application that clusters dirty hardware inventory data using semantic embeddings and visualizes the results in a D3.js radial network graph. The backend runs live clustering with configurable thresholds and matches clusters to real-world devices from the iFixit catalog plus a custom device list.

## Features

### GUI

- **Interactive Network Graph**: Zoom, pan, and drag nodes. Hover for quick tooltips; click to open the detail panel.
- **Search Clusters**: Real-time search by category, subcategory, or record content (e.g., Router, iPhone, 4331).
- **Clustering Threshold**: Adjust the similarity threshold (0.10–0.90) and click **Recalculate Clusters** to re-run clustering.
- **Min Cluster Size**: Filter out small clusters with a slider.
- **Upload Dataset**: Upload your own `.csv` or `.xlsx` file. The server generates embeddings and clusters on the fly.
- **Reset View**: Reset zoom and clear search/filters.
- **Detail Panel**: Click any node to see:
  - **Category / Subcategory** (e.g., Computer Hardware / Mouse)
  - Sample records
  - **Matched iFixit Device** (when a cluster matches a known device, with link to iFixit)

### Backend

- **Embedding-based clustering**: Uses sentence-transformers with the local `minilm-en-he-fp16` model (or a HuggingFace model ID) for semantic similarity.
- **Hebrew & multilingual support**: The model and tokenization support Hebrew and other Unicode scripts. Category inference includes Hebrew keywords (e.g. מסך, מקלדת, עכבר).
- **Dynamic category inference**: Token-based matching against iFixit categories and `custom_devices.py`.
- **Subcategory inference**: Finer-grained labels (e.g., Keyboard, Monitor, Printer) derived from the device list.
- **Device matching**: IDF-scored matching of cluster records to iFixit devices and custom entries.

## Device List

The app uses two device sources:

1. **iFixit catalog** (`ifixit_devices.json`): Run `fetch_ifixit_devices.py` to populate.
2. **Custom devices** (`custom_devices.py`): Generic hardware that supplements iFixit for category/subcategory inference and device matching.

### Custom Device Categories

| Category | Subcategories / Types |
|---------|------------------------|
| **SIM Card** | Nano, Micro, Standard, eSIM |
| **Cable** | USB-C, Lightning, HDMI, DisplayPort, Ethernet, VGA, DVI, Thunderbolt, SATA, etc. |
| **Adapter** | USB hubs, HDMI/DisplayPort adapters |
| **Storage** | MicroSD, SD, Flash Drive, HDD, SSD, NAS, CD/DVD/Blu-ray drives |
| **PC Component** | RAM, CPU, GPU, Motherboard, PSU, SSD, Cooling, Thermal paste |
| **Computer Hardware** | Keyboard, Mouse, Laptop, Monitor, Webcam, Dashcam, Projector, Printer |
| **Audio** | Headphones, Earbuds, Microphone |
| **Telecom** | Router, Modem, Network Switch, Access Point, enterprise gear |

Edit `custom_devices.py` to add or adjust devices. Each entry has `name`, `category`, `subcategory`, and optional `url`.

### Hebrew / Multilingual Datasets

The app works with Hebrew and other Unicode datasets. The default model (`paraphrase-multilingual-MiniLM-L12-v2`) supports 50+ languages. Category inference includes Hebrew boost keywords (מסך, מקלדת, עכבר, כבל, etc.), and `custom_devices.py` has Hebrew device entries for matching. Upload a Hebrew CSV and the clustering and categorization will work out of the box.

## Getting Started

### Prerequisites

- Python 3.x with [Anaconda](https://www.anaconda.com/) or Miniconda
- Dependencies installed in the `dataset_clustering` conda environment (see below)

### Setup

1. Create and activate the conda environment:

```bash
conda create -n dataset_clustering python=3.11
conda activate dataset_clustering
pip install fastapi uvicorn[standard] python-multipart pandas openpyxl sentence-transformers scikit-learn
```

2. (Optional) Fetch the iFixit device catalog:

```bash
python fetch_ifixit_devices.py
```

### Running the App

1. Start the server:

```bash
conda run -n dataset_clustering python server.py
```

2. Open [http://localhost:8001](http://localhost:8001) in your browser.

**Command-line options** (run `python server.py --help` for details):

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `minilm-en-he-fp16` | Path to local model folder or HuggingFace ID |
| `--device` | `cpu` | Device for embeddings: `cpu` or `cuda` |
| `--threshold` | `0.2` | Default clustering threshold (0.1–0.9) |
| `--batch-size` | `512` | Batch size for embedding generation |
| `--data` | `data/dirty_hardware_data_40k.csv` | Path to default CSV dataset |
| `--host` | `localhost` | Host to bind |
| `--port` | `8001` | Port to run on |
| `--no-reload` | — | Disable auto-reload on file changes |

The server loads the default dataset (`data/dirty_hardware_data_40k.csv`), generates embeddings on startup, and serves the visualization. Use **Recalculate Clusters** or **Upload Dataset** to change the data or clustering.

## Model

The default embedding model is the local `minilm-en-he-fp16` folder in the project root (sentence-transformers compatible). Place your model files there, or pass a HuggingFace model ID via `--model`.

## Data

CSV files live in the `data/` folder. The default dataset is `data/dirty_hardware_data_40k.csv`. Use `--data` or `--input`/`--output` to point scripts at other paths.

## Data Processing Workflow

- `dataset_generator.py`: Generates raw `data/dirty_hardware_data_40k.csv`.
- `cluster_hardware.py`: Produces `data/clustered_output.csv` from embeddings and clustering.
- `prepare_viz_data.py`: Builds static `cluster_viz/data.json` for the legacy static workflow.
- `server.py`: Serves the live app with on-demand clustering and device matching.
