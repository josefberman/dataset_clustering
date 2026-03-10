# Hardware Dataset Clustering Visualization

An interactive web application built with D3.js to visualize clusters from hardware dataset groupings. The tool allows users to explore data in a dynamic radial network graph with extensive search and exploration capabilities.

## Features

- **Interactive Network Graph**: View clusters with intuitive zooming, panning, and node hovering for rapid information discovery.
- **Search & Filtering**: Real-time filtering by hardware categories (e.g., Router, iPhone) and fine-tuning by minimum cluster sizes using an adjustable slider.
- **Detailed Cluster Information**: Click on any node to slide open an interactive side panel with specific metrics and records for that cluster.

## Web Application Previews

### Overview & Exploration
A quick demonstration showing how to pan, zoom, and intuitively hover to explore the dataset clusters.

![App Overview](demo/app_overview_1773133931554.webp)

### Search & Filtering
A demonstration highlighting the use of the search feature to locate specific categories, adjust the minimal cluster size filter, and access real-time details through the side panel.

![Search & Filter Demo](demo/search_filter_demo_1773134114919.webp)

## Getting Started

### Prerequisites
- Python 3.x (to run a basic local web server)

### Running the App Locally
1. Start a local HTTP server from the root of the project directory on port 8001:

```bash
python -m http.server 8001
```

2. Open your web browser and navigate to the application:
[http://localhost:8001/cluster_viz/index.html](http://localhost:8001/cluster_viz/index.html)

## Data Processing Workflow

- `dataset_generator.py`: Generates the raw, initial `dirty_hardware_data.csv` dataset.
- `cluster_hardware.py`: Consumes the raw data, processing and saving output to `clustered_output.csv`.
- `prepare_viz_data.py`: Summarizes the clustering data into the static `cluster_viz/data.json` required by the front-end web app.
