import json
from collections import Counter
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from contextlib import asynccontextmanager

from cluster_hardware import load_data, build_text_representations, generate_embeddings, cluster_embeddings
from prepare_viz_data import CATEGORY_KEYWORDS

# Global state to hold embeddings in memory so we don't recompute
app_state = {
    "df": None,
    "embeddings": None,
    "columns": None
}

def infer_category(records):
    """Infer the dominant hardware category from a cluster's records."""
    text = " ".join(
        " ".join(str(v) for v in row) for row in records
    ).lower()
    scores = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        scores[cat] = sum(text.count(kw) for kw in keywords)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Other"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting up server and generating initial embeddings...")
    # 1. Load data
    df = load_data("dirty_hardware_data.csv")
    
    # We strip out the generated cluster id if it already existed in the dataset
    if "cluster_id" in df.columns:
         df = df.drop(columns=["cluster_id"])
    
    app_state["columns"] = df.columns.tolist()
    app_state["df"] = df
    
    # 2. Build text representations
    texts = build_text_representations(df)

    # 3. Generate embeddings
    app_state["embeddings"] = generate_embeddings(texts, "all-MiniLM-L6-v2", 512)
    print("✅ Initialization complete. Ready to serve requests!")
    
    yield
    
    print("🛑 Shutting down server...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/clusters")
def get_clusters(threshold: float = 0.3):
    print(f"🔄 Reclustering with threshold {threshold}...")
    
    # 1. Run clustering with the requested threshold
    # Note: cluster_embeddings switches to two-phase for large datasets automatically
    labels = cluster_embeddings(app_state["embeddings"], threshold)
    
    # 2. Attach labels to our in-memory dataframe
    df = app_state["df"].copy()
    df["cluster_id"] = labels
    columns = app_state["columns"]
    
    # 3. Process clusters into JSON format for the visualization
    clusters = []
    category_counts = Counter()

    for cluster_id, group in df.groupby("cluster_id"):
        records = group[columns].values.tolist()
        category = infer_category(records)
        category_counts[category] += 1

        # Limit sample records to 30 for the viz
        sample = records[:30]

        clusters.append({
            "id": int(cluster_id),
            "size": len(group),
            "category": category,
            "sample_records": sample,
            "columns": columns,
        })

    # Sort by size descending
    clusters.sort(key=lambda x: x["size"], reverse=True)

    viz_data = {
        "total_rows": len(df),
        "total_clusters": len(clusters),
        "columns": columns,
        "category_counts": dict(category_counts),
        "clusters": clusters,
    }
    
    return viz_data

# Mount the visualization frontend
app.mount("/", StaticFiles(directory="cluster_viz", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)
