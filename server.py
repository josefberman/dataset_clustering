import json
from collections import Counter
import os
import math

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from contextlib import asynccontextmanager
import io

from cluster_hardware import load_data, build_text_representations, generate_embeddings, cluster_embeddings
from custom_devices import CUSTOM_DEVICES

# Global state to hold embeddings in memory so we don't recompute
app_state = {
    "df": None,
    "embeddings": None,
    "columns": None,
    "device_list": [],      # Flat list of iFixit devices
    "device_idx": {},       # Inverted index: word -> set of device indices
    "device_idf": {},       # IDF weights: word -> float weight
    "device_by_name": {},   # Fast name -> device info lookup
}

def infer_category(records):
    """Infer the dominant hardware category from a cluster's records."""
    text = " ".join(
        " ".join(str(v) for v in row) for row in records
    ).lower()
    scores = {}
    keywords_dict = app_state.get("category_keywords", {})
    for cat, keywords in keywords_dict.items():
        scores[cat] = sum(text.count(kw) for kw in keywords)
    
    if not scores:
        return "Other"
        
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Other"


def build_device_index(device_list):
    """Build an inverted word index over device names for fast lookup with IDF scoring."""
    idx = {}
    by_name = {}
    stop = {"the", "and", "for", "with", "pro", "gen", "new", "plus", "max",
            "mini", "lite", "air", "one", "black", "white", "silver", "inch"}
    
    total_docs = len(device_list)
    for i, device in enumerate(device_list):
        tokens = set(
            w.lower() for w in device["name"].replace("-", " ").replace("/", " ").split()
            if len(w) >= 3 and w.lower() not in stop
        )
        for tok in tokens:
            idx.setdefault(tok, set()).add(i)
        by_name[device["name"]] = device
        
    # Calculate Inverse Document Frequency (IDF) for each word
    # Rare words (like model numbers) get high scores, common words get low scores
    idf = {}
    for word, doc_indices in idx.items():
        doc_freq = len(doc_indices)
        idf[word] = math.log(total_docs / (1 + doc_freq))
        
    return idx, idf, by_name


def match_device(records):
    """Quickly match a cluster's sample records to the closest iFixit device using IDF scoring."""
    if not app_state["device_list"]:
        return None, None, None

    # Build a combined search string from the clustemotor's sample records (first 10)
    search_text = " ".join(
        " ".join(str(v) for v in row) for row in records[:10]
    )
    # Tokenize: words of 3+ chars (preserving case for device names)
    stop = {"the", "and", "for", "with", "pro", "gen", "new", "plus", "max",
            "mini", "lite", "air", "one", "black", "white", "silver", "inch"}
    query_tokens = [
        w.lower() for w in search_text.replace("-", " ").replace("/", " ").split()
        if len(w) >= 3 and w.lower() not in stop
    ]
    if not query_tokens:
        return None, None, None

    # Score each candidate device by summing the IDF weight of matched tokens
    idx = app_state["device_idx"]
    idf = app_state["device_idf"]
    candidate_scores = {}
    
    # We want to match unique tokens in the query, not repeated ones
    unique_query_tokens = set(query_tokens)
    
    for tok in unique_query_tokens:
        weight = idf.get(tok, 0)
        for dev_i in idx.get(tok, set()):
            candidate_scores[dev_i] = candidate_scores.get(dev_i, 0.0) + weight

    if not candidate_scores:
        return None, None, None

    # Pick the device with the highest IDF overlap score
    best_i = max(candidate_scores, key=candidate_scores.get)
    best_score = candidate_scores[best_i]
    
    # Require a minimum score threshold. A very common word might have an IDF of 2.0.
    # A rare model number might easily have an IDF of 8.0+.
    if best_score < 4.0:
        return None, None, None

    device = app_state["device_list"][best_i]
    return device["name"], device.get("category"), device.get("url")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting up server and generating initial embeddings...")
    
    # 0. Load iFixit device catalog and build inverted index
    if os.path.exists("ifixit_devices.json"):
        with open("ifixit_devices.json", "r", encoding="utf-8") as f:
            app_state["device_list"] = json.load(f)
        
        # Inject custom manual devices
        app_state["device_list"].extend(CUSTOM_DEVICES)
        idx, idf, by_name = build_device_index(app_state["device_list"])
        app_state["device_idx"] = idx
        app_state["device_idf"] = idf
        app_state["device_by_name"] = by_name
        
        # Build dynamic CATEGORY_KEYWORDS from the loaded device array
        dynamic_keywords = {}
        for d in app_state["device_list"]:
            cat = d.get("category")
            if not cat: continue
            if cat not in dynamic_keywords:
                # Add the category name as the primary keyword
                dynamic_keywords[cat] = {cat.lower()}
            # Add subcategory names as keywords for the parent category
            sub = d.get("subcategory")
            if sub:
                dynamic_keywords[cat].add(sub.lower())
                
        # Convert sets to lists
        app_state["category_keywords"] = {k: list(v) for k, v in dynamic_keywords.items()}
        
        print(f"📚 Loaded {len(app_state['device_list']):,} devices, indexed {len(idx):,} word tokens.")
        print(f"🏷️  Built {len(app_state['category_keywords'])} dynamic categories.")
    else:
        print("⚠️  ifixit_devices.json not found. Run fetch_ifixit_devices.py first.")
    
    # 1. Load data
    df = load_data("dirty_hardware_data_40k.csv")
    
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
        
        # Match cluster to a real-world iFixit device
        device_name, device_category, device_url = match_device(records)

        clusters.append({
            "id": int(cluster_id),
            "size": len(group),
            "category": category,
            "sample_records": sample,
            "columns": columns,
            "matched_device": device_name,
            "matched_device_category": device_category,
            "device_url": device_url,
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

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...), threshold: float = Form(0.5)):
    print(f"📥 Received file upload: {file.filename}")
    
    try:
        content = await file.read()
        
        # Determine file type and load into pandas
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content), dtype=str).fillna("")
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(content), dtype=str).fillna("")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a .csv or .xlsx file.")
            
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="The uploaded file contains no data rows.")
            
        print(f"   Loaded {len(df):,} rows from uploaded file.")
        
        # We strip out the generated cluster id if it already existed in the dataset
        if "cluster_id" in df.columns:
             df = df.drop(columns=["cluster_id"])
             
        app_state["columns"] = df.columns.tolist()
        app_state["df"] = df
        
        # Generate new embeddings
        texts = build_text_representations(df)
        app_state["embeddings"] = generate_embeddings(texts, "all-MiniLM-L6-v2", 512)
        
        # Trigger reclustering using the global method we already have
        return get_clusters(threshold)
        
    except Exception as e:
        print(f"❌ Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Mount the visualization frontend
app.mount("/", StaticFiles(directory="cluster_viz", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)
