import argparse
import json
from collections import Counter
import os
import math

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from contextlib import asynccontextmanager
import io

from cluster_hardware import load_data, build_text_representations, generate_embeddings, cluster_embeddings
from custom_devices import CUSTOM_DEVICES

# Server config (populated from CLI args when run as __main__)
SERVER_CONFIG = {
    "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "threshold": 0.5,
    "batch_size": 512,
    "data_path": "dirty_hardware_data_40k.csv",
    "host": "localhost",
    "port": 8001,
    "reload": True,
}


def parse_args():
    """Parse command-line arguments for the server."""
    parser = argparse.ArgumentParser(
        description="Cluster visualization server: load hardware data, generate embeddings, serve the interactive GUI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python server.py
  python server.py --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --threshold 0.5
  python server.py --data my_data.csv --port 9000
  python server.py --no-reload
        """,
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence-transformer model for embeddings (default: paraphrase-multilingual-MiniLM-L12-v2)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Default clustering threshold. Lower = tighter/more clusters, higher = looser/fewer clusters. "
             "Range 0.1–0.9 (default: 0.5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for embedding generation (default: 512)",
    )
    parser.add_argument(
        "--data",
        default="dirty_hardware_data_40k.csv",
        help="Path to the default CSV dataset (default: dirty_hardware_data_40k.csv)",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to bind the server (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to run the server on (default: 8001)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=True,
        help="Enable auto-reload on file changes (default: True)",
    )
    parser.add_argument(
        "--no-reload",
        action="store_false",
        dest="reload",
        help="Disable auto-reload",
    )
    return parser.parse_args()


# Global state to hold embeddings in memory so we don't recompute
app_state = {
    "df": None,
    "embeddings": None,
    "columns": None,
    "device_list": [],      # Flat list of iFixit devices
    "device_idx": {},       # Inverted index: word -> set of device indices
    "device_idf": {},       # IDF weights: word -> float weight
    "device_by_name": {},   # Fast name -> device info lookup
    "subcategory_keywords": {},  # category -> {subcategory -> [keywords]}
}

def infer_category(records):
    """Infer the dominant hardware category from a cluster's records."""
    import re

    text = " ".join(" ".join(str(v) for v in row) for row in records).lower()
    # Tokenize on non-alphanumerics to avoid substring bugs (e.g. "monitor" matching "motor")
    tokens = re.findall(r"[a-z0-9]+", text)
    if not tokens:
        return "Other"

    token_counts = Counter(tokens)
    scores = {}
    keywords_dict = app_state.get("category_keywords", {})

    # Hand-tuned keyword boosts for messy inventory text.
    # These categories are common and should win when their tokens appear,
    # even if iFixit category/subcategory keywords are sparse.
    boost_keywords = {
        "Cable": {
            "cable", "usb", "lightning", "hdmi", "displayport", "ethernet", "toslink",
            "sata", "coax", "vga", "dvi", "thunderbolt", "cat5", "cat5e", "cat6",
        },
        "Adapter": {
            "adapter", "dongle", "hub", "converter", "usbc", "usba", "lightning", "hdmi",
            "displayport", "ethernet",
        },
        "Storage": {"ssd", "hdd", "nvme", "microsd", "sd", "flash", "usb", "drive", "nas"},
        "SIM Card": {"sim", "esim", "nano", "micro"},
        "Audio": {"headphone", "headphones", "earbud", "earbuds", "over", "ear", "inear", "overear", "mic", "microphone"},
        "Computer Hardware": {"mouse", "mice", "logitech", "steelseries", "rival", "ergonomic", "wired", "wireless",
                             "monitor", "display", "lcd", "led", "ultrawide"},
    }

    for cat, keywords in keywords_dict.items():
        score = 0
        for kw in keywords:
            if not kw:
                continue
            for kw_tok in re.findall(r"[a-z0-9]+", str(kw).lower()):
                score += token_counts.get(kw_tok, 0)
        # Apply boost if this category has strong evidence in tokens.
        if cat in boost_keywords:
            score += 5 * sum(token_counts.get(t, 0) for t in boost_keywords[cat])
        scores[cat] = score

    if not scores:
        return "Other"

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Other"


def infer_subcategory(records, category: str):
    """Infer subcategory for a given category based on tokens."""
    import re

    if not category or category == "Other":
        return ""

    sub_kw = app_state.get("subcategory_keywords", {}).get(category, {})
    if not sub_kw:
        return ""

    text = " ".join(" ".join(str(v) for v in row) for row in records).lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    if not tokens:
        return ""

    token_counts = Counter(tokens)
    best_sub = ""
    best_score = 0
    for sub, kws in sub_kw.items():
        score = 0
        for kw in kws:
            for kw_tok in re.findall(r"[a-z0-9]+", str(kw).lower()):
                score += token_counts.get(kw_tok, 0)
        if score > best_score:
            best_sub = sub
            best_score = score

    return best_sub if best_score > 0 else ""


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

    # Build a combined search string from the cluster's sample records (first 10)
    search_text = " ".join(
        " ".join(str(v) for v in row) for row in records[:10]
    )
    # Tokenize: words of 3+ chars (preserving case for device names)
    stop = {"the", "and", "for", "with", "pro", "gen", "new", "plus", "max",
            "mini", "lite", "air", "one", "black", "white", "silver", "inch"}
    def is_informative_token(tok: str) -> bool:
        # Avoid numeric-only inventory IDs like "427" dominating matches.
        # Keep alphanumeric model tokens like "m3max" or "xps15".
        if tok.isdigit():
            return False
        if any(ch.isalpha() for ch in tok):
            return True
        return False

    query_tokens = []
    for raw in search_text.replace("-", " ").replace("/", " ").split():
        tok = raw.lower().strip()
        if len(tok) < 3 or tok in stop:
            continue
        if not is_informative_token(tok):
            continue
        query_tokens.append(tok)
    if not query_tokens:
        return None, None, None

    # Score each candidate device by summing the IDF weight of matched tokens
    idx = app_state["device_idx"]
    idf = app_state["device_idf"]
    candidate_scores = {}
    candidate_token_sets = {}
    
    # We want to match unique tokens in the query, not repeated ones
    unique_query_tokens = set(query_tokens)
    
    for tok in unique_query_tokens:
        weight = idf.get(tok, 0.0)
        if weight <= 0:
            continue
        for dev_i in idx.get(tok, set()):
            candidate_scores[dev_i] = candidate_scores.get(dev_i, 0.0) + weight
            if dev_i not in candidate_token_sets:
                candidate_token_sets[dev_i] = set()
            candidate_token_sets[dev_i].add(tok)

    if not candidate_scores:
        return None, None, None

    # Pick the device with the highest IDF overlap score
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    best_i, best_score = sorted_candidates[0]
    second_score = sorted_candidates[1][1] if len(sorted_candidates) > 1 else 0.0

    # How many distinct informative tokens contributed to this best match
    token_count = len(candidate_token_sets.get(best_i, set()))

    # Require:
    # - a reasonably strong absolute score
    # - at least 2 distinct matching tokens (to avoid "coffee maker" style random matches)
    # - the best match to be clearly better than the next-best one
    if best_score < 6.0 or token_count < 2 or (second_score > 0 and best_score / second_score < 1.2):
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
        
        # Build dynamic CATEGORY_KEYWORDS and SUBCATEGORY_KEYWORDS from the loaded device array
        dynamic_keywords = {}
        dynamic_sub_keywords = {}
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
                dynamic_sub_keywords.setdefault(cat, {}).setdefault(sub, set()).add(sub.lower())
                
        # Convert sets to lists
        app_state["category_keywords"] = {k: list(v) for k, v in dynamic_keywords.items()}
        app_state["subcategory_keywords"] = {
            cat: {sub: list(v) for sub, v in subs.items()}
            for cat, subs in dynamic_sub_keywords.items()
        }
        
        print(f"📚 Loaded {len(app_state['device_list']):,} devices, indexed {len(idx):,} word tokens.")
        print(f"🏷️  Built {len(app_state['category_keywords'])} dynamic categories.")
    else:
        print("⚠️  ifixit_devices.json not found. Run fetch_ifixit_devices.py first.")
    
    # 1. Load data
    data_path = SERVER_CONFIG["data_path"]
    df = load_data(data_path)
    
    # We strip out the generated cluster id if it already existed in the dataset
    if "cluster_id" in df.columns:
         df = df.drop(columns=["cluster_id"])
    
    app_state["columns"] = df.columns.tolist()
    app_state["df"] = df
    
    # 2. Build text representations
    texts = build_text_representations(df)

    # 3. Generate embeddings
    model_name = SERVER_CONFIG["model"]
    batch_size = SERVER_CONFIG["batch_size"]
    app_state["embeddings"] = generate_embeddings(texts, model_name, batch_size)
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
def get_clusters(threshold: float = None):
    if threshold is None:
        threshold = SERVER_CONFIG["threshold"]
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
        subcategory = infer_subcategory(records, category)
        category_counts[category] += 1

        # Limit sample records to 30 for the viz
        sample = records[:30]
        
        # Match cluster to a real-world iFixit device
        device_name, device_category, device_url = match_device(records)

        clusters.append({
            "id": int(cluster_id),
            "size": len(group),
            "category": category,
            "subcategory": subcategory,
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


@app.get("/api/export")
def export_clusters_excel(threshold: float = None):
    """Export all clusters to Excel: original columns + cluster_id, cluster_category, cluster_subcategory, matched_device."""
    if threshold is None:
        threshold = SERVER_CONFIG["threshold"]
    print(f"📤 Exporting clusters (threshold={threshold}) to Excel...")

    try:
        if app_state["embeddings"] is None or app_state["df"] is None:
            raise HTTPException(status_code=503, detail="Data not loaded yet. Wait for the server to finish initializing.")
    except HTTPException:
        raise

    try:
        labels = cluster_embeddings(app_state["embeddings"], threshold)
        df = app_state["df"].copy()
        df["cluster_id"] = labels
        columns = app_state["columns"]

        # Build cluster metadata for each cluster
        cluster_meta = {}
        for cluster_id, group in df.groupby("cluster_id"):
            records = group[columns].values.tolist()
            category = infer_category(records)
            subcategory = infer_subcategory(records, category)
            device_name, _, _ = match_device(records)
            cluster_meta[int(cluster_id)] = {
                "cluster_category": category,
                "cluster_subcategory": subcategory,
                "matched_device": device_name or "",
            }

        # Add metadata columns to each row
        df["cluster_category"] = df["cluster_id"].map(lambda cid: cluster_meta.get(int(cid), {}).get("cluster_category", ""))
        df["cluster_subcategory"] = df["cluster_id"].map(lambda cid: cluster_meta.get(int(cid), {}).get("cluster_subcategory", ""))
        df["matched_device"] = df["cluster_id"].map(lambda cid: cluster_meta.get(int(cid), {}).get("matched_device", ""))

        # Reorder: original columns first, then cluster_id, category, subcategory, matched_device
        export_columns = columns + ["cluster_id", "cluster_category", "cluster_subcategory", "matched_device"]
        export_df = df[export_columns]

        buffer = io.BytesIO()
        export_df.to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=clusters_export.xlsx"},
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        model_name = SERVER_CONFIG["model"]
        batch_size = SERVER_CONFIG["batch_size"]
        app_state["embeddings"] = generate_embeddings(texts, model_name, batch_size)
        
        # Trigger reclustering using the global method we already have
        return get_clusters(threshold)
        
    except Exception as e:
        print(f"❌ Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Mount the visualization frontend
app.mount("/", StaticFiles(directory="cluster_viz", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    args = parse_args()
    SERVER_CONFIG["model"] = args.model
    SERVER_CONFIG["threshold"] = args.threshold
    SERVER_CONFIG["batch_size"] = args.batch_size
    SERVER_CONFIG["data_path"] = args.data
    SERVER_CONFIG["host"] = args.host
    SERVER_CONFIG["port"] = args.port
    SERVER_CONFIG["reload"] = args.reload
    uvicorn.run(
        "server:app",
        host=SERVER_CONFIG["host"],
        port=SERVER_CONFIG["port"],
        reload=SERVER_CONFIG["reload"],
    )
