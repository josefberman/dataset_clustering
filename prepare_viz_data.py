#!/usr/bin/env python3
"""
Prepare clustered data for the interactive visualization.
Reads clustered_output.csv and generates a JSON file for the D3.js web app.
"""

import json
import sys
import pandas as pd
from collections import Counter

import os
from custom_devices import CUSTOM_DEVICES

def load_category_keywords():
    """Build dynamic CATEGORY_KEYWORDS from iFixit devices + custom ones."""
    device_list = []
    if os.path.exists("ifixit_devices.json"):
        with open("ifixit_devices.json", "r", encoding="utf-8") as f:
            device_list = json.load(f)
    device_list.extend(CUSTOM_DEVICES)
    
    dynamic_keywords = {}
    for d in device_list:
        cat = d.get("category")
        if not cat: continue
        if cat not in dynamic_keywords:
            dynamic_keywords[cat] = {cat.lower()}
        sub = d.get("subcategory")
        if sub:
            dynamic_keywords[cat].add(sub.lower())
    
    return {k: list(v) for k, v in dynamic_keywords.items()}

CATEGORY_KEYWORDS = load_category_keywords()


def main():
    input_file = sys.argv[1] if len(sys.argv) > 1 else "clustered_output.csv"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "cluster_viz/data.json"

    print(f"📂 Loading {input_file}...")
    df = pd.read_csv(input_file, dtype=str).fillna("")

    columns = [c for c in df.columns if c != "cluster_id"]
    df["cluster_id"] = df["cluster_id"].astype(int)

    df["cluster_id"] = df["cluster_id"].astype(int)

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

    print("🔧 Processing clusters...")
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

    print(f"💾 Writing {output_file}...")
    with open(output_file, "w") as f:
        json.dump(viz_data, f)

    print(f"✅ Done! {len(clusters)} clusters exported.")
    print(f"   Categories: {dict(category_counts)}")


if __name__ == "__main__":
    main()
