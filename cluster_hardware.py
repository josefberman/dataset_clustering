#!/usr/bin/env python3
"""
Cluster dirty hardware records using text embeddings + agglomerative hierarchical clustering.

For large datasets (>10K rows), uses a two-phase approach:
  Phase 1: Pre-group rows with MiniBatchKMeans into manageable chunks
  Phase 2: Run agglomerative clustering within each chunk, then merge labels

Usage:
    python cluster_hardware.py --input dirty_hardware_data_40k.csv --threshold 0.3
    python cluster_hardware.py --input dirty_hardware_data_40k.csv --threshold 0.2 --sample-size 5000
"""

import argparse
import time
import sys

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans


# Maximum rows for direct agglomerative clustering (above this, use two-phase approach)
DIRECT_CLUSTERING_LIMIT = 15_000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster dirty hardware CSV records using embeddings + agglomerative clustering."
    )
    parser.add_argument(
        "--input", default="dirty_hardware_data_40k.csv",
        help="Path to the input CSV file (default: dirty_hardware_data_40k.csv)"
    )
    parser.add_argument(
        "--output", default="clustered_output.csv",
        help="Path to the output CSV file (default: clustered_output.csv)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3,
        help="Distance threshold for clustering. Lower = tighter/more clusters, higher = looser/fewer clusters. "
             "Range 0.0–2.0 for cosine distance (default: 0.3)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=512,
        help="Batch size for embedding generation (default: 512)"
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Use only the first N rows for quick testing (default: use all rows)"
    )
    parser.add_argument(
        "--model", default="all-MiniLM-L6-v2",
        help="Sentence-transformer model name (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--pre-clusters", type=int, default=None,
        help="Number of KMeans pre-clusters for large datasets. Auto-calculated if not set."
    )
    return parser.parse_args()


def load_data(path, sample_size=None):
    """Load CSV and optionally sample rows."""
    print(f"📂 Loading data from {path}...")
    df = pd.read_csv(path, dtype=str).fillna("")
    if sample_size is not None:
        df = df.head(sample_size)
    print(f"   Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


def build_text_representations(df):
    """Concatenate all columns into a single text string per row."""
    print("📝 Building text representations...")
    texts = df.apply(lambda row: " | ".join(row.values), axis=1).tolist()
    print(f"   Example: {texts[0]!r}")
    return texts


def generate_embeddings(texts, model_name, batch_size):
    """Generate embeddings using a sentence-transformer model."""
    print(f"🤖 Loading model '{model_name}'...")
    model = SentenceTransformer(model_name)

    print(f"⚡ Generating embeddings for {len(texts):,} texts (batch_size={batch_size})...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # Pre-normalize for cosine similarity
    )
    elapsed = time.time() - t0
    print(f"   Done in {elapsed:.1f}s — shape: {embeddings.shape}")
    return embeddings


def cluster_direct(embeddings, threshold):
    """Run agglomerative clustering directly (for smaller datasets)."""
    print(f"🔗 Clustering {len(embeddings):,} rows with distance_threshold={threshold}...")
    t0 = time.time()
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)
    elapsed = time.time() - t0
    n_clusters = len(set(labels))
    print(f"   Found {n_clusters:,} clusters in {elapsed:.1f}s")
    return labels


def cluster_twophase(embeddings, threshold, n_pre_clusters=None):
    """
    Two-phase clustering for large datasets:
      Phase 1: MiniBatchKMeans to create manageable pre-groups
      Phase 2: Agglomerative clustering within each pre-group
    """
    n = len(embeddings)
    if n_pre_clusters is None:
        # Aim for pre-groups of ~5000 rows each
        n_pre_clusters = max(10, n // 5000)
    n_pre_clusters = min(n_pre_clusters, n)

    print(f"🔗 Phase 1: Pre-grouping {n:,} rows into {n_pre_clusters} groups with KMeans...")
    t0 = time.time()
    kmeans = MiniBatchKMeans(
        n_clusters=n_pre_clusters,
        batch_size=min(4096, n),
        random_state=42,
        n_init=3,
    )
    pre_labels = kmeans.fit_predict(embeddings)
    elapsed = time.time() - t0
    print(f"   Pre-grouping done in {elapsed:.1f}s")

    print(f"🔗 Phase 2: Agglomerative clustering within each group (threshold={threshold})...")
    t0 = time.time()
    final_labels = np.zeros(n, dtype=int)
    global_cluster_id = 0

    for group_id in range(n_pre_clusters):
        mask = pre_labels == group_id
        group_embeddings = embeddings[mask]
        group_size = len(group_embeddings)

        if group_size <= 1:
            final_labels[mask] = global_cluster_id
            global_cluster_id += 1
            continue

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric="cosine",
            linkage="average",
        )
        group_labels = clustering.fit_predict(group_embeddings)
        n_group_clusters = len(set(group_labels))

        # Map local cluster IDs to global IDs
        final_labels[mask] = group_labels + global_cluster_id
        global_cluster_id += n_group_clusters

    elapsed = time.time() - t0
    n_clusters = len(set(final_labels))
    print(f"   Found {n_clusters:,} clusters in {elapsed:.1f}s")
    return final_labels


def cluster_embeddings(embeddings, threshold, n_pre_clusters=None):
    """Choose clustering strategy based on dataset size."""
    n = len(embeddings)
    if n <= DIRECT_CLUSTERING_LIMIT:
        return cluster_direct(embeddings, threshold)
    else:
        print(f"ℹ️  Dataset has {n:,} rows (>{DIRECT_CLUSTERING_LIMIT:,}), using two-phase clustering")
        return cluster_twophase(embeddings, threshold, n_pre_clusters)


def print_summary(df, max_clusters=30, samples_per_cluster=3):
    """Print a summary of the clustering results."""
    cluster_sizes = df.groupby("cluster_id").size().sort_values(ascending=False)
    n_clusters = len(cluster_sizes)

    print("\n" + "=" * 80)
    print(f"📊 CLUSTERING SUMMARY")
    print(f"   Total rows:     {len(df):,}")
    print(f"   Total clusters: {n_clusters:,}")
    print(f"   Largest:        {cluster_sizes.iloc[0]:,} rows")
    print(f"   Smallest:       {cluster_sizes.iloc[-1]:,} rows")
    print(f"   Median size:    {int(cluster_sizes.median()):,} rows")
    print("=" * 80)

    # Show the top clusters with sample rows
    show_n = min(max_clusters, n_clusters)
    print(f"\n🔍 Top {show_n} clusters by size:\n")

    for i, (cluster_id, size) in enumerate(cluster_sizes.head(show_n).items()):
        cluster_rows = df[df["cluster_id"] == cluster_id]
        # Show a few sample rows (excluding the cluster_id column for readability)
        display_cols = [c for c in df.columns if c != "cluster_id"]
        samples = cluster_rows[display_cols].head(samples_per_cluster)

        print(f"  Cluster {cluster_id} ({size:,} rows):")
        for _, row in samples.iterrows():
            print(f"    → {' | '.join(row.values)}")
        print()


def main():
    args = parse_args()

    print(f"\n{'=' * 80}")
    print(f"  Hardware Record Clustering")
    print(f"  Threshold: {args.threshold}  |  Model: {args.model}")
    print(f"{'=' * 80}\n")

    # 1. Load data
    df = load_data(args.input, args.sample_size)

    # 2. Build text representations
    texts = build_text_representations(df)

    # 3. Generate embeddings
    embeddings = generate_embeddings(texts, args.model, args.batch_size)

    # 4. Cluster
    labels = cluster_embeddings(embeddings, args.threshold, args.pre_clusters)

    # 5. Attach labels and sort by cluster
    df["cluster_id"] = labels
    df = df.sort_values("cluster_id").reset_index(drop=True)

    # 6. Save output
    df.to_csv(args.output, index=False)
    print(f"\n💾 Saved clustered output to {args.output}")

    # 7. Print summary
    print_summary(df)


if __name__ == "__main__":
    main()
