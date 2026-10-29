"""Step 4: index the item embeddings from step 3 with FAISS and benchmark it.

Given a user vector, retrieval means "find the items whose vectors are closest".
Exact search compares the user against every item. Approximate indexes (IVF, HNSW)
skip most items and are much faster once the catalog gets big, at the price of
sometimes missing a true neighbour. This script builds all three and measures
that trade-off on real user vectors:

    flat   exact search, used as the ground truth
    ivf    items grouped into clusters, only the nearest clusters are scanned
    hnsw   graph-based search

Embeddings are L2-normalised, so inner product == cosine similarity.

Run:
    python src/retrieval/build_faiss_index.py --emb models/retrieval --out models/faiss
"""
import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np

TOP_K = 100


def build_flat(vecs):
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index


def build_ivf(vecs, nprobe):
    n, d = vecs.shape
    nlist = max(4, int(np.sqrt(n)))            # rule of thumb: ~sqrt(N) clusters
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(vecs)                          # k-means to find the clusters
    index.add(vecs)
    index.nprobe = nprobe                      # clusters scanned per query
    return index


def build_hnsw(vecs, ef_search, m=32):
    index = faiss.IndexHNSWFlat(vecs.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.add(vecs)
    index.hnsw.efSearch = ef_search            # bigger = more accurate, slower
    return index


def overlap_at_k(approx_ids, exact_ids):
    """Share of the exact top-K that the approximate index also returned."""
    hits = [len(set(a) & set(e)) / len(e) for a, e in zip(approx_ids, exact_ids)]
    return float(np.mean(hits))


def latency_ms(index, queries, k):
    """Median and p95 time for a single-user query, plus batched throughput."""
    single = []
    for q in queries[:300]:
        t0 = time.perf_counter()
        index.search(q[None, :], k)
        single.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter()
    index.search(queries, k)
    batch_s = time.perf_counter() - t0
    return {
        "p50_ms": round(float(np.percentile(single, 50)), 3),
        "p95_ms": round(float(np.percentile(single, 95)), 3),
        "queries_per_sec_batched": int(len(queries) / batch_s),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=Path, default=Path("models/retrieval"))
    ap.add_argument("--out", type=Path, default=Path("models/faiss"))
    ap.add_argument("--n-queries", type=int, default=2000)
    ap.add_argument("--nprobe", type=int, default=8)
    ap.add_argument("--ef-search", type=int, default=64)
    args = ap.parse_args()

    items = np.load(args.emb / "item_embeddings.npy").astype("float32")
    users = np.load(args.emb / "user_embeddings.npy").astype("float32")
    k = min(TOP_K, len(items))

    rng = np.random.default_rng(42)
    queries = users[rng.choice(len(users), size=min(args.n_queries, len(users)), replace=False)]
    print(f"items={len(items):,} dim={items.shape[1]} queries={len(queries):,}")

    indexes = {
        "flat": build_flat(items),
        "ivf": build_ivf(items, args.nprobe),
        "hnsw": build_hnsw(items, args.ef_search),
    }

    _, exact_ids = indexes["flat"].search(queries, k)
    results = {}
    for name, index in indexes.items():
        _, ids = index.search(queries, k)
        results[name] = {"overlap_with_exact": round(overlap_at_k(ids, exact_ids), 4)}
        results[name].update(latency_ms(index, queries, k))

    args.out.mkdir(parents=True, exist_ok=True)
    for name, index in indexes.items():
        faiss.write_index(index, str(args.out / f"items_{name}.index"))
    (args.out / "faiss_benchmark.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
