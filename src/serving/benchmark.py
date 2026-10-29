"""Step 7b: measure serving latency, so the report can quote real numbers.

Measures, on random known users:
    uncached   full path: FAISS candidates -> features -> ranker -> top N
    cached     the same request answered from Redis (only if --redis is given)
    cold-start the three fallback paths

The first few requests are discarded: they include one-time warm-up (graph tracing, caches).

Run:
    python src/serving/benchmark.py --artifacts-root outputs --data processed --features features \
        --retrieval retrieval --faiss faiss --ranking ranking_v3 --redis localhost:6379
"""
import os

# TFRS needs Keras 2. Must be set before tensorflow is imported.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from recommender import RecommendationCache, Recommender  # noqa: E402

SEED = 42


def summarize(times_ms):
    t = np.array(times_ms)
    return {"requests": int(len(t)), "p50_ms": round(float(np.percentile(t, 50)), 2),
            "p95_ms": round(float(np.percentile(t, 95)), 2), "p99_ms": round(float(np.percentile(t, 99)), 2),
            "single_thread_rps": int(1000 / t.mean())}


def timed(fn, items):
    out = []
    for x in items:
        t0 = time.perf_counter()
        fn(x)
        out.append((time.perf_counter() - t0) * 1000)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-root", default=".")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--retrieval", default="models/retrieval")
    ap.add_argument("--faiss", default="models/faiss")
    ap.add_argument("--ranking", default="models/ranking_v3")
    ap.add_argument("--redis", default="")
    ap.add_argument("--requests", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", default="outputs/benchmark")
    args = ap.parse_args()

    root = Path(args.artifacts_root)
    rec = Recommender(root / args.data, root / args.features, root / args.retrieval,
                      root / args.faiss, root / args.ranking)
    rng = np.random.default_rng(SEED)
    users = [int(u) for u in rng.choice(rec.user_ids, args.warmup + args.requests, replace=False)]
    warm, users = users[:args.warmup], users[args.warmup:]

    results = {}
    rec.cache = RecommendationCache()                       # cache off
    timed(lambda u: rec.recommend(u, 10), warm)
    results["uncached_ranker_path"] = summarize(timed(lambda u: rec.recommend(u, 10), users))

    if args.redis:
        import redis
        host, port = args.redis.split(":")
        client = redis.Redis(host=host, port=int(port), decode_responses=True)
        client.ping()
        rec.cache = RecommendationCache(client, ttl=3600, prefix="rec:bench")
        timed(lambda u: rec.recommend(u, 10), users)        # fill the cache
        rec.cache.hits = rec.cache.misses = 0               # count only the second pass
        results["cached_redis_hit"] = summarize(timed(lambda u: rec.recommend(u, 10), users))
        results["cache_hit_rate_in_second_pass"] = round(rec.cache.hits / (rec.cache.hits + rec.cache.misses), 3)
        for u in users:                                      # leave Redis clean
            client.delete(f"rec:bench:{u}:10")

    results["cold_popular"] = summarize(timed(lambda _: rec.recommend(-1, 10), range(args.requests)))
    results["cold_genre"] = summarize(timed(lambda _: rec.recommend(-1, 10, genres=["Comedy"]), range(args.requests)))
    liked = [int(i) for i in rec.item_ids[:3]]
    results["cold_taste_vector"] = summarize(timed(lambda _: rec.recommend(-1, 10, liked_item_ids=liked), range(args.requests)))

    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
