"""Step 6: the recommender that ties everything together, with cold-start fallbacks and a Redis cache.

    known user ........ FAISS candidates -> DCN ranker -> top N          (source: "ranker")
    new user, liked movies given ... average of those movies' vectors -> FAISS   ("taste_vector")
    new user, genres given ......... most-liked movies inside those genres       ("popular_by_genre")
    new user, nothing given ........ most-liked movies overall                   ("popular")

Cold-start users have no history, so the ranker (which relies on user stats) is skipped for them.
The "taste_vector" path is a heuristic: user and movie vectors live in the same space, so the
mean of a few liked movies is a reasonable stand-in for a user vector. It is not evaluated offline.

Results for known users are cached in Redis for `ttl` seconds. The cache is optional and can never
break a request: if Redis is down, the recommender just computes the answer.

Try it:
    python src/serving/recommender.py --demo --artifacts-root . --ranking models/ranking_v3
"""
import os

# TFRS needs Keras 2. Must be set before tensorflow is imported.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import argparse
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.append(str(Path(__file__).resolve().parents[1] / "ranking"))
from train_candidate_ranker import CandidateRanker, FEATURES, feature_matrix, genre_profiles, retrieve  # noqa: E402
from train_ranker import genre_table  # noqa: E402


class RecommendationCache:
    """Thin Redis wrapper. Any Redis error is treated as a cache miss."""

    def __init__(self, client=None, ttl=3600, prefix="rec:v1"):
        self.client, self.ttl, self.prefix = client, ttl, prefix
        self.hits = self.misses = 0

    def _key(self, user_id, k):
        return f"{self.prefix}:{user_id}:{k}"

    def get(self, user_id, k):
        if self.client is None:
            return None
        try:
            raw = self.client.get(self._key(user_id, k))
        except Exception:
            return None
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(raw)

    def set(self, user_id, k, items):
        if self.client is None:
            return
        try:
            self.client.set(self._key(user_id, k), json.dumps(items), ex=self.ttl)
        except Exception:
            pass


class TritonScorer:
    """Scores candidates on a Triton server. The ONNX model takes RAW features (it standardises inside).

    Any failure raises, and the recommender then falls back to its local ranker.
    """

    def __init__(self, url, model_name="candidate_ranker", timeout=2.0):
        import tritonclient.http as httpclient
        self._http = httpclient
        self.client = httpclient.InferenceServerClient(url=url, connection_timeout=timeout, network_timeout=timeout)
        self.model_name = model_name

    def score(self, raw_features, item_rows):
        inputs = [self._http.InferInput("num", list(raw_features.shape), "FP32"),
                  self._http.InferInput("item", list(item_rows.shape), "INT32")]
        inputs[0].set_data_from_numpy(raw_features.astype("float32"))
        inputs[1].set_data_from_numpy(item_rows.astype("int32"))
        result = self.client.infer(self.model_name, inputs, outputs=[self._http.InferRequestedOutput("score")])
        return result.as_numpy("score")


class Recommender:
    def __init__(self, data_dir, features_dir, retrieval_dir, faiss_dir, ranking_dir,
                 index_name="hnsw", n_candidates=200, cache=None, triton_url=None):
        data_dir, features_dir = Path(data_dir), Path(features_dir)
        retrieval_dir, faiss_dir, ranking_dir = Path(retrieval_dir), Path(faiss_dir), Path(ranking_dir)

        self.user_ids = pd.read_parquet(retrieval_dir / "users.parquet")["user_id"].to_numpy()
        self.item_ids = pd.read_parquet(retrieval_dir / "items.parquet")["item_id"].to_numpy()
        self.user_vocab, self.item_vocab = pd.Index(self.user_ids), pd.Index(self.item_ids)
        self.user_vecs = np.load(retrieval_dir / "user_embeddings.npy").astype("float32")
        self.item_vecs = np.load(retrieval_dir / "item_embeddings.npy").astype("float32")
        self.index = faiss.read_index(str(faiss_dir / f"items_{index_name}.index"))
        self.n_candidates = n_candidates
        self.cache = cache or RecommendationCache()

        meta = pd.read_parquet(data_dir / "items.parquet").set_index("item_id").loc[self.item_ids]
        self.titles = meta["title"].to_numpy()
        self.genre_names = meta["genres"].fillna("").str.split("|").tolist()
        self.genres = genre_table(pd.read_parquet(data_dir / "items.parquet"), self.item_ids)[1:]

        self.user_feats = pd.read_parquet(features_dir / "user_features.parquet")
        self.item_feats = pd.read_parquet(features_dir / "item_features.parquet")

        train = pd.read_parquet(data_dir / "train.parquet", columns=["user_id", "item_id", "label"])
        u, i = self.user_vocab.get_indexer(train["user_id"]), self.item_vocab.get_indexer(train["item_id"])
        ok = (u >= 0) & (i >= 0)
        self.seen = sp.csr_matrix((np.ones(ok.sum(), dtype=np.int8), (u[ok], i[ok])),
                                  shape=(len(self.user_ids), len(self.item_ids)))
        self.profiles = genre_profiles(train, self.user_vocab, self.item_vocab, self.genres)

        # most-liked movies first (count of positive ratings in train)
        likes = self.item_feats.set_index("item_id").reindex(self.item_ids)["item_pos"].fillna(0).to_numpy()
        self.popular_order = np.argsort(-likes)

        # one boolean mask per genre, for the genre fallback
        self.genre_masks = {}
        for g in sorted({g for gs in self.genre_names for g in gs if g}):
            self.genre_masks[g.lower()] = np.array([g in gs for gs in self.genre_names])

        cfg = json.loads((ranking_dir / "candidate_ranker_config.json").read_text())
        assert cfg["features"] == FEATURES, "ranker was trained with a different feature list"
        self.mean = np.array(cfg["mean"], dtype="float32")
        self.std = np.array(cfg["std"], dtype="float32")
        self.ranker = CandidateRanker(self.genres)
        self.ranker({"num": np.zeros((1, len(FEATURES)), "float32"), "item": np.zeros(1, "int32")})  # build
        self.ranker.load_weights(str(ranking_dir / "candidate_ranker.weights.h5"))

        self.triton, self.triton_failures = None, 0
        if triton_url:
            try:
                self.triton = TritonScorer(triton_url)
            except Exception as exc:            # e.g. tritonclient not installed
                print(f"Triton client unavailable ({exc}); scoring locally")

    # ------------------------------------------------------------ public API

    def recommend(self, user_id, k=10, genres=None, liked_item_ids=None):
        t0 = time.perf_counter()
        cached = False
        if user_id in self.user_vocab:
            items = self.cache.get(user_id, k)
            cached = items is not None
            if items is None:
                items = self._format(self._rank_known_user(self.user_vocab.get_loc(user_id), k))
                self.cache.set(user_id, k, items)
            source = "ranker"
        elif liked_item_ids:
            idx, source = self._taste_vector(liked_item_ids, k)
            items = self._format(idx)
        elif genres:
            idx, source = self._popular_by_genre(genres, k)
            items = self._format(idx)
        else:
            source, items = "popular", self._format(self.popular_order[:k])
        return {"source": source, "cached": cached, "items": items,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}

    # ------------------------------------------------------------ recommendation paths

    def _rank_known_user(self, u, k):
        cand_items, cand_scores = retrieve(self.index, self.user_vecs[u:u + 1], [u], self.seen, self.n_candidates)
        X, item_rows, _ = feature_matrix(np.array([u]), cand_items, cand_scores, self.user_ids, self.item_ids,
                                         self.user_feats, self.item_feats, self.profiles, self.genres)
        prob = None
        if self.triton is not None:
            try:
                prob = self.triton.score(X, item_rows)          # Triton standardises inside the model
            except Exception:
                self.triton_failures += 1                       # server down or slow: use the local copy
        if prob is None:
            scaled = ((X - self.mean) / self.std).astype("float32")
            prob = self.ranker({"num": scaled, "item": item_rows}, training=False).numpy()
        return cand_items[0][np.argsort(-prob)[:k]]

    def _taste_vector(self, liked_item_ids, k):
        liked = self.item_vocab.get_indexer(liked_item_ids)
        liked = liked[liked >= 0]
        if len(liked) == 0:
            return self.popular_order[:k], "popular"
        vec = self.item_vecs[liked].mean(axis=0, keepdims=True)
        vec /= np.linalg.norm(vec) + 1e-9
        _, ids = self.index.search(vec.astype("float32"), k + len(liked) + 10)
        already = set(liked.tolist())
        ids = [i for i in ids[0] if i >= 0 and i not in already]
        return np.array(ids[:k]), "taste_vector"

    def _popular_by_genre(self, genres, k):
        masks = [self.genre_masks[g.lower()] for g in genres if g.lower() in self.genre_masks]
        if not masks:
            return self.popular_order[:k], "popular"
        wanted = np.logical_or.reduce(masks)
        in_genre = self.popular_order[wanted[self.popular_order]]
        return in_genre[:k], "popular_by_genre"

    def _format(self, idx):
        return [{"item_id": int(self.item_ids[i]), "title": str(self.titles[i])} for i in idx]


# ---------------------------------------------------------------- command line / demo

def build_from_args(args):
    root = Path(args.artifacts_root)
    cache = RecommendationCache()
    if args.redis:
        import redis
        host, port = args.redis.split(":")
        client = redis.Redis(host=host, port=int(port), decode_responses=True)
        try:
            client.ping()
            cache = RecommendationCache(client, ttl=args.ttl)
            print(f"Redis cache on at {args.redis} (ttl {args.ttl}s)")
        except Exception as exc:
            print(f"Redis not reachable ({exc}); running without cache")
    return Recommender(root / args.data, root / args.features, root / args.retrieval,
                       root / args.faiss, root / args.ranking, cache=cache,
                       triton_url=args.triton or None)


def show(label, result):
    print(f"\n{label}\n  source={result['source']} cached={result['cached']} {result['latency_ms']} ms")
    for row in result["items"]:
        print(f"   {row['item_id']:>8}  {row['title']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-root", default=".")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--retrieval", default="models/retrieval")
    ap.add_argument("--faiss", default="models/faiss")
    ap.add_argument("--ranking", default="models/ranking_v3")
    ap.add_argument("--redis", default="", help="host:port, e.g. localhost:6379 (optional)")
    ap.add_argument("--ttl", type=int, default=3600)
    ap.add_argument("--triton", default="", help="host:port of a Triton server, e.g. localhost:8000 (optional)")
    ap.add_argument("--user-id", type=int)
    ap.add_argument("--genres", default="", help="pipe separated, e.g. 'Comedy|Drama'")
    ap.add_argument("--liked", default="", help="comma separated movie ids")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    rec = build_from_args(args)
    genres = [g for g in args.genres.split("|") if g]
    liked = [int(x) for x in args.liked.split(",") if x]

    if args.demo:
        known = int(rec.user_ids[0])
        show("known user, first request", rec.recommend(known, args.k))
        show("known user, same request again", rec.recommend(known, args.k))
        show("new user, nothing known", rec.recommend(-1, args.k))
        show("new user, likes Comedy", rec.recommend(-1, args.k, genres=["Comedy"]))
        show("new user, likes 3 movies", rec.recommend(-1, args.k, liked_item_ids=[int(i) for i in rec.item_ids[:3]]))
    else:
        show(f"user {args.user_id}", rec.recommend(args.user_id if args.user_id is not None else -1,
                                                 args.k, genres=genres, liked_item_ids=liked))


if __name__ == "__main__":
    main()
