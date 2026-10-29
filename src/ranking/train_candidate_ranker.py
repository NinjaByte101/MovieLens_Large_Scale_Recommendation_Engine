"""Step 5 (v3): train the ranker on the candidates retrieval actually produces.

The v1/v2 ranker learned "given that this user rated this movie, did they like
it?". But at serving time it re-orders movies the user has NOT rated, and the
question there is "will they watch it and like it?". Those are different
questions, so a good AUC did not turn into a better ranking.

Here the training rows look exactly like serving:
    for each user with activity in the validation period
        1. take the FAISS candidates (movies not yet seen in train)
        2. features = retrieval score, rank, user stats, movie stats, genre affinity
        3. label    = 1 if the user liked that movie in the validation period

The retrieval embeddings were trained on the train period only, so the
retrieval score is a legitimate feature here (it never saw these labels).
Users are split into folds, and each user is scored by a model that never
trained on that user, so the comparison with retrieval order is fair.

Run:
    python src/ranking/train_candidate_ranker.py --data data/processed --features data/features \
        --retrieval models/retrieval --faiss models/faiss --out models/ranking_v3
"""
import os

# TFRS needs Keras 2. Must be set before tensorflow is imported.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
import tensorflow_recommenders as tfrs

from train_ranker import SEED, genre_table, rank_metrics, snapshot_rows

FEATURES = ["retrieval_score", "log_rank", "user_n", "user_avg_rating", "user_pos_rate",
            "item_log_pop", "item_avg_rating", "item_pos_rate", "genre_affinity"]


# ---------------------------------------------------------------- candidates

def retrieve(index, user_vecs, users, seen, n_keep, fetch=1000):
    """FAISS top candidates per user, dropping movies the user already interacted with."""
    scores, ids = index.search(user_vecs, min(fetch, index.ntotal))
    items, kept_scores = [], []
    for u, s_row, i_row in zip(users, scores, ids):
        keep = (i_row >= 0) & ~np.isin(i_row, seen[u].indices)
        items.append(i_row[keep][:n_keep])
        kept_scores.append(s_row[keep][:n_keep])
    return items, kept_scores


def genre_profiles(train, user_vocab, item_vocab, genres):
    """Share of each genre among the movies a user liked in train (users x genres)."""
    pos = train[train["label"] == 1]
    u, i = user_vocab.get_indexer(pos["user_id"]), item_vocab.get_indexer(pos["item_id"])
    ok = (u >= 0) & (i >= 0)
    liked = sp.csr_matrix((np.ones(ok.sum(), dtype="float32"), (u[ok], i[ok])),
                          shape=(len(user_vocab), len(item_vocab)))
    liked.data[:] = 1.0
    counts = liked @ genres
    return counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)


def feature_matrix(users, cand_items, cand_scores, user_ids, item_ids,
                   user_feats, item_feats, profiles, genres):
    """One row per (user, candidate). Used by training AND serving, so they cannot drift apart."""
    lens = np.array([len(c) for c in cand_items])
    u_rows = np.repeat(users, lens)
    i_rows = np.concatenate(cand_items)
    rank = np.concatenate([np.arange(n) for n in lens])

    frame = snapshot_rows(pd.DataFrame({"user_id": user_ids[u_rows], "item_id": item_ids[i_rows]}),
                          user_feats, item_feats)
    g = genres[i_rows]
    affinity = (profiles[u_rows] * g).sum(axis=1) / np.maximum(g.sum(axis=1), 1.0)

    X = np.column_stack([
        np.concatenate(cand_scores), np.log1p(rank), np.log1p(frame["user_n"]),
        frame["user_avg_rating"], frame["user_pos_rate"], frame["item_log_pop"],
        frame["item_avg_rating"], frame["item_pos_rate"], affinity,
    ]).astype("float32")
    return X, i_rows.astype("int32"), lens


def build_rows(users, cand_items, cand_scores, truth, *feature_args):
    """Training rows: the shared features plus the label (did the user like it in the next period)."""
    X, i_rows, lens = feature_matrix(users, cand_items, cand_scores, *feature_args)
    y = np.concatenate([np.isin(c, list(truth[u])) for u, c in zip(users, cand_items)]).astype("float32")
    return X, i_rows, y, lens


# ---------------------------------------------------------------- model

class CandidateRanker(tf.keras.Model):
    """DCN over the numeric features plus the movie's genres (no ID embeddings)."""

    def __init__(self, genres, n_cross=2, deep=(128, 64), dropout=0.2):
        super().__init__()
        self.genres = tf.constant(genres)
        self.cross = [tfrs.layers.dcn.Cross(kernel_initializer="glorot_uniform") for _ in range(n_cross)]
        hidden = []
        for units in deep:
            hidden += [tf.keras.layers.Dense(units, activation="relu"), tf.keras.layers.Dropout(dropout)]
        self.deep = tf.keras.Sequential(hidden)
        self.out = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        x0 = tf.concat([inputs["num"], tf.gather(self.genres, inputs["item"])], axis=-1)
        x = x0
        for layer in self.cross:
            x = layer(x0, x)
        return tf.squeeze(self.out(self.deep(x, training=training)), axis=-1)


def fit(X, items, y, genres, args):
    model = CandidateRanker(genres, dropout=args.dropout)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss=tf.keras.losses.BinaryCrossentropy())
    ds = (tf.data.Dataset.from_tensor_slices(({"num": X, "item": items}, y))
          .shuffle(200_000, seed=SEED).batch(args.batch).prefetch(tf.data.AUTOTUNE))
    model.fit(ds, epochs=args.epochs, verbose=0)
    return model


def predict(model, X, items):
    return model.predict({"num": X, "item": items}, batch_size=65536, verbose=0)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--features", type=Path, default=Path("data/features"))
    ap.add_argument("--retrieval", type=Path, default=Path("models/retrieval"))
    ap.add_argument("--faiss", type=Path, default=Path("models/faiss"))
    ap.add_argument("--out", type=Path, default=Path("models/ranking_v3"))
    ap.add_argument("--index", default="hnsw", choices=["flat", "ivf", "hnsw"])
    ap.add_argument("--candidates", type=int, default=200)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    args = ap.parse_args()

    tf.random.set_seed(SEED)
    rng = np.random.default_rng(SEED)

    user_ids = pd.read_parquet(args.retrieval / "users.parquet")["user_id"].to_numpy()
    item_ids = pd.read_parquet(args.retrieval / "items.parquet")["item_id"].to_numpy()
    user_vocab, item_vocab = pd.Index(user_ids), pd.Index(item_ids)
    genres = genre_table(pd.read_parquet(args.data / "items.parquet"), item_ids)[1:]   # drop the unknown row

    user_feats = pd.read_parquet(args.features / "user_features.parquet")
    item_feats = pd.read_parquet(args.features / "item_features.parquet")
    train = pd.read_parquet(args.data / "train.parquet", columns=["user_id", "item_id", "label"])
    val = pd.read_parquet(args.data / "val.parquet", columns=["user_id", "item_id", "label"])

    # what each user already interacted with in train (never recommend it again)
    tu, ti = user_vocab.get_indexer(train["user_id"]), item_vocab.get_indexer(train["item_id"])
    ok = (tu >= 0) & (ti >= 0)
    seen = sp.csr_matrix((np.ones(ok.sum(), dtype=np.int8), (tu[ok], ti[ok])),
                         shape=(len(user_ids), len(item_ids)))

    # labels: movies each user liked in the validation period
    val = val[val["label"] == 1]
    vu, vi = user_vocab.get_indexer(val["user_id"]), item_vocab.get_indexer(val["item_id"])
    ok = (vu >= 0) & (vi >= 0)
    truth = pd.Series(vi[ok]).groupby(vu[ok]).agg(set)
    users = truth.index.to_numpy()
    print(f"users with validation activity: {len(users):,}")

    index = faiss.read_index(str(args.faiss / f"items_{args.index}.index"))
    user_vecs = np.load(args.retrieval / "user_embeddings.npy")[users].astype("float32")
    cand_items, cand_scores = retrieve(index, user_vecs, users, seen, args.candidates)

    profiles = genre_profiles(train, user_vocab, item_vocab, genres)
    X, items, y, lens = build_rows(users, cand_items, cand_scores, truth, user_ids, item_ids,
                                   user_feats, item_feats, profiles, genres)
    print(f"candidate rows: {len(y):,}  positive rate: {y.mean():.3f}")

    mean, std = X.mean(axis=0), np.maximum(X.std(axis=0), 1e-6)
    X = ((X - mean) / std).astype("float32")

    # out-of-fold scores: every user is scored by a model that never trained on that user
    user_fold = rng.permutation(len(users)) % args.folds
    row_fold = np.repeat(user_fold, lens)
    oof = np.zeros(len(y), dtype="float32")
    for fold in range(args.folds):
        tr, te = row_fold != fold, row_fold == fold
        model = fit(X[tr], items[tr], y[tr], genres, args)
        oof[te] = predict(model, X[te], items[te])
        print(f"fold {fold + 1}/{args.folds} done")

    auc = tf.keras.metrics.AUC()
    auc.update_state(y, oof)

    ranked, blended, start = {"retrieval": [], "ranker": []}, [], 0
    for c in cand_items:
        s = oof[start:start + len(c)]
        by_ranker = np.argsort(-s)
        ranked["retrieval"].append(c)
        ranked["ranker"].append(c[by_ranker])
        pos = np.empty(len(c))
        pos[by_ranker] = np.arange(len(c))
        fused = 1.0 / (60 + np.arange(len(c))) + 1.0 / (60 + pos)    # reciprocal-rank fusion
        blended.append(c[np.argsort(-fused)])
        start += len(c)

    truths = [truth[u] for u in users]
    metrics = {
        "users_evaluated": int(len(users)),
        "candidates_per_user": args.candidates,
        "candidate_recall_ceiling": round(float(np.mean([len(set(c) & t) / len(t) for c, t in zip(cand_items, truths)])), 4),
        "out_of_fold_auc": round(float(auc.result().numpy()), 4),
        "retrieval_order": rank_metrics(ranked["retrieval"], truths),
        "ranker_order": rank_metrics(ranked["ranker"], truths),
        "blended_order": rank_metrics(blended, truths),
    }
    print(json.dumps(metrics, indent=2))

    # final model on all users, for the serving step
    args.out.mkdir(parents=True, exist_ok=True)
    final = fit(X, items, y, genres, args)
    final.save_weights(str(args.out / "candidate_ranker.weights.h5"))
    (args.out / "candidate_ranker_config.json").write_text(json.dumps({
        **{k: str(v) for k, v in vars(args).items()},
        "features": FEATURES, "mean": mean.tolist(), "std": std.tolist(),
    }, indent=2))
    (args.out / "candidate_ranker_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
