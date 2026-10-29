"""Step 3: two-tower retrieval model (TensorFlow Recommenders).

The user tower and the item tower each turn their input into a vector of the
same size. Training pushes the vector of a user close to the vectors of movies
that user liked, and away from other movies in the same batch.

Run:
    python src/retrieval/train_two_tower.py --data data/processed --out models/retrieval

Outputs in --out:
    item_embeddings.npy, user_embeddings.npy   (L2-normalised, ready for FAISS)
    users.parquet, items.parquet               (id -> row index in the .npy files)
    metrics.json                               (recall / NDCG vs popularity baseline)
"""
import os

# TFRS needs Keras 2. Must be set before tensorflow is imported.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
import tensorflow_recommenders as tfrs

SEED = 42
TEMPERATURE = 0.05   # small value = sharper softmax over the in-batch candidates


# ---------------------------------------------------------------- data

def load_data(data_dir: Path, sample_frac: float):
    train = pd.read_parquet(data_dir / "train.parquet", columns=["user_id", "item_id", "label"])
    val = pd.read_parquet(data_dir / "val.parquet", columns=["user_id", "item_id", "label"])
    items = pd.read_parquet(data_dir / "items.parquet")

    if sample_frac < 1.0:
        keep_users = train["user_id"].drop_duplicates().sample(frac=sample_frac, random_state=SEED)
        train = train[train["user_id"].isin(keep_users)]
        val = val[val["user_id"].isin(keep_users)]

    # only items seen in train get a vector; the rest are cold items (step 6)
    item_ids = np.sort(train["item_id"].unique())
    user_ids = np.sort(train.loc[train["label"] == 1, "user_id"].unique())
    return train, val, items, user_ids, item_ids


def genre_matrix(items: pd.DataFrame, item_ids: np.ndarray):
    """Multi-hot genre matrix, one row per item (same order as item_ids)."""
    items = items.set_index("item_id").loc[item_ids]
    lists = items["genres"].fillna("").str.split("|")
    vocab = sorted({g for gs in lists for g in gs if g and g != "(no genres listed)"})
    gid = {g: i for i, g in enumerate(vocab)}
    mat = np.zeros((len(item_ids), len(vocab)), dtype=np.float32)
    for row, gs in enumerate(lists):
        for g in gs:
            if g in gid:
                mat[row, gid[g]] = 1.0
    return mat, vocab


# ---------------------------------------------------------------- model

class UserTower(tf.keras.Model):
    def __init__(self, n_users, dim):
        super().__init__()
        self.emb = tf.keras.layers.Embedding(n_users, dim)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(dim),
        ])

    def call(self, user_idx):
        return tf.math.l2_normalize(self.mlp(self.emb(user_idx)), axis=-1)


class ItemTower(tf.keras.Model):
    def __init__(self, n_items, genres, dim):
        super().__init__()
        self.id_emb = tf.keras.layers.Embedding(n_items, dim)
        self.genres = tf.constant(genres)
        self.genre_proj = tf.keras.layers.Dense(dim, use_bias=False)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(dim),
        ])

    def call(self, item_idx):
        g = self.genre_proj(tf.gather(self.genres, item_idx))
        x = tf.concat([self.id_emb(item_idx), g], axis=-1)
        return tf.math.l2_normalize(self.mlp(x), axis=-1)


class TwoTower(tfrs.Model):
    def __init__(self, n_users, n_items, genres, item_probs, dim):
        super().__init__()
        self.user_tower = UserTower(n_users, dim)
        self.item_tower = ItemTower(n_items, genres, dim)
        self.item_probs = tf.constant(item_probs, dtype=tf.float32)
        self.task = tfrs.tasks.Retrieval(temperature=TEMPERATURE, remove_accidental_hits=True)

    def compute_loss(self, features, training=False):
        user_vec = self.user_tower(features["user"])
        item_vec = self.item_tower(features["item"])
        # popular movies show up in many batches, so correct for how often
        # each one is sampled (otherwise the model just learns "recommend hits")
        probs = tf.gather(self.item_probs, features["item"])
        return self.task(
            user_vec, item_vec,
            candidate_sampling_probability=probs,
            candidate_ids=features["item"],
        )


class SaveWeightsEachEpoch(tf.keras.callbacks.Callback):
    """Keep a checkpoint after every epoch, so a disconnect doesn't lose the whole run."""

    def __init__(self, out_dir: Path):
        super().__init__()
        self.out_dir = out_dir

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_weights(str(self.out_dir / f"weights_epoch_{epoch + 1:02d}.weights.h5"))


# ---------------------------------------------------------------- evaluation

def evaluate(user_vecs, item_vecs, seen, truth, item_pop, ks=(10, 50, 100), max_k=100, batch=1024):
    """
    seen:  CSR matrix (users x items) of everything the user already interacted with in train
    truth: CSR matrix (users x items) of positives in the validation period
    Returns recall@K / NDCG@K for the model and for a most-popular baseline.
    """
    n_users = user_vecs.shape[0]
    discounts = 1.0 / np.log2(np.arange(2, max_k + 2))
    res = {name: {f"recall@{k}": 0.0 for k in ks} | {f"ndcg@{k}": 0.0 for k in ks}
           for name in ("two_tower", "popularity")}
    counted = 0

    for start in range(0, n_users, batch):
        end = min(start + batch, n_users)
        has_truth = np.asarray(truth[start:end].sum(axis=1)).ravel() > 0
        if not has_truth.any():
            continue
        mask = seen[start:end].toarray().astype(bool)
        truth_b = truth[start:end].toarray().astype(bool)

        score_sets = {
            "two_tower": user_vecs[start:end] @ item_vecs.T,
            "popularity": np.tile(item_pop, (end - start, 1)).astype(np.float32),
        }
        for name, scores in score_sets.items():
            scores = np.where(mask, -np.inf, scores)
            top = np.argpartition(-scores, max_k, axis=1)[:, :max_k]
            order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
            top = np.take_along_axis(top, order, axis=1)
            hits = np.take_along_axis(truth_b, top, axis=1)       # users x max_k
            n_true = truth_b.sum(axis=1)

            for k in ks:
                h = hits[:, :k]
                recall = h.sum(axis=1) / np.maximum(n_true, 1)
                dcg = (h * discounts[:k]).sum(axis=1)
                ideal = np.array([discounts[:min(int(n), k)].sum() for n in n_true])
                ndcg = dcg / np.maximum(ideal, 1e-9)
                res[name][f"recall@{k}"] += recall[has_truth].sum()
                res[name][f"ndcg@{k}"] += ndcg[has_truth].sum()
        counted += int(has_truth.sum())

    for name in res:
        for key in res[name]:
            res[name][key] = round(float(res[name][key] / max(counted, 1)), 4)
    res["users_evaluated"] = counted
    return res


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=Path("models/retrieval"))
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sample-frac", type=float, default=1.0, help="use a fraction of users (quick tests)")
    ap.add_argument("--eval-users", type=int, default=20000)
    args = ap.parse_args()

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    train, val, items, user_ids, item_ids = load_data(args.data, args.sample_frac)
    genres, genre_vocab = genre_matrix(items, item_ids)
    print(f"users={len(user_ids):,} items={len(item_ids):,} genres={len(genre_vocab)}")

    user_index = pd.Index(user_ids)
    item_index = pd.Index(item_ids)

    # training pairs = (user, liked movie)
    pos = train[train["label"] == 1]
    u = user_index.get_indexer(pos["user_id"])
    i = item_index.get_indexer(pos["item_id"])
    ok = (u >= 0) & (i >= 0)
    u, i = u[ok].astype(np.int32), i[ok].astype(np.int32)
    print(f"training pairs: {len(u):,}")

    # how often each item appears as a positive -> sampling probability
    counts = np.bincount(i, minlength=len(item_ids)).astype(np.float64) + 1.0
    item_probs = (counts / counts.sum()).astype(np.float32)

    ds = (tf.data.Dataset.from_tensor_slices({"user": u, "item": i})
          .shuffle(1_000_000, seed=SEED)
          .batch(args.batch, drop_remainder=True)
          .prefetch(tf.data.AUTOTUNE))

    model = TwoTower(len(user_ids), len(item_ids), genres, item_probs, args.dim)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    model.fit(ds, epochs=args.epochs, verbose=2, callbacks=[SaveWeightsEachEpoch(args.out)])
    model.save_weights(str(args.out / "two_tower_final.weights.h5"))

    # export embeddings
    item_vecs = model.item_tower(tf.range(len(item_ids))).numpy()
    user_vecs = model.user_tower(tf.range(len(user_ids))).numpy()

    # evaluation on the validation period
    seen_u = user_index.get_indexer(train["user_id"])
    seen_i = item_index.get_indexer(train["item_id"])
    k = (seen_u >= 0) & (seen_i >= 0)
    shape = (len(user_ids), len(item_ids))
    seen = sp.csr_matrix((np.ones(k.sum(), dtype=np.int8), (seen_u[k], seen_i[k])), shape=shape)
    seen.data[:] = 1

    vpos = val[val["label"] == 1]
    vu = user_index.get_indexer(vpos["user_id"])
    vi = item_index.get_indexer(vpos["item_id"])
    k = (vu >= 0) & (vi >= 0)
    truth = sp.csr_matrix((np.ones(k.sum(), dtype=np.int8), (vu[k], vi[k])), shape=shape)
    truth.data[:] = 1

    # evaluate on a random subset of users that have validation positives
    rng = np.random.default_rng(SEED)
    active = np.unique(vu[k])
    eval_rows = np.sort(rng.choice(active, size=min(args.eval_users, len(active)), replace=False))
    metrics = evaluate(
        user_vecs[eval_rows], item_vecs, seen[eval_rows], truth[eval_rows],
        item_pop=counts.astype(np.float32),
    )
    print(json.dumps(metrics, indent=2))

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "item_embeddings.npy", item_vecs)
    np.save(args.out / "user_embeddings.npy", user_vecs)
    pd.DataFrame({"user_id": user_ids, "user_idx": np.arange(len(user_ids))}).to_parquet(args.out / "users.parquet", index=False)
    pd.DataFrame({"item_id": item_ids, "item_idx": np.arange(len(item_ids))}).to_parquet(args.out / "items.parquet", index=False)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
