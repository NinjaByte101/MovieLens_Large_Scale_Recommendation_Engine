"""Step 5: DCN ranking model that re-scores the candidates found by retrieval.

Retrieval only knows "how close is this user's vector to this movie's vector".
The ranker looks at one (user, movie) pair at a time and predicts the chance the
user likes it, using richer signals: user activity and average rating, movie
popularity and average rating, genres, and learned user / movie embeddings.

Architecture (DCN, stacked): the cross layers build explicit feature
interactions, and a deep MLP on top learns the rest.

Training data is `ranker_train` from step 2. Every feature in it only uses
interactions that happened BEFORE that row, so the model never sees the future.

Evaluation answers the question that matters: take FAISS candidates for
validation users, drop what they already watched, and compare
    retrieval order   vs   ranker order
on NDCG@K / Recall@K with the same users and candidates.

Run:
    python src/ranking/train_ranker.py --data data/processed --features data/features \
        --retrieval models/retrieval --faiss models/faiss --out models/ranking
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

SEED = 42
NUM_COLS = ["user_n", "user_avg_rating", "user_pos_rate",
            "item_log_pop", "item_avg_rating", "item_pos_rate"]


# ---------------------------------------------------------------- features

def numeric_matrix(df: pd.DataFrame) -> np.ndarray:
    x = df[NUM_COLS].astype("float32").copy()
    x["user_n"] = np.log1p(x["user_n"])      # activity is heavily skewed, compress it
    return x.to_numpy()


def to_index(ids: pd.Series, vocab: pd.Index) -> np.ndarray:
    """Row in the retrieval vocabulary + 1. Unknown ids map to 0 (the 'unknown' slot)."""
    return (vocab.get_indexer(ids) + 1).astype("int32")


def genre_table(items: pd.DataFrame, item_ids: np.ndarray) -> np.ndarray:
    """Multi-hot genres, one row per item in retrieval order. Row 0 is the unknown slot."""
    lists = items.set_index("item_id").loc[item_ids, "genres"].fillna("").str.split("|")
    vocab = sorted({g for gs in lists for g in gs if g and g != "(no genres listed)"})
    gid = {g: i for i, g in enumerate(vocab)}
    table = np.zeros((len(item_ids) + 1, len(vocab)), dtype="float32")
    for row, gs in enumerate(lists, start=1):
        for g in gs:
            if g in gid:
                table[row, gid[g]] = 1.0
    return table


def snapshot_rows(df, user_feats, item_feats):
    """Attach end-of-train user / item stats to plain (user, item) rows."""
    out = df.merge(user_feats[["user_id", "user_n", "user_avg_rating", "user_pos_rate"]], on="user_id", how="left")
    out = out.merge(item_feats[["item_id", "item_log_pop", "item_avg_rating", "item_pos_rate"]], on="item_id", how="left")
    return out.fillna(0.0)


# ---------------------------------------------------------------- model

class DCNRanker(tf.keras.Model):
    def __init__(self, n_users, n_items, genres, dim=16, n_cross=2, deep=(256, 128), dropout=0.2):
        super().__init__()
        self.user_emb = tf.keras.layers.Embedding(n_users + 1, dim)
        self.item_emb = tf.keras.layers.Embedding(n_items + 1, dim)
        self.genres = tf.constant(genres)
        self.cross = [tfrs.layers.dcn.Cross(kernel_initializer="glorot_uniform") for _ in range(n_cross)]
        hidden = []
        for units in deep:
            hidden += [tf.keras.layers.Dense(units, activation="relu"), tf.keras.layers.Dropout(dropout)]
        self.deep = tf.keras.Sequential(hidden)
        self.out = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        x0 = tf.concat([
            self.user_emb(inputs["user"]),
            self.item_emb(inputs["item"]),
            tf.gather(self.genres, inputs["item"]),
            inputs["num"],
        ], axis=-1)
        x = x0
        for layer in self.cross:          # each layer mixes x0 with the previous output
            x = layer(x0, x)
        return tf.squeeze(self.out(self.deep(x, training=training)), axis=-1)


def make_dataset(user, item, num, label, batch, shuffle):
    ds = tf.data.Dataset.from_tensor_slices(({"user": user, "item": item, "num": num}, label))
    if shuffle:
        ds = ds.shuffle(500_000, seed=SEED)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------- ranking metrics

def rank_metrics(ranked, truths, ks=(10, 50)):
    """ranked: list of item-index arrays (best first). truths: list of sets of item indices."""
    discounts = 1.0 / np.log2(np.arange(2, max(ks) + 2))
    res = {f"{m}@{k}": 0.0 for k in ks for m in ("recall", "ndcg")}
    for items, truth in zip(ranked, truths):
        hits = np.array([i in truth for i in items[:max(ks)]], dtype=float)
        for k in ks:
            res[f"recall@{k}"] += hits[:k].sum() / len(truth)
            ideal = discounts[:min(len(truth), k)].sum()
            res[f"ndcg@{k}"] += (hits[:k] * discounts[:k]).sum() / ideal
    return {key: round(v / len(ranked), 4) for key, v in res.items()}


def evaluate_rerank(model, args, user_ids, item_ids, num_mean, num_std, user_feats, item_feats):
    rng = np.random.default_rng(SEED)
    user_vocab, item_vocab = pd.Index(user_ids), pd.Index(item_ids)

    train = pd.read_parquet(args.data / "train.parquet", columns=["user_id", "item_id"])
    val = pd.read_parquet(args.data / "val.parquet", columns=["user_id", "item_id", "label"])
    val = val[val["label"] == 1]

    tu, ti = user_vocab.get_indexer(train["user_id"]), item_vocab.get_indexer(train["item_id"])
    ok = (tu >= 0) & (ti >= 0)
    seen = sp.csr_matrix((np.ones(ok.sum(), dtype=np.int8), (tu[ok], ti[ok])),
                         shape=(len(user_ids), len(item_ids)))

    vu, vi = user_vocab.get_indexer(val["user_id"]), item_vocab.get_indexer(val["item_id"])
    ok = (vu >= 0) & (vi >= 0)
    truth_by_user = pd.Series(vi[ok]).groupby(vu[ok]).agg(set)

    eval_users = np.sort(rng.choice(truth_by_user.index.to_numpy(),
                                    size=min(args.eval_users, len(truth_by_user)), replace=False))

    # candidates from the FAISS index, minus anything the user already interacted with
    index = faiss.read_index(str(args.faiss / f"items_{args.index}.index"))
    user_vecs = np.load(args.retrieval / "user_embeddings.npy")[eval_users].astype("float32")
    _, fetched = index.search(user_vecs, min(1000, len(item_ids)))
    cands = []
    for row, u in zip(fetched, eval_users):
        already = set(seen[u].indices)
        cands.append(np.array([i for i in row if i not in already][:args.candidates]))

    # features for every (user, candidate) pair, built from end-of-train snapshots
    lens = [len(c) for c in cands]
    rows = pd.DataFrame({
        "user_id": np.repeat(user_ids[eval_users], lens),
        "item_id": item_ids[np.concatenate(cands)],
    })
    rows = snapshot_rows(rows, user_feats, item_feats)
    num = (numeric_matrix(rows) - num_mean) / num_std
    scores = model.predict(
        {"user": np.repeat(eval_users + 1, lens).astype("int32"),
         "item": (np.concatenate(cands) + 1).astype("int32"),
         "num": num.astype("float32")},
        batch_size=65536, verbose=0)

    reranked, blended, start = [], [], 0
    for c in cands:
        s = scores[start:start + len(c)]
        by_ranker = np.argsort(-s)
        reranked.append(c[by_ranker])
        # reciprocal-rank fusion: reward items that both stages put near the top
        ranker_pos = np.empty(len(c))
        ranker_pos[by_ranker] = np.arange(len(c))
        fused = 1.0 / (60 + np.arange(len(c))) + 1.0 / (60 + ranker_pos)
        blended.append(c[np.argsort(-fused)])
        start += len(c)

    truths = [truth_by_user[u] for u in eval_users]
    ceiling = float(np.mean([len(set(c) & t) / len(t) for c, t in zip(cands, truths)]))
    return {
        "users_evaluated": int(len(eval_users)),
        "candidates_per_user": args.candidates,
        "candidate_recall_ceiling": round(ceiling, 4),
        "retrieval_order": rank_metrics(cands, truths),
        "ranker_order": rank_metrics(reranked, truths),
        "blended_order": rank_metrics(blended, truths),
    }


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--features", type=Path, default=Path("data/features"))
    ap.add_argument("--retrieval", type=Path, default=Path("models/retrieval"))
    ap.add_argument("--faiss", type=Path, default=Path("models/faiss"))
    ap.add_argument("--out", type=Path, default=Path("models/ranking"))
    ap.add_argument("--index", default="hnsw", choices=["flat", "ivf", "hnsw"])
    ap.add_argument("--max-rows", type=int, default=4_000_000, help="training rows sampled from ranker_train")
    ap.add_argument("--val-rows", type=int, default=500_000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--id-dropout", type=float, default=0.05,
                    help="share of rows where the user / movie ID is hidden, so the 'unknown' slot gets trained")
    ap.add_argument("--since-quantile", type=float, default=0.7,
                    help="train only on the latest part of the train period (0 = use everything)")
    ap.add_argument("--candidates", type=int, default=200)
    ap.add_argument("--eval-users", type=int, default=5000)
    args = ap.parse_args()

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    user_ids = pd.read_parquet(args.retrieval / "users.parquet")["user_id"].to_numpy()
    item_ids = pd.read_parquet(args.retrieval / "items.parquet")["item_id"].to_numpy()
    user_vocab, item_vocab = pd.Index(user_ids), pd.Index(item_ids)
    items_meta = pd.read_parquet(args.data / "items.parquet")
    genres = genre_table(items_meta, item_ids)

    user_feats = pd.read_parquet(args.features / "user_features.parquet")
    item_feats = pd.read_parquet(args.features / "item_features.parquet")

    # training rows: the point-in-time features written by Spark in step 2
    cols = ["user_id", "item_id", "label", "ts"] + NUM_COLS
    train = pd.read_parquet(args.features / "ranker_train.parquet", columns=cols)
    if args.since_quantile > 0:
        # early rows have tiny counts (few interactions so far); serving sees much bigger ones
        train = train[train["ts"] >= train["ts"].quantile(args.since_quantile)]
    if len(train) > args.max_rows:
        train = train.sample(args.max_rows, random_state=SEED)
    print(f"training rows: {len(train):,}  positive rate: {train['label'].mean():.3f}")

    num_train = numeric_matrix(train)
    num_mean, num_std = num_train.mean(axis=0), np.maximum(num_train.std(axis=0), 1e-6)
    num_train = ((num_train - num_mean) / num_std).astype("float32")

    # validation rows (a later time period) use end-of-train snapshots, like serving would
    val = pd.read_parquet(args.data / "val.parquet", columns=["user_id", "item_id", "label"])
    if len(val) > args.val_rows:
        val = val.sample(args.val_rows, random_state=SEED)
    val = snapshot_rows(val, user_feats, item_feats)
    num_val = ((numeric_matrix(val) - num_mean) / num_std).astype("float32")

    # if the validation features sit far from the training ones (z-score mean far from 0),
    # the model is being asked about numbers it never saw: a train/serve mismatch
    print("validation feature shift (z-score mean vs training):",
          dict(zip(NUM_COLS, num_val.mean(axis=0).round(2).tolist())))

    u_train = to_index(train["user_id"], user_vocab)
    i_train = to_index(train["item_id"], item_vocab)
    rng = np.random.default_rng(SEED)
    u_train[rng.random(len(u_train)) < args.id_dropout] = 0     # train the 'unknown user' slot
    i_train[rng.random(len(i_train)) < args.id_dropout] = 0     # and the 'unknown movie' slot

    # the ranker only ever re-scores candidates for known users, so monitor it on those rows
    u_val, i_val = to_index(val["user_id"], user_vocab), to_index(val["item_id"], item_vocab)
    warm = (u_val > 0) & (i_val > 0)
    print(f"validation rows: {len(val):,}  known user and movie: {warm.mean():.1%}")

    train_ds = make_dataset(u_train, i_train, num_train, train["label"].to_numpy("float32"),
                            args.batch, shuffle=True)
    val_ds = make_dataset(u_val[warm], i_val[warm], num_val[warm],
                          val["label"].to_numpy("float32")[warm], args.batch, shuffle=False)

    model = DCNRanker(len(user_ids), len(item_ids), genres, dim=args.dim, dropout=args.dropout)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.AUC(name="auc")])

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ranker_config.json").write_text(json.dumps({
        **{k: str(v) for k, v in vars(args).items()},
        "num_cols": NUM_COLS, "num_mean": num_mean.tolist(), "num_std": num_std.tolist(),
    }, indent=2))
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(args.out / "ranker_epoch_{epoch:02d}.weights.h5"), save_weights_only=True)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=1, restore_best_weights=True)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
              callbacks=[checkpoint, early_stop], verbose=2)
    model.save_weights(str(args.out / "ranker_final.weights.h5"))

    metrics = evaluate_rerank(model, args, user_ids, item_ids, num_mean, num_std, user_feats, item_feats)
    print(json.dumps(metrics, indent=2))
    (args.out / "ranker_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()