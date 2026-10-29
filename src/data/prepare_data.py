"""Step 1: clean MovieLens 25M and split by TIME (not randomly).

Usage:
    python src/data/prepare_data.py --raw data/raw/ml-25m --out data/processed
    python src/data/prepare_data.py --synthetic --out data/processed   # quick test, no download

Outputs (parquet) in --out:
    train.parquet, val.parquet, test.parquet  -> user_id, item_id, rating, label, ts
    items.parquet                              -> item_id, title, genres
    split_info.json                            -> cutoffs and cold-start stats
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

POSITIVE_THRESHOLD = 4.0   # rating >= 4 counts as a positive interaction ("click")
MIN_USER_INTERACTIONS = 5
MIN_ITEM_INTERACTIONS = 5
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1  # remaining 0.1 = test


def load_movielens(raw_dir: Path):
    ratings = pd.read_csv(raw_dir / "ratings.csv")
    movies = pd.read_csv(raw_dir / "movies.csv")
    ratings = ratings.rename(columns={"userId": "user_id", "movieId": "item_id", "timestamp": "ts"})
    movies = movies.rename(columns={"movieId": "item_id"})
    return ratings, movies


def make_synthetic(n_users=2000, n_items=500, n_rows=100_000, seed=42):
    rng = np.random.default_rng(seed)
    genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Romance", "Thriller", "Animation"]
    ratings = pd.DataFrame({
        "user_id": rng.integers(1, n_users + 1, n_rows),
        "item_id": rng.zipf(1.3, n_rows) % n_items + 1,   # popularity skew
        "rating": rng.choice([1, 2, 3, 3.5, 4, 4.5, 5], n_rows),
        "ts": np.sort(rng.integers(1_000_000_000, 1_600_000_000, n_rows)),
    })
    movies = pd.DataFrame({
        "item_id": np.arange(1, n_items + 1),
        "title": [f"Movie {i}" for i in range(1, n_items + 1)],
        "genres": ["|".join(rng.choice(genres, rng.integers(1, 4), replace=False)) for _ in range(n_items)],
    })
    return ratings, movies


def clean(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    ratings = ratings.drop_duplicates(["user_id", "item_id", "ts"]).dropna()
    ratings = ratings[ratings["item_id"].isin(movies["item_id"])]
    # iterative k-core style filter (2 passes is enough in practice)
    for _ in range(2):
        ratings = ratings[ratings.groupby("user_id")["item_id"].transform("size") >= MIN_USER_INTERACTIONS]
        ratings = ratings[ratings.groupby("item_id")["user_id"].transform("size") >= MIN_ITEM_INTERACTIONS]
    ratings = ratings.copy()
    ratings["label"] = (ratings["rating"] >= POSITIVE_THRESHOLD).astype("int8")
    return ratings.sort_values("ts").reset_index(drop=True)


def temporal_split(df: pd.DataFrame):
    t_train = df["ts"].quantile(TRAIN_FRAC)
    t_val = df["ts"].quantile(TRAIN_FRAC + VAL_FRAC)
    train = df[df["ts"] <= t_train]
    val = df[(df["ts"] > t_train) & (df["ts"] <= t_val)]
    test = df[df["ts"] > t_val]
    return train, val, test, int(t_train), int(t_val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw/ml-25m"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--synthetic", action="store_true", help="generate fake data for testing")
    args = ap.parse_args()

    ratings, movies = make_synthetic() if args.synthetic else load_movielens(args.raw)
    print(f"raw interactions: {len(ratings):,}")

    df = clean(ratings, movies)
    print(f"after cleaning:   {len(df):,}  users={df.user_id.nunique():,}  items={df.item_id.nunique():,}")

    train, val, test, t_train, t_val = temporal_split(df)

    # cold-start stats: who/what appears in eval but never in train?
    train_users, train_items = set(train.user_id), set(train.item_id)
    info = {
        "train_rows": len(train), "val_rows": len(val), "test_rows": len(test),
        "train_cutoff_ts": t_train, "val_cutoff_ts": t_val,
        "test_cold_users": int((~test.user_id.isin(train_users)).sum()),
        "test_cold_items": int((~test.item_id.isin(train_items)).sum()),
        "test_cold_user_ids": int(test.loc[~test.user_id.isin(train_users), "user_id"].nunique()),
        "positive_rate_train": float(train.label.mean()),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_parquet(args.out / f"{name}.parquet", index=False)
    movies[movies.item_id.isin(df.item_id.unique())].to_parquet(args.out / "items.parquet", index=False)
    (args.out / "split_info.json").write_text(json.dumps(info, indent=2))

    # sanity: no time leakage
    assert train.ts.max() <= val.ts.min() <= val.ts.max() <= test.ts.min() or len(val) == 0
    print(json.dumps(info, indent=2))
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
