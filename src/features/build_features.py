"""Step 2: build user / item / history features with Spark.

Reads the time-split parquet files from step 1 and writes:
    item_features.parquet    final snapshot of item stats (train period only)
    user_features.parquet    final snapshot of user stats (train period only)
    user_history.parquet     last N liked items per user, newest first
    ranker_train.parquet     train interactions with point-in-time features

Run:
    python src/features/build_features.py --data data/processed --out data/features
"""
import argparse
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

HISTORY_LEN = 20


def get_spark():
    return (
        SparkSession.builder.appName("recsys-features")
        .master("local[*]")
        .config("spark.driver.memory", "8g")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def item_snapshot(train):
    """Per-item stats over the whole train period."""
    return train.groupBy("item_id").agg(
        F.count("*").alias("item_pop"),
        F.sum("label").alias("item_pos"),
        F.avg("rating").alias("item_avg_rating"),
    ).withColumn("item_pos_rate", F.col("item_pos") / F.col("item_pop")) \
     .withColumn("item_log_pop", F.log1p("item_pop"))


def top_genre_per_user(train, items):
    """Most frequent genre among the movies a user liked."""
    liked = train.filter(F.col("label") == 1).select("user_id", "item_id")
    exploded = items.select("item_id", F.explode(F.split("genres", r"\|")).alias("genre"))
    counts = liked.join(exploded, "item_id").groupBy("user_id", "genre").count()
    w = Window.partitionBy("user_id").orderBy(F.col("count").desc(), F.col("genre"))
    return (counts.withColumn("rn", F.row_number().over(w))
            .filter("rn = 1")
            .select("user_id", F.col("genre").alias("user_top_genre")))


def user_snapshot(train, items):
    stats = train.groupBy("user_id").agg(
        F.count("*").alias("user_n"),
        F.avg("rating").alias("user_avg_rating"),
        F.avg("label").alias("user_pos_rate"),
        F.max("ts").alias("user_last_ts"),
    )
    return stats.join(top_genre_per_user(train, items), "user_id", "left") \
                .fillna({"user_top_genre": "unknown"})


def user_history(train):
    """Last HISTORY_LEN liked items per user, newest first."""
    w = Window.partitionBy("user_id").orderBy(F.col("ts").desc(), F.col("item_id"))
    recent = (train.filter(F.col("label") == 1)
              .withColumn("rn", F.row_number().over(w))
              .filter(F.col("rn") <= HISTORY_LEN))
    return (recent.groupBy("user_id")
            .agg(F.sort_array(F.collect_list(F.struct("rn", "item_id"))).alias("h"))
            .select("user_id", F.col("h.item_id").alias("history")))


def ranker_training_table(train, items):
    """
    Each row only sees stats from interactions BEFORE it. Otherwise the user's
    average rating would include the very rating we are trying to predict.
    """
    # the frame ends at the previous row (-1), so the current row is excluded
    wu = Window.partitionBy("user_id").orderBy("ts", "item_id") \
               .rowsBetween(Window.unboundedPreceding, -1)
    wi = Window.partitionBy("item_id").orderBy("ts", "item_id") \
               .rowsBetween(Window.unboundedPreceding, -1)

    df = (train
          .withColumn("user_n", F.count("*").over(wu))
          .withColumn("user_avg_rating", F.avg("rating").over(wu))
          .withColumn("user_pos_rate", F.avg("label").over(wu))
          .withColumn("item_pop", F.count("*").over(wi))
          .withColumn("item_avg_rating", F.avg("rating").over(wi))
          .withColumn("item_pos_rate", F.avg("label").over(wi))
          .withColumn("item_log_pop", F.log1p("item_pop")))

    # first interaction of a user/item has no past -> fall back to neutral values
    df = df.fillna({
        "user_avg_rating": 0.0, "user_pos_rate": 0.0,
        "item_avg_rating": 0.0, "item_pos_rate": 0.0,
    })
    return df.join(items.select("item_id", "genres"), "item_id", "left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=Path("data/features"))
    args = ap.parse_args()

    spark = get_spark()
    train = spark.read.parquet(str(args.data / "train.parquet"))
    items = spark.read.parquet(str(args.data / "items.parquet"))

    outputs = {
        "item_features": item_snapshot(train),
        "user_features": user_snapshot(train, items),
        "user_history": user_history(train),
        "ranker_train": ranker_training_table(train, items),
    }
    for name, frame in outputs.items():
        path = str(args.out / f"{name}.parquet")
        frame.write.mode("overwrite").parquet(path)
        print(f"{name}: {frame.count():,} rows -> {path}")

    spark.stop()


if __name__ == "__main__":
    main()
