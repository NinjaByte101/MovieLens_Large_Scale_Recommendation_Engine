"""Step 7a: a simple A/B test of two ranking policies, replayed on the TEST period.

There is no live traffic, so this is a replay: each user is deterministically assigned to
arm A or B, served that arm's top-K list, and "clicks" are the movies the user really liked
(rating >= 4) in the test period, data that neither retrieval nor the ranker ever trained on.

    A (control)    FAISS retrieval order
    B (treatment)  retrieval candidates re-ranked by the DCN ranker

Reported per arm: hit rate (share of users who liked at least one recommended movie),
CTR (liked movies / recommended movies), Recall@K and NDCG@K, with 95% bootstrap intervals
and Welch t-test p-values. A paired comparison (every user scored under both policies) is
also printed, because it needs far fewer users to detect a difference.

Limits worth stating: replay data only contains movies users chose to rate, there is no
position bias model, and nothing here is a live experiment.

Run:
    python src/serving/ab_test.py --artifacts-root outputs --data processed --features features \
        --retrieval retrieval --faiss faiss --ranking ranking_v3 --out outputs/ab_test
"""
import os

# TFRS needs Keras 2. Must be set before tensorflow is imported.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parent))
from recommender import Recommender  # noqa: E402
from train_candidate_ranker import retrieve  # noqa: E402

SEED = 42
METRICS = ["hit", "ctr", "recall", "ndcg"]


def assign_arm(user_id, salt):
    """Stable 50/50 split: the same user always lands in the same arm."""
    digest = hashlib.md5(f"{salt}:{user_id}".encode()).hexdigest()
    return "A" if int(digest, 16) % 2 == 0 else "B"


def user_metrics(recs, truth, k):
    hits = np.array([i in truth for i in recs[:k]], dtype=float)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    ideal = discounts[:min(len(truth), k)].sum()
    return {
        "hit": float(hits.sum() > 0),
        "ctr": hits.sum() / k,
        "recall": hits.sum() / len(truth),
        "ndcg": float((hits * discounts[:len(hits)]).sum() / ideal),
    }


def bootstrap_diff(a, b, rng, n=2000):
    """95% interval for mean(b) - mean(a), resampling users inside each arm."""
    diffs = [rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean() for _ in range(n)]
    return [round(float(x), 4) for x in np.percentile(diffs, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-root", default=".")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--retrieval", default="models/retrieval")
    ap.add_argument("--faiss", default="models/faiss")
    ap.add_argument("--ranking", default="models/ranking_v3")
    ap.add_argument("--out", default="outputs/ab_test")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-users", type=int, default=5000)
    ap.add_argument("--salt", default="ab-2026-10")
    args = ap.parse_args()

    root = Path(args.artifacts_root)
    rec = Recommender(root / args.data, root / args.features, root / args.retrieval,
                      root / args.faiss, root / args.ranking)

    # at test time a user has already seen everything from train AND validation
    val = pd.read_parquet(root / args.data / "val.parquet", columns=["user_id", "item_id"])
    vu, vi = rec.user_vocab.get_indexer(val["user_id"]), rec.item_vocab.get_indexer(val["item_id"])
    ok = (vu >= 0) & (vi >= 0)
    val_seen = sp.csr_matrix((np.ones(ok.sum(), dtype=np.int8), (vu[ok], vi[ok])), shape=rec.seen.shape)
    rec.seen = (rec.seen + val_seen).tocsr()
    rec.seen.data[:] = 1

    # "clicks": movies the user liked in the test period
    test = pd.read_parquet(root / args.data / "test.parquet", columns=["user_id", "item_id", "label"])
    test = test[test["label"] == 1]
    tu, ti = rec.user_vocab.get_indexer(test["user_id"]), rec.item_vocab.get_indexer(test["item_id"])
    ok = (tu >= 0) & (ti >= 0)
    truth = pd.Series(ti[ok]).groupby(tu[ok]).agg(set)

    rng = np.random.default_rng(SEED)
    users = truth.index.to_numpy()
    if len(users) > args.max_users:
        users = np.sort(rng.choice(users, args.max_users, replace=False))
    print(f"users with test-period activity: {len(users):,}")

    arms = np.array([assign_arm(int(rec.user_ids[u]), args.salt) for u in users])
    n_a, n_b = int((arms == "A").sum()), int((arms == "B").sum())
    srm = stats.binomtest(n_a, n_a + n_b, 0.5).pvalue
    print(f"assignment: A={n_a:,} B={n_b:,}  sample-ratio-mismatch p={srm:.3f} (want > 0.01)")

    rows = {"A": [], "B": []}
    for u in users:
        control, _ = retrieve(rec.index, rec.user_vecs[u:u + 1], [u], rec.seen, rec.n_candidates)
        rows["A"].append(user_metrics(control[0], truth[u], args.k))
        rows["B"].append(user_metrics(rec._rank_known_user(u, args.k), truth[u], args.k))
    per_user = {arm: pd.DataFrame(r) for arm, r in rows.items()}

    results = {"users": int(len(users)), "k": args.k, "arm_sizes": {"A": n_a, "B": n_b},
               "srm_p_value": round(float(srm), 4), "ab": {}, "paired": {}}
    for m in METRICS:
        # A/B: each user only counts in the arm they were assigned to
        a = per_user["A"][m].to_numpy()[arms == "A"]
        b = per_user["B"][m].to_numpy()[arms == "B"]
        results["ab"][m] = {
            "A": round(float(a.mean()), 4), "B": round(float(b.mean()), 4),
            "relative_lift_pct": round(float((b.mean() / a.mean() - 1) * 100), 1) if a.mean() else None,
            "diff_95ci": bootstrap_diff(a, b, rng),
            "p_value": round(float(stats.ttest_ind(b, a, equal_var=False).pvalue), 4),
        }
        # paired: every user scored under both policies
        pa, pb = per_user["A"][m].to_numpy(), per_user["B"][m].to_numpy()
        d = pb - pa
        boot = [rng.choice(d, len(d)).mean() for _ in range(2000)]
        results["paired"][m] = {
            "A": round(float(pa.mean()), 4), "B": round(float(pb.mean()), 4),
            "relative_lift_pct": round(float((pb.mean() / pa.mean() - 1) * 100), 1) if pa.mean() else None,
            "diff_95ci": [round(float(x), 4) for x in np.percentile(boot, [2.5, 97.5])],
            "p_value": round(float(stats.ttest_rel(pb, pa).pvalue), 4),
        }

    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "ab_test_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
