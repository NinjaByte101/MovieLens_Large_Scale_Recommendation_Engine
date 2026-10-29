# Results

Every number below was printed by the scripts in this repository on real MovieLens 25M data (Google Colab),
except where a row says otherwise. Validation and test come from later time periods than training.

## 1. Data and split

| Item | Value |
|---|---|
| Raw interactions | 25,000,095 |
| After cleaning (users and movies with at least 5 ratings; likes = rating >= 4) | 24,945,870 (162,541 users, 32,720 movies) |
| Train / validation / test rows | 19,956,696 / 2,494,587 / 2,494,587 |
| Positive rate in train | 49.9% |
| Test rows from users never seen in train | 14,489 users (2,169,895 rows) |
| Feature tables | 26,228 movies, 137,734 users, 137,533 histories, 19,956,696 ranker rows |

The time split is why cold-start matters here: a large share of later-period users never appear in training.

## 2. Retrieval (two-tower) vs the popularity baseline

5,063 validation users who have training history. Both rankings hide movies the user already interacted with.
Training: 9,963,670 (user, liked movie) pairs, 137,533 users, 26,228 movies, 64-d vectors, 3 epochs, batch 4,096
(about 27 min per epoch on a Colab CPU). Loss per epoch: 31,235, 30,748, 30,663 (flattening).

| Metric | Popularity | Two-tower | Relative gain |
|---|---|---|---|
| Recall@10 | 0.0321 | 0.0468 | +46% |
| Recall@50 | 0.0975 | 0.1521 | +56% |
| Recall@100 | 0.1501 | 0.2372 | +58% |
| NDCG@10 | 0.0869 | 0.1099 | +26% |
| NDCG@50 | 0.0900 | 0.1253 | +39% |
| NDCG@100 | 0.1050 | 0.1516 | +44% |

First version for context: an ID-only two-tower (no genres, no popularity correction) scored **below** popularity
(Recall@10 0.0260 vs 0.0282, NDCG@10 0.0520 vs 0.0732). That run used 985 users and different filters, so compare the
direction, not the absolute numbers.

## 3. FAISS

26,228 movie vectors (64-d), 2,000 query users. "Overlap" is the share of the exact top 100 that the index returns.

| Index | Overlap | p50 | p95 | Batched queries/s |
|---|---|---|---|---|
| Flat (exact) | 1.000 | 0.793 ms | 0.905 ms | 11,804 |
| IVF (nprobe 8) | 0.961 | 0.049 ms | 0.065 ms | 30,917 |
| HNSW (efSearch 64) | 0.970 | 0.076 ms | 0.105 ms | 21,748 |

At 26k movies exact search is already under 1 ms, so the speed-up matters mainly for much larger catalogs.

## 4. Ranker: how it got to the final version

All rows: 5,000+ validation users, 200 FAISS candidates each (movies not yet seen). Retrieval order is the baseline.
The candidate pool contains about 30.3% of each user's later likes, which is the ceiling for any re-ordering.

| Version | What it learned | Validation AUC | NDCG@10 (ranker) | NDCG@10 (retrieval) |
|---|---|---|---|---|
| v1 pointwise | "did they like what they rated" (all rows) | 0.58 then 0.48 (collapsed) | 0.0734 | 0.1100 |
| v2 pointwise + fixes | same task, recent rows, ID dropout, known users only | 0.78 | 0.0691 | 0.1100 |
| **v3 candidate-trained** | "will they like this retrieved candidate next" | 0.70 (out-of-fold) | **0.1307** | 0.1102 |

Why v1 and v2 did not help: they were trained on movies users had already rated, but the ranker re-orders movies
the user has not rated. A good AUC on the first question did not answer the second. v3 trains on exactly what serving
sees: retrieval candidates labelled with what the user liked in the next period. (This explanation fits every number
but was not tested separately.)

### v3 in detail (5,063 users, 1,010,218 candidate rows, 3.9% positives, 5-fold cross-validation by user)

| Metric | Retrieval order | Ranker | Blend of both |
|---|---|---|---|
| Recall@10 | 0.0473 | **0.0574** | 0.0521 |
| NDCG@10 | 0.1102 | **0.1307** | 0.1210 |
| Recall@50 | 0.1511 | **0.1657** | 0.1638 |
| NDCG@50 | 0.1248 | **0.1397** | 0.1348 |

## 5. Replay A/B test on the held-out test period

3,811 users with test-period likes. Arm A = retrieval order, arm B = ranker. Assignment by salted hash,
split 1,904 / 1,907, sample-ratio check p = 0.974. "Clicks" are movies the user really liked in the test period.

Paired comparison (every user scored under both policies):

| Metric | A (retrieval) | B (ranker) | Lift | 95% interval of difference | p |
|---|---|---|---|---|---|
| Hit rate@10 | 0.3647 | 0.4372 | +19.9% | 0.059 to 0.086 | < 0.001 |
| CTR@10 | 0.0740 | 0.0933 | +26.0% | 0.016 to 0.022 | < 0.001 |
| Recall@10 | 0.0455 | 0.0601 | +32.3% | 0.011 to 0.018 | < 0.001 |
| NDCG@10 | 0.0885 | 0.1142 | +29.0% | 0.022 to 0.030 | < 0.001 |

Split-arm comparison (each user counted only in the arm they were assigned to; fewer data, wider intervals):

| Metric | A | B | Lift | p |
|---|---|---|---|---|
| Hit rate@10 | 0.3687 | 0.4316 | +17.1% | 0.0001 |
| CTR@10 | 0.0753 | 0.0908 | +20.7% | 0.0008 |
| Recall@10 | 0.0431 | 0.0609 | +41.3% | < 0.0001 |
| NDCG@10 | 0.0888 | 0.1122 | +26.3% | < 0.0001 |

CTR here = liked movies divided by recommended movies. Limits: replay data only contains movies users chose to
rate, there is no position-bias model, and the ranker's training used validation-period labels of users who also
appear in the test period (the ranker has no user ID, and the test labels come from a later period).

## 6. Serving latency

Single thread, 500 requests after warm-up, Redis on the same machine (no network hop).

| Path | p50 | p95 | p99 | Requests/s |
|---|---|---|---|---|
| Known user, uncached (FAISS + features + ranker) | 17.61 ms | 21.52 ms | 27.21 ms | 55 |
| Known user, Redis hit | 0.18 ms | 0.22 ms | 0.25 ms | 5,419 |
| New user, most popular | 0.01 ms | 0.01 ms | 0.02 ms | 122,008 |
| New user, genre fallback | 0.19 ms | 0.28 ms | 0.31 ms | 4,950 |
| New user, liked-movies vector | 0.23 ms | 0.32 ms | 0.41 ms | 4,195 |

The cache hit rate of 100% in that benchmark is by construction (the second pass reads what the first pass wrote),
not a measured production hit rate.

## 7. What was checked but not measured on real services

| Item | Status |
|---|---|
| ONNX export vs Keras model | Matches to 8.9e-8 on the real trained ranker (2,048 rows, run in Colab) |
| Triton client | Checked against a stand-in server speaking Triton's request format: identical top-10 for 40/40 users. No real Triton server was run |
| Fallback when Triton is down | Tested: recommender serves from the local ranker |
| Airflow DAG | Loads in Airflow 3.x with no import errors, 8 tasks in the right order. Tasks were not executed end to end |
| Quality gate | Passes the v3 numbers, blocks the v2 numbers |
| Unit tests | `pytest -q` |
