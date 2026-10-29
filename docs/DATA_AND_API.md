# Data and API reference

## 1. Data files

Everything below is produced by the scripts, in this order. Paths are the defaults.

| File | Made by | Columns / content |
|---|---|---|
| `data/processed/train.parquet`, `val.parquet`, `test.parquet` | `prepare_data.py` | `user_id, item_id, rating, label, ts` (label = 1 if rating >= 4; ts = unix seconds) |
| `data/processed/items.parquet` | `prepare_data.py` | `item_id, title, genres` (genres separated by `\|`) |
| `data/processed/split_info.json` | `prepare_data.py` | the two timestamp cut points and row counts |
| `data/features/item_features.parquet` | `build_features.py` | per movie: log popularity, average rating, positive rate (training period only) |
| `data/features/user_features.parquet` | `build_features.py` | per user: number of ratings, average rating, positive rate |
| `data/features/user_history.parquet` | `build_features.py` | per user: movies already seen (used to hide them from recommendations) |
| `data/features/ranker_train.parquet` | `build_features.py` | point-in-time features per training row (computed from earlier rows only) |
| `models/retrieval/{user,item}_embeddings.npy` | `train_two_tower.py` | 64-d, L2-normalised; row i belongs to `users.parquet` / `items.parquet` `*_idx` = i |
| `models/retrieval/metrics.json` | `train_two_tower.py` | recall/NDCG of the two-tower and of the popularity baseline |
| `models/faiss/items_{flat,ivf,hnsw}.index` | `build_faiss_index.py` | FAISS indexes over item vectors (inner product = cosine) |
| `models/faiss/faiss_benchmark.json` | `build_faiss_index.py` | overlap and latency per index |
| `models/ranking_v3/candidate_ranker.weights.h5` | `train_candidate_ranker.py` | DCN weights |
| `models/ranking_v3/candidate_ranker_config.json` | `train_candidate_ranker.py` | feature names, mean and std used for standardising |
| `models/ranking_v3/candidate_ranker_metrics.json` | `train_candidate_ranker.py` | cross-validated ranker vs retrieval numbers |
| `outputs/ab_test/`, `outputs/benchmark/` | `ab_test.py`, `benchmark.py` | result JSON/CSV |
| `model_repository/candidate_ranker/` | `export_triton.py` | `config.pbtxt` and `1/model.onnx` |

### Ranker features (the 9 numbers the DCN sees, in this order)

`retrieval_score, log_rank, user_n, user_avg_rating, user_pos_rate, item_log_pop, item_avg_rating, item_pos_rate, genre_affinity`

plus the movie's genres (a multi-hot vector inside the model). `genre_affinity` is how much the user's liked
movies overlap with this movie's genres. There are deliberately no user or movie ID embeddings in the ranker.

### Leakage rules

- Split is by time, not at random. Validation and test are later than training.
- Features for a training row use only rows with an earlier timestamp (`rowsBetween(unboundedPreceding, -1)`).
- Positive label threshold is rating >= 4.
- The ranker is trained on validation-period users with labels from the following period, never on test rows.

## 2. Python API

```python
from recommender import Recommender      # src/serving/recommender.py

rec = Recommender(data_dir, features_dir, retrieval_dir, faiss_dir, ranking_dir,
                  cache=None, triton_url=None)          # see build_from_args for the exact wiring
result = rec.recommend(user_id=42, k=10)                # known user
result = rec.recommend(user_id=-1, k=10, genres=["Comedy", "Drama"])
result = rec.recommend(user_id=-1, k=10, liked_item_ids=[1, 260, 296])
```

`recommend()` returns:

```json
{"source": "ranker", "cached": false, "latency_ms": 17.6,
 "items": [{"item_id": 318, "title": "Shawshank Redemption, The (1994)"}]}
```

`source` tells you which path answered:

| source | When | Evaluated offline? |
|---|---|---|
| `ranker` | user has training history: FAISS top 200 unseen movies, re-scored by the DCN | yes |
| `taste_vector` | new user who gave liked movie ids: FAISS around the average of those movies' vectors | no |
| `popular_by_genre` | new user who gave genres | no |
| `popular` | nothing known (or none of the inputs matched) | baseline only |

Cached results are stored under the Redis key `rec:v1:{user_id}:{k}` with a TTL (default 3600 s). Any Redis
error is treated as a cache miss, so Redis being down never breaks a request.

## 3. Command line

```bash
python src/serving/recommender.py --user-id 42 -k 10
python src/serving/recommender.py --genres "Comedy|Drama"
python src/serving/recommender.py --liked 1,260,296
python src/serving/recommender.py --demo --redis localhost:6379
python src/serving/recommender.py --user-id 42 --triton localhost:8000   # Triton, local fallback if down
```

## 4. Triton model contract

Model name `candidate_ranker`, platform `onnxruntime_onnx`, HTTP port 8000.

| Tensor | Direction | Type | Shape | Meaning |
|---|---|---|---|---|
| `num` | input | FP32 | `[-1, 9]` | the 9 raw (not standardised) features above, one row per candidate |
| `item` | input | INT32 | `[-1]` | movie row index (picks the genre vector inside the model) |
| `score` | output | FP32 | `[-1]` | probability the user likes this candidate |

Standardisation and the genre table are baked into the ONNX graph, so clients send raw features.
`export_triton.py` writes `config.pbtxt` from the real ONNX input/output names and checks the ONNX scores
against the Keras model (the difference should be below 1e-4).

## 5. Airflow DAG

`airflow/dags/recsys_retrain_dag.py`, weekly, tasks in order:
`prepare_data -> build_features -> train_retrieval -> build_faiss -> train_ranker -> quality_gate -> export_triton -> publish`.
Set `RECSYS_ROOT` to the project folder. `quality_gate` fails the run (so `publish` never happens) if retrieval
does not beat popularity or the ranker does not beat retrieval order. `publish` swaps the model repository
atomically (keeping the previous one as `.old`) and clears the Redis cache.
