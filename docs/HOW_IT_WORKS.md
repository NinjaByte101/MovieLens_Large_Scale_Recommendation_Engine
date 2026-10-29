# How the recommender works

## The problem

MovieLens has 162k users and about 33k movies. Scoring every movie for a user with a big model at request time
is too slow. Showing everyone the most popular movies is fast but not personal. So the system works in two stages,
like Netflix or YouTube.

## Stage 0: prepare the data honestly

1. Remove users and movies with fewer than 5 ratings. A rating >= 4 means "liked".
2. Split by time: oldest 80% of ratings = train, next 10% = validation, last 10% = test. A random split would let the
   model see the future and look better than it really is.
3. Spark builds features such as a user's average rating or a movie's popularity, using **only earlier ratings** for
   each row. Otherwise a feature quietly contains the answer.

## Stage 1: retrieval with a two-tower model

Two small networks. The user tower turns a user (their id) into 64 numbers. The item tower turns a movie (its id and
genres) into 64 numbers. Both vectors are scaled to length 1, so their dot product is their cosine similarity.

Training shows the model pairs (user, movie they liked). For each pair, the other movies in the same batch act as
"not chosen" examples (in-batch softmax). The model is pushed to give the real pair a higher score than the others.
Popular movies appear in batches far more often, so a correction (logQ) lowers the credit they get for free.
Without it the model just learned "recommend popular things" and lost to the popularity baseline.

After training, every movie vector is stored. For a user, the best candidates are the movies whose vectors are
closest to the user's vector.

## FAISS: finding the nearest vectors fast

Comparing a user with all movies is cheap at 26k movies but not at millions. FAISS builds an index that finds
near-best neighbours without checking everything. We built three: Flat (exact), IVF (groups vectors in clusters),
HNSW (a layered graph). HNSW returned 97% of the exact top 100 in 0.076 ms. The retrieval step returns 200 movies the
user has not seen yet.

## Stage 2: ranking with a Deep & Cross Network

The 200 candidates are in order of vector similarity, which is a rough guess. The ranker looks at richer information
per candidate: the retrieval score and rank, how many movies the user rated and how generous they are, the movie's
popularity and average rating, and how well the movie's genres match what this user likes. The DCN has two parts:
cross layers that multiply features together to find combinations ("high popularity AND strong genre match"), and an
ordinary neural network for everything else. The output is a probability that the user will like the movie. Sort by it,
take the top 10.

### Why it took three tries (this is the story to tell)

- v1 and v2 learned "will this user like a movie they rated" and got a decent AUC, but they did **not** improve the
  recommendations. They were trained on movies people had already rated, while serving ranks movies they have not.
- v3 was trained on exactly the serving situation: take the retrieval candidates for validation-period users, label
  each by whether the user liked it in the next period, and learn from that. Using five folds split by user, NDCG@10
  went from 0.110 to 0.131.

## Cold start

A brand-new user has no vector. Options in the code: most popular movies; most popular within chosen genres; or the
average vector of movies they say they like, searched in FAISS. The ranker is skipped for these users. Only the
popularity versions are standard baselines; the taste-vector path is a heuristic that was not measured.

## Serving

Request -> Redis cache hit? return in 0.18 ms. Otherwise FAISS -> features -> ranker -> top N, about 18 ms, then
store in Redis with a TTL. The ranker can run locally or on NVIDIA Triton (exported to ONNX); if Triton is
unreachable the local copy answers. Weekly, an Airflow DAG retrains everything; a quality gate stops the
publish step if the new models are worse, and publishing clears the cache so nobody sees stale results.

## The A/B test

There was no live traffic, so it is a replay: for users in the test period, arm A shows the retrieval order and
arm B shows the ranker's order. A "click" is a movie the user really rated >= 4 in the test period. Users are assigned
to arms by a salted hash, checked for an even split, and differences are tested with bootstrap intervals and t-tests.
The ranker was better on every metric (NDCG@10 +29%). It is evidence, not proof of what live users would do.

## Questions you may get asked, and honest answers

- **Why two stages?** Speed. Cheap model narrows 33k to 200, expensive model orders those 200.
- **Why a time split?** To avoid leakage; production predicts the future.
- **Why did the first ranker fail?** Wrong training question (see above).
- **Is the A/B test real?** No, it is an offline replay.
- **Did you run Triton / Airflow for real?** Triton: tested against a stand-in server only. Airflow: loaded and
  validated the DAG, did not run the tasks end to end.
- **What would you do next?** Run Triton in Docker and measure; add live logging for a real A/B; evaluate the cold
  start paths; add sequence features (what they watched last); tune the candidate pool size since retrieval only
  contains about 30% of later likes.
