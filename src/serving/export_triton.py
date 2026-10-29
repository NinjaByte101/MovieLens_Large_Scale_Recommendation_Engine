"""Step 7c: export the ranker to ONNX and lay out a Triton model repository.

Why ONNX and not a TensorFlow SavedModel: NVIDIA deprecated Triton's TensorFlow backend in
release 25.03 (25.02 is the last container that includes it). ONNX Runtime is supported by
current Triton releases.

The exported graph takes RAW features. The standardisation (x - mean) / std is baked into the
model, so the client cannot forget it and training and serving cannot drift apart.

The script also checks the ONNX model against the original Keras model on random inputs and
stops if they disagree.

Output layout:
    <out>/candidate_ranker/config.pbtxt
    <out>/candidate_ranker/1/model.onnx

Run:
    python src/serving/export_triton.py --artifacts-root outputs --data processed \
        --retrieval retrieval --ranking ranking_v3 --out model_repository
"""
import os

# TFRS needs Keras 2. Must be set before tensorflow is imported.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import tensorflow as tf
import tf2onnx

sys.path.append(str(Path(__file__).resolve().parents[1] / "ranking"))
from train_candidate_ranker import CandidateRanker, FEATURES  # noqa: E402
from train_ranker import genre_table  # noqa: E402

MODEL_NAME = "candidate_ranker"
ONNX_TYPES = {1: "TYPE_FP32", 6: "TYPE_INT32", 7: "TYPE_INT64"}


def dims_of(value_info):
    return [d.dim_value if d.dim_value > 0 else -1 for d in value_info.type.tensor_type.shape.dim]


def write_config(proto, path):
    """Triton config generated from the real ONNX input/output names, so they cannot disagree."""
    def block(infos):
        rows = []
        for v in infos:
            dtype = ONNX_TYPES[v.type.tensor_type.elem_type]
            rows.append(f'  {{ name: "{v.name}" data_type: {dtype} dims: {dims_of(v)} }}')
        return ",\n".join(rows)

    path.write_text(
        f'name: "{MODEL_NAME}"\n'
        'platform: "onnxruntime_onnx"\n'
        "max_batch_size: 0\n"                      # the batch dimension is part of dims (-1)
        f"input [\n{block(proto.graph.input)}\n]\n"
        f"output [\n{block(proto.graph.output)}\n]\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-root", default=".")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--retrieval", default="models/retrieval")
    ap.add_argument("--ranking", default="models/ranking_v3")
    ap.add_argument("--out", default="model_repository")
    args = ap.parse_args()

    root = Path(args.artifacts_root)
    cfg = json.loads((root / args.ranking / "candidate_ranker_config.json").read_text())
    assert cfg["features"] == FEATURES, "ranker was trained with a different feature list"
    mean, std = np.array(cfg["mean"], dtype="float32"), np.array(cfg["std"], dtype="float32")

    item_ids = pd.read_parquet(root / args.retrieval / "items.parquet")["item_id"].to_numpy()
    genres = genre_table(pd.read_parquet(root / args.data / "items.parquet"), item_ids)[1:]

    model = CandidateRanker(genres)
    model({"num": np.zeros((1, len(FEATURES)), "float32"), "item": np.zeros(1, "int32")})   # build
    model.load_weights(str(root / args.ranking / "candidate_ranker.weights.h5"))
    # an eager constant would be exported as a model INPUT; a variable gets frozen into the graph
    model.genres = tf.Variable(genres, trainable=False)

    @tf.function(input_signature=[tf.TensorSpec([None, len(FEATURES)], tf.float32, name="num"),
                                  tf.TensorSpec([None], tf.int32, name="item")])
    def serve(num, item):
        scaled = (num - mean) / std                      # standardisation lives inside the model
        return {"score": model({"num": scaled, "item": item}, training=False)}

    proto, _ = tf2onnx.convert.from_function(serve, input_signature=serve.input_signature, opset=17)
    onnx.checker.check_model(proto)

    version_dir = root / args.out / MODEL_NAME / "1"
    version_dir.mkdir(parents=True, exist_ok=True)
    onnx.save(proto, str(version_dir / "model.onnx"))
    write_config(proto, root / args.out / MODEL_NAME / "config.pbtxt")

    # parity check: realistic raw features (mean + std * noise), random valid movies
    rng = np.random.default_rng(0)
    raw = (mean + std * rng.normal(size=(2048, len(FEATURES)))).astype("float32")
    items = rng.integers(0, len(item_ids), size=2048).astype("int32")
    expected = model({"num": (raw - mean) / std, "item": items}, training=False).numpy()
    session = ort.InferenceSession(str(version_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    got = session.run(None, {"num": raw, "item": items})[0]
    worst = float(np.abs(expected - got).max())
    print(f"inputs: {[(i.name, dims_of(i)) for i in proto.graph.input]}  outputs: {[o.name for o in proto.graph.output]}")
    print(f"max |keras - onnx| over 2048 rows: {worst:.2e}")
    assert worst < 1e-4, "ONNX model does not match the Keras model"
    print(f"model repository written to {root / args.out}")


if __name__ == "__main__":
    main()
