"""Demo UI for the screen recording: inspect a known user, or describe a brand-new one.

Known users go through the full path (FAISS -> DCN ranker), new users through the cold-start
fallbacks. The page shows what the system knows about the user next to what it recommends.

Run:
    python app/demo_app.py --artifacts-root outputs --data processed --features features \
        --retrieval retrieval --faiss faiss --ranking ranking_v3 [--redis localhost:6379] [--share]

In Colab use --share to get a public link, since the script runs as a subprocess.
"""
import os

# TFRS needs Keras 2. Must be set before tensorflow is imported.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import argparse
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "serving"))
from recommender import build_from_args  # noqa: E402

SOURCE_TEXT = {
    "ranker": "known user: FAISS candidates re-ranked by the DCN ranker",
    "taste_vector": "new user: average of the liked movies' vectors, searched in FAISS",
    "popular_by_genre": "new user: most-liked movies inside the chosen genres",
    "popular": "new user: most-liked movies overall",
}
COLUMNS = ["#", "Movie", "Genres"]


class Demo:
    def __init__(self, rec, features_dir):
        self.rec = rec
        self.history = pd.read_parquet(Path(features_dir) / "user_history.parquet").set_index("user_id")["history"]
        self.genre_vocab = sorted({g for gs in rec.genre_names for g in gs if g and g != "(no genres listed)"})
        top = rec.popular_order[:1500]
        self.title_to_id = {str(rec.titles[i]): int(rec.item_ids[i]) for i in top}
        self.popular_titles = list(self.title_to_id)

    def _row_info(self, item_id):
        idx = self.rec.item_vocab.get_indexer([item_id])[0]
        if idx < 0:
            return None
        return str(self.rec.titles[idx]), ", ".join(g for g in self.rec.genre_names[idx] if g)

    def _table(self, items):
        rows = []
        for rank, item in enumerate(items, start=1):
            info = self._row_info(item["item_id"])
            rows.append([rank, item["title"], info[1] if info else ""])
        return pd.DataFrame(rows, columns=COLUMNS)

    def known_user(self, user_id, k):
        empty = pd.DataFrame(columns=COLUMNS)
        if user_id is None or int(user_id) not in self.rec.user_vocab:
            return "That user id has no training history. Try the **New user** tab.", empty
        uid, k = int(user_id), int(k)
        u = self.rec.user_vocab.get_loc(uid)

        profile = self.rec.profiles[u]
        favourite = ", ".join(f"{self.genre_vocab[i]} ({profile[i]:.0%})"
                              for i in np.argsort(-profile)[:3] if profile[i] > 0)
        liked = []
        for item_id in list(self.history.get(uid, []))[:8]:
            info = self._row_info(int(item_id))
            if info:
                liked.append(info[0])

        result = self.rec.recommend(uid, k)
        text = (f"**User {uid}**\n\n"
                f"**Favourite genres:** {favourite or 'n/a'}\n\n"
                f"**Recently liked:** {'; '.join(liked) or 'n/a'}\n\n"
                f"_{SOURCE_TEXT[result['source']]} | cached: {result['cached']} | {result['latency_ms']} ms_")
        return text, self._table(result["items"])

    def new_user(self, genres, liked_titles, k):
        liked_ids = [self.title_to_id[t] for t in (liked_titles or []) if t in self.title_to_id]
        result = self.rec.recommend(-1, int(k), genres=genres or None, liked_item_ids=liked_ids or None)
        text = f"_{SOURCE_TEXT[result['source']]} | {result['latency_ms']} ms_"
        return text, self._table(result["items"])

    def random_user(self):
        return int(np.random.choice(self.rec.user_ids))


def build_ui(demo):
    with gr.Blocks(title="Movie recommender demo") as ui:
        gr.Markdown("# Movie recommender demo\n"
                    "Two-stage recommender on MovieLens 25M: FAISS retrieval, then a DCN ranker. "
                    "New users get cold-start fallbacks.")
        with gr.Tab("Existing user"):
            with gr.Row():
                uid = gr.Number(label="User id", value=int(demo.rec.user_ids[0]), precision=0)
                k = gr.Slider(5, 20, value=10, step=1, label="How many movies")
            with gr.Row():
                go = gr.Button("Recommend", variant="primary")
                rnd = gr.Button("Random user")
            summary = gr.Markdown()
            table = gr.Dataframe(headers=COLUMNS, interactive=False)
            go.click(demo.known_user, [uid, k], [summary, table])
            rnd.click(demo.random_user, None, uid).then(demo.known_user, [uid, k], [summary, table])

        with gr.Tab("New user"):
            genres = gr.CheckboxGroup(choices=demo.genre_vocab, label="Genres they like")
            liked = gr.Dropdown(choices=demo.popular_titles, multiselect=True,
                                label="Or movies they like (pick a few; this takes priority)")
            k2 = gr.Slider(5, 20, value=10, step=1, label="How many movies")
            go2 = gr.Button("Recommend", variant="primary")
            summary2 = gr.Markdown()
            table2 = gr.Dataframe(headers=COLUMNS, interactive=False)
            go2.click(demo.new_user, [genres, liked, k2], [summary2, table2])
    return ui


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-root", default=".")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--retrieval", default="models/retrieval")
    ap.add_argument("--faiss", default="models/faiss")
    ap.add_argument("--ranking", default="models/ranking_v3")
    ap.add_argument("--redis", default="")
    ap.add_argument("--ttl", type=int, default=3600)
    ap.add_argument("--triton", default="")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="public link (needed in Colab)")
    args = ap.parse_args()

    rec = build_from_args(args)
    demo = Demo(rec, Path(args.artifacts_root) / args.features)
    build_ui(demo).launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
