"""
prepare_search_test.py
----------------------
Creates data/test_merged_text.parquet for Qilin search_test and appends only
missing search_test query/item embeddings to data/embeddings_map.pkl.

This intentionally reuses data_merge.flatten_and_merge so search_test receives
the same impression-level preprocessing as search_train. It does not regenerate
or overwrite existing search_train/DQA embeddings.
"""

import pickle
from pathlib import Path

import numpy as np
import polars as pl
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from data_merge import flatten_and_merge

DATA_DIR = Path("data")
EMB_PATH = DATA_DIR / "embeddings_map.pkl"
TEST_OUT = DATA_DIR / "test_merged_text.parquet"
MODEL_NAME = "shibing624/text2vec-base-chinese"
EMBEDDING_DIM = 768


def encode_missing(model, texts):
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32, copy=False)


def load_existing_embedding_map():
    if not EMB_PATH.exists():
        raise FileNotFoundError(
            f"{EMB_PATH} does not exist. Run the original embedding generation "
            "for search_train/DQA first, then run this script to append search_test."
        )

    print(f"Loading existing embedding map from {EMB_PATH}...")
    with open(EMB_PATH, "rb") as f:
        emb_map = pickle.load(f)

    required_keys = {"query", "item", "dqa"}
    missing_keys = required_keys - set(emb_map)
    if missing_keys:
        raise KeyError(f"{EMB_PATH} is missing required maps: {sorted(missing_keys)}")

    return emb_map


def main():
    DATA_DIR.mkdir(exist_ok=True)

    print("Loading search_test and notes from HuggingFace...")
    search_test = load_dataset("THUIR/Qilin", "search_test", split="train")
    notes = load_dataset("THUIR/Qilin", "notes", split="train")

    df_test = pl.from_arrow(search_test.data.table)
    df_notes = pl.from_arrow(notes.data.table)

    test_imp = flatten_and_merge(df_test, df_notes, is_dqa=False)
    test_imp = test_imp.filter(pl.col("position") > 0)
    test_imp.write_parquet(TEST_OUT)

    print(f"Saved {TEST_OUT}: {test_imp.height} impressions")

    emb_map = load_existing_embedding_map()

    query_map = emb_map["query"]
    item_map = emb_map["item"]

    unique_queries = test_imp.select("query").unique().to_series().to_list()

    unique_notes = (
        test_imp
        .select(["note_idx", "note_title", "note_content"])
        .unique(subset=["note_idx"])
    )

    note_indices = unique_notes["note_idx"].to_list()
    # Same title/content text as text_embeddings.py, with nulls made explicit.
    note_texts = (
        unique_notes["note_title"].fill_null("")
        + " "
        + unique_notes["note_content"].fill_null("")
    ).to_list()

    missing_queries = [q for q in unique_queries if q not in query_map]

    missing_items = [
        (idx, text)
        for idx, text in zip(note_indices, note_texts)
        if idx not in item_map
    ]

    print(f"Missing query embeddings: {len(missing_queries)}")
    print(f"Missing item embeddings: {len(missing_items)}")

    if not missing_queries and not missing_items:
        print("No missing embeddings. Existing embedding map already covers search_test.")
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Encoding missing embeddings on: {device}")

    model = SentenceTransformer(MODEL_NAME, device=device)

    if missing_queries:
        print("Encoding missing search_test queries...")
        query_embs = encode_missing(model, missing_queries)
        query_map.update(dict(zip(missing_queries, query_embs)))

    if missing_items:
        print("Encoding missing search_test items...")
        missing_item_indices = [x[0] for x in missing_items]
        missing_item_texts = [x[1] for x in missing_items]

        item_embs = encode_missing(model, missing_item_texts)
        item_map.update(dict(zip(missing_item_indices, item_embs)))

    emb_map["query"] = query_map
    emb_map["item"] = item_map

    tmp_path = EMB_PATH.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(emb_map, f)
    tmp_path.replace(EMB_PATH)

    print(f"Updated {EMB_PATH} with search_test embeddings.")


if __name__ == "__main__":
    main()
