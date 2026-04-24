import polars as pl
import torch
from sentence_transformers import SentenceTransformer
import pickle
from tqdm import tqdm

def generate_embeddings():
    # 1. Setup Device (Optimized for Apple M4)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Load the data
    df_train = pl.read_parquet("../data/train_merged_text.parquet")
    df_dqa = pl.read_parquet("../data/dqa_merged_text.parquet")

    # 3. Prepare Unique Text Sets (Efficiency Hack)
    # Combine title and content for the Item Embedding
    print("Preparing unique text sets...")
    
    # Extract unique Queries
    unique_queries = pl.concat([
        df_train.select("query"),
        df_dqa.select("query")
    ]).unique().to_series().to_list()

    # Extract unique DQA Outputs
    unique_dqa = df_dqa.select("dqa_output").unique().to_series().to_list()

    # Extract unique Note Content (Concatenating title and content)
    unique_notes = pl.concat([
        df_train.select(["note_idx", "note_title", "note_content"]),
        df_dqa.select(["note_idx", "note_title", "note_content"])
    ]).unique(subset=["note_idx"])
    
    # Create the combined string for the Item Tower
    note_texts = (unique_notes["note_title"] + " " + unique_notes["note_content"]).to_list()
    note_indices = unique_notes["note_idx"].to_list()

    # 4. Load the Sentence-Transformer
    model = SentenceTransformer('shibing624/text2vec-base-chinese', device=device)

    # 5. Generate Embeddings
    print(f"Vectorizing {len(unique_queries)} unique queries...")
    query_embeddings = model.encode(unique_queries, batch_size=128, show_progress_bar=True, convert_to_numpy=True)
    query_map = dict(zip(unique_queries, query_embeddings))

    print(f"Vectorizing {len(unique_dqa)} unique DQA responses...")
    dqa_embeddings = model.encode(unique_dqa, batch_size=128, show_progress_bar=True, convert_to_numpy=True)
    dqa_map = dict(zip(unique_dqa, dqa_embeddings))

    print(f"Vectorizing {len(note_texts)} unique items...")
    item_embeddings = model.encode(note_texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True)
    item_map = dict(zip(note_indices, item_embeddings))

    # 6. Save to Disk
    print("Saving embedding maps to disk...")
    with open("embeddings_map.pkl", "wb") as f:
        pickle.dump({
            "query": query_map,
            "dqa": dqa_map,
            "item": item_map
        }, f)

    print("Phase 2 Complete. Embeddings stored in embeddings_map.pkl")

if __name__ == "__main__":
    generate_embeddings()