import polars as pl
from datasets import load_dataset
import pyarrow as pa

def load_qilin_subsets():
    """
    Loads the required Qilin subsets directly into Polars DataFrames 
    via PyArrow for zero-copy memory efficiency.
    """
    print("Downloading/Loading Qilin subsets from HuggingFace...")
    
    # Load raw datasets
    search_train = load_dataset("THUIR/Qilin", "search_train", split="train")
    dqa = load_dataset("THUIR/Qilin", "dqa", split="train")
    
    # The supervisor's mandate: Locate the raw Chinese text
    # Assuming the subset is named "note" (verify the exact subset name on HF)
    notes = load_dataset("THUIR/Qilin", "notes", split="train") 
    
    # Convert directly to Polars via Arrow
    df_train = pl.from_arrow(search_train.data.table)
    df_dqa = pl.from_arrow(dqa.data.table)
    df_notes = pl.from_arrow(notes.data.table)
    
    return df_train, df_dqa, df_notes

def flatten_and_merge(df, df_notes, is_dqa=False):
    """
    Explodes the nested session logs and joins the raw note text using Polars.
    Strictly extracts ONLY note_title and note_content to ensure memory safety 
    and prevent feature leakage into the Relevance Tower.
    """
    col = 'search_result_details_with_idx'
    
    print(f"Flattening {len(df)} sessions to impression level...")
    
    # 1. Explode the lists into rows and drop nulls
    exploded = df.explode(col).drop_nulls(subset=[col])
    
    # 2. Unnest the dictionaries into distinct columns
    # Polars native struct unnesting is highly optimized for this architecture
    unnested = exploded.unnest(col)
    
    # 3. Retain relevant session/query columns 
    keep_cols = ['session_idx', 'search_idx', 'query']
    if is_dqa and 'dqa_output' in unnested.columns:
        keep_cols.append('dqa_output')
        
    # Combine kept columns with the unnested impression data
    impression_cols = [c for c in unnested.columns if c not in df.columns]
    final_cols = keep_cols + impression_cols
    base_imp = unnested.select(final_cols)
    
    # 4. The Relational Merge: Join raw text based on note_idx
    print("Merging raw item text (note_title and note_content)...")
    
    # Ensure note_idx types match before joining (cast to Int64)
    base_imp = base_imp.with_columns(pl.col("note_idx").cast(pl.Int64))
    df_notes = df_notes.with_columns(pl.col("note_idx").cast(pl.Int64))
    
    # Perform the Left Join - strictly select ONLY text columns to save memory
    merged_imp = base_imp.join(
        df_notes.select(['note_idx', 'note_title', 'note_content']), 
        on="note_idx", 
        how="left"
    )
    
    # 5. Add Position/Layout Bias Features
    # Odd positions map to the left column in the two-column interface
    merged_imp = merged_imp.with_columns([
        (pl.col('position') % 2 != 0).alias('is_left_column')
    ])
    
    return merged_imp

if __name__ == "__main__":
    # 1. Load Data
    df_train, df_dqa, df_notes = load_qilin_subsets()
    
    # 2. Filter DQA for valid outputs (failed DQA module = -1)
    df_dqa_valid = df_dqa.filter(pl.col("dqa_output") != "-1")
    
    # 3. Process Train & DQA Data
    print("\n--- Processing Train Data ---")
    train_impressions = flatten_and_merge(df_train, df_notes, is_dqa=False)
    
    print("\n--- Processing DQA Data ---")
    dqa_impressions = flatten_and_merge(df_dqa_valid, df_notes, is_dqa=True)
    
    # 4. Filter top-level ad insertions (position 0)
    train_impressions = train_impressions.filter(pl.col("position") > 0)
    dqa_impressions = dqa_impressions.filter(pl.col("position") > 0)
    
    print(f"\nFinal Train Impressions: {train_impressions.height}")
    print(f"Final DQA Impressions: {dqa_impressions.height}")
    
    # Save to disk as Parquet (highly optimized for M4 / Polars)
    # This prepares the data perfectly for Phase 2's Embedding Generation
    train_impressions.write_parquet("train_merged_text.parquet")
    dqa_impressions.write_parquet("dqa_merged_text.parquet")
    print("\nPhase 1 Complete. Saved as Parquet files.")      