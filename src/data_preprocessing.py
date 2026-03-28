import pandas as pd
import numpy as np
from datasets import load_dataset

def flatten_impressions(df, col='search_result_details_with_idx'):
    """
    Explodes the nested Qilin session logs into impression-level rows.
    """
    # Explode lists into rows and drop missing values
    exp = df.dropna(subset=[col]).explode(col).reset_index(drop=True)
    # Expand dicts into columns and merge with session/search IDs
    return exp[['session_idx', 'search_idx']].join(pd.json_normalize(exp[col]))

def load_raw_qilin():
    """
    Helper function to download/load the raw Qilin data from HuggingFace.
    """
    print("Downloading/Loading Qilin from HuggingFace...")
    dqa = load_dataset("THUIR/Qilin", "dqa", split="train").to_pandas()
    search_train = load_dataset("THUIR/Qilin", "search_train", split="train").to_pandas()
    search_test = load_dataset("THUIR/Qilin", "search_test", split="train").to_pandas()
    
    return search_train, search_test, dqa

def get_behavioral_datasets():
    """
    Prepares the merged baseline and DQA datasets specifically for SQ1 and SQ2.
    WARNING: Do not use the baseline output of this function for model training (SQ3) 
    as it contains the test set (data leakage).
    """
    search_train, search_test, dqa_full = load_raw_qilin()
    
    print("Partitioning DQA and merging behavioral baseline...")
    dqa_valid = dqa_full[dqa_full['dqa_output'] != '-1'].copy()
    dqa_failed = dqa_full[dqa_full['dqa_output'] == '-1'].copy()
    
    # Three-way merge for statistical analysis
    standard_baseline = pd.concat([search_train, search_test, dqa_failed], ignore_index=True)
    
    print("Flattening to impression level...")
    base_imp = flatten_impressions(standard_baseline)
    dqa_imp = flatten_impressions(dqa_valid)
    
    # Filter out top-level insertions (Position 0) for the baseline
    base_imp = base_imp[base_imp['position'] > 0].reset_index(drop=True)
    
    # Add column placement feature
    base_imp['is_left_column'] = base_imp['position'] % 2 != 0
    dqa_imp['is_left_column'] = dqa_imp['position'] % 2 != 0
    
    print(f"SQ1/SQ2 Baseline Impressions: {len(base_imp)}")
    print(f"SQ1/SQ2 DQA-Present Impressions: {len(dqa_imp)}")
    
    return base_imp, dqa_imp

def get_modeling_datasets():
    """
    Prepares the strictly separated train, test, and DQA datasets for SQ3.
    This ensures no data leakage occurs between train and test splits.
    """
    search_train, search_test, dqa_full = load_raw_qilin()
    
    dqa_valid = dqa_full[dqa_full['dqa_output'] != '-1'].copy()
    
    print("Flattening modeling splits to impression level...")
    train_imp = flatten_impressions(search_train)
    test_imp = flatten_impressions(search_test)
    dqa_imp = flatten_impressions(dqa_valid)
    
    # Add column placement feature
    for df in [train_imp, test_imp, dqa_imp]:
        df['is_left_column'] = df['position'] % 2 != 0
        
    return train_imp, test_imp, dqa_imp