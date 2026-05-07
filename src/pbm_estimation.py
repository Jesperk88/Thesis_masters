import pandas as pd
import numpy as np

def estimate_pbm_examination(imp_df, max_position=10):
    """
    Estimates the examination probability P(E) for each position using 
    the PBM framework. Assumes relevance is distributed evenly across 
    large sample sizes, allowing CTR to proxy relative examination bias.
    
    Parameters:
    imp_df (pd.DataFrame): The impression-level dataframe.
    max_position (int): The maximum rank to analyze (default 10).
    
    Returns:
    pd.DataFrame: A dataframe containing position, CTR, and estimated P(E).
    """
    # Filter to top N positions
    df_filtered = imp_df[imp_df['position'] <= max_position].copy()
    
    # Calculate empirical CTR per position
    pos_stats = df_filtered.groupby('position').agg(
        impressions=('click', 'count'),
        clicks=('click', 'sum')
    ).reset_index()
    
    # Handle edge cases where impressions might be 0
    pos_stats['empirical_ctr'] = np.where(
        pos_stats['impressions'] > 0, 
        pos_stats['clicks'] / pos_stats['impressions'], 
        0
    )
    
    # Normalize against Rank 1 (or max CTR) to isolate relative examination bias.
    # Because P(C) = P(E) * P(R), if P(R) is constant, P(E) is proportional to P(C).
    max_ctr = pos_stats['empirical_ctr'].max()
    pos_stats['examination_prob'] = pos_stats['empirical_ctr'] / max_ctr if max_ctr > 0 else 0
    
    return pos_stats[['position', 'empirical_ctr', 'examination_prob']]