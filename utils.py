"""Utility functions for data processing and analysis in the pairs trading pipeline.

This module provides functions to clean price data, split datasets into training, testing,
and validation subsets, and display combined summaries of cointegration tests.
"""

import numpy as np
import pandas as pd

def clean_prices(data):
    data = data.copy()
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.drop(columns=['Price'])
    # Convert objects to numeric values
    for col in data.columns:
        if col != 'Date':
            data[col] = pd.to_numeric(data[col], errors='coerce')

    data = data.set_index('Date')
    data = data.dropna()

    data.to_csv('data/prices.csv')
    return data


def split_dfs(data: pd.DataFrame, train: int, test: int, validation: int) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a time-ordered DataFrame into training, testing, and validation sets.

    Parameters
    ----------
    data : pd.DataFrame
        The complete, time-sorted DataFrame to be split.
    train : int
        Percentage of data to allocate to the training set (e.g., 60).
    test : int
        Percentage of data to allocate to the test set (e.g., 20).
    validation : int
        Percentage of data to allocate to the validation set (e.g., 20).

    Returns
    -------
    tuple of pd.DataFrame
        A tuple containing the training, testing, and validation DataFrames in that order.
    """

    # --- Validate proportions ---
    # Ensure the three percentages cover 100% of the dataset.
    assert train + test + validation == 100, "The sum of train, test, and validation must be exactly 100."

    # --- Calculate cutoff indices ---
    # Define the boundaries of each block based on the percentages.
    n = len(data)
    train_cutoff = int(n * train / 100)
    test_cutoff = train_cutoff + int(n * test / 100)

    # --- Create subsets ---
    # Extract the partitions in order: Train (start -> train_cutoff),
    # Test (train_cutoff -> test_cutoff), and Validation (test_cutoff -> end).
    train_df = data.iloc[:train_cutoff]
    test_df = data.iloc[train_cutoff:test_cutoff]
    validation_df = data.iloc[test_cutoff:]

    # --- Return ---
    # Return the three partitions
    print("✅ Data split successfully\n")

    train_df.to_csv('data/train.csv')
    test_df.to_csv('data/test.csv')
    validation_df.to_csv('data/validation.csv')

    return train_df, test_df, validation_df

import pandas as pd

def show_cointegration_summary(adf_df: pd.DataFrame,
                               johansen_df: pd.DataFrame,
                               decimals: int = 4,
                               save_csv: bool = True) -> pd.DataFrame:
    """
    Combine and display the results of cointegration tests (ADF and Johansen).

    Parameters
    ----------
    adf_df : pd.DataFrame
        DataFrame containing OLS and ADF test results.
    johansen_df : pd.DataFrame
        DataFrame containing Johansen test results.
    decimals : int, optional
        Number of decimal places to round the values to (default is 4).
    save_csv : bool, optional
        If True, saves the combined table to 'data/summary_cointegration_table.csv'.

    Returns
    -------
    pd.DataFrame
        Combined table with correlation and cointegration statistics.
    """

    # Combine DataFrames by asset pairs
    merged = pd.merge(adf_df, johansen_df, on=['Asset_1', 'Asset_2'], how='left')

    # Select relevant columns if they exist
    cols = [
        'Asset_1', 'Asset_2', 'Mean_Correlation',
        'beta1_OLS', 'ADF_stat', 'ADF_pvalue', 'Stationary',
        'Trace_Stat', 'Crit_Value_5%', 'Rank_detected'
    ]
    existing_cols = [c for c in cols if c in merged.columns]
    merged = merged[existing_cols]

    # Round numeric values
    merged = merged.round(decimals)

    # Sort: first cointegrated pairs, then by highest correlation
    if 'Stationary' in merged.columns and 'Mean_Correlation' in merged.columns:
        merged = merged.sort_values(by=['Stationary'], ascending=[False])

    # Display readable table
    print("\n=== Correlation and Cointegration Summary Table ===\n")
    print(merged.head(5))

    # Optional CSV save
    if save_csv:
        merged.to_csv('data/summary_cointegration_table.csv', index=False)

    return merged