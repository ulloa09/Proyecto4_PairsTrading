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
    Splits a time-ordered DataFrame into three sets:
    Training, Test, and Validation.

    Args:
        data (pd.DataFrame): The complete, time-sorted DataFrame.
        train (int): Percentage for the training set (e.g., 60).
        test (int): Percentage for the test set (e.g., 20).
        validation (int): Percentage for the validation set (e.g., 20).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - train_df
            - test_df
            - validation_df
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
    print("✅Data split successfully\n")

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
    Combina y muestra los resultados de las pruebas de cointegración (ADF + Johansen).

    Parameters
    ----------
    adf_df : pd.DataFrame
        DataFrame con resultados OLS + ADF.
    johansen_df : pd.DataFrame
        DataFrame con resultados de Johansen.
    decimals : int, optional
        Número de decimales para redondear los valores (por defecto 4).
    save_csv : bool, optional
        Si True, guarda la tabla combinada en 'data/summary_cointegration_table.csv'.

    Returns
    -------
    pd.DataFrame
        Tabla combinada con estadísticas de correlación y cointegración.
    """

    # Combinar los DataFrames por par de activos
    merged = pd.merge(adf_df, johansen_df, on=['Asset_1', 'Asset_2'], how='left')

    # Seleccionar columnas relevantes si existen
    cols = [
        'Asset_1', 'Asset_2', 'Mean_Correlation',
        'beta1_OLS', 'ADF_stat', 'ADF_pvalue', 'Stationary',
        'Trace_Stat', 'Crit_Value_5%', 'Rank_detected'
    ]
    existing_cols = [c for c in cols if c in merged.columns]
    merged = merged[existing_cols]

    # Redondear valores numéricos
    merged = merged.round(decimals)

    # Ordenar: primero pares cointegrados y luego por mayor correlación
    if 'Stationary' in merged.columns and 'Mean_Correlation' in merged.columns:
        merged = merged.sort_values(by=['Stationary'], ascending=[False])

    # Mostrar tabla legible
    print("\n=== Tabla resumen de correlaciones y cointegración ===\n")
    print(merged.head(5))

    # Guardar CSV opcional
    if save_csv:
        merged.to_csv('data/summary_cointegration_table.csv', index=False)

    return merged