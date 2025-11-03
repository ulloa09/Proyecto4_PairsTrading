import pandas as pd

from kalman_hedge import run_kalman_on_pair
from pairs_search import find_correlated_pairs, ols_and_adf, run_johansen_test, extract_pair
from utils import clean_prices

# Reading data CSV
data = pd.read_csv('data/prices.csv')
data = clean_prices(data)
print(f"Tickers utilizados:\n{data.columns.values}")
print(data.info())

# Encontrar pares correlacionados
correlated_pairs = find_correlated_pairs(data, window=252, threshold=0.6)
correlated_pairs.to_csv('data/correlated_pairs.csv', index=False)

# Regresión OLS y prueba ADF (sobre los residuos) para confirmar estacionariedad
ols_adf_results = ols_and_adf(data, correlated_pairs, save_path=f'data/results_ols_adf.csv')

# Run Johansen test (solo para activos con residuos estacionarios)
johansen_results = run_johansen_test(data, ols_adf_results, save_path=f'data/results_johansen.csv')

# Obtain individual DFs for cointegrated pairs
pair1_df = extract_pair(data, johansen_results, index=0)
pair2_df = extract_pair(data, johansen_results, index=1)

# Kalman Filter 1: Dynamic Hedge Ratio
kalman1_pair1 = run_kalman_on_pair(pair1_df)
kalman1_pair2 = run_kalman_on_pair(pair2_df)