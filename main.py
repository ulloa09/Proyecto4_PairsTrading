import pandas as pd

from graphs import plot_hedge_ratio
from kalman_hedge import run_kalman_on_pair
from kalman_spread import run_kalman_spread
from pairs_search import find_correlated_pairs, ols_and_adf, run_johansen_test, extract_pair
from utils import clean_prices, split_dfs

# Pre-processing data
data = pd.read_csv('data/raw_prices.csv')
data = clean_prices(data)
data.to_csv('data/prices.csv')
train_df, test_df, val_df = split_dfs(data, 60, 20, 20)
train_df.to_csv('data/train.csv')
test_df.to_csv('data/test.csv')
val_df.to_csv('data/validation.csv')
print(f"Tickers utilizados:\n{data.columns.values}")

# Encontrar pares correlacionados
correlated_pairs = find_correlated_pairs(train_df, window=252, threshold=0.7)
correlated_pairs.to_csv('data/correlated_pairs.csv', index=False)

# -- PRUEBAS DE COINTEGRACIÓN --
# Regresión OLS y prueba ADF (sobre los residuos) para confirmar estacionariedad
ols_adf_results = ols_and_adf(train_df, correlated_pairs, save_path=f'data/results_ols_adf.csv')
# Run Johansen test (solo para activos con residuos estacionarios)
johansen_results = run_johansen_test(train_df, ols_adf_results, save_path=f'data/results_johansen.csv')

# -- SELECCIÓN DEL PAR --
# Obtain individual DFs for cointegrated pairs found
pair1_df = extract_pair(train_df, johansen_results, index=0)
pair2_df = extract_pair(train_df, johansen_results, index=1)

# -- FILTROS DE KALMAN --
# Kalman Filter 1: Dynamic Hedge Ratio (for individual cointegrated pairs found)
kalman1_pair1 = run_kalman_on_pair(pair1_df)
kalman1_pair2 = run_kalman_on_pair(pair2_df)
plot_hedge_ratio(kalman1_pair1)
plot_hedge_ratio(kalman1_pair2)

# Kalman Filter 2: Dynamic Spread (mean + z-score)
kalman2_pair1 = run_kalman_spread(kalman1_pair1)
kalman2_pair2 = run_kalman_spread(kalman1_pair2)


