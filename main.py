import pandas as pd
from matplotlib import pyplot as plt

from graphs import plot_splits, plot_normalized_prices, plot_spreads, \
    plot_hedge_ratios, plot_kalman_fits, plot_vecm_mean, plot_dynamic_eigenvectors, plot_vecm_signals
from kalman_hedge import run_kalman_on_pair
from kalman_spread import run_kalman2_vecm
from pairs_search import find_correlated_pairs, ols_and_adf, run_johansen_test, extract_pair
from utils import clean_prices, split_dfs

CORR_THRESHOLD = 0.6
THETA = 2
WINDOW = 252


# Pre-processing data
data = pd.read_csv('data/raw_prices.csv')
data = clean_prices(data)
train_df, test_df, val_df = split_dfs(data, 60, 20, 20)
#plot_splits(train_df, test_df, val_df)
print(f"Tickers utilizados:\n{data.columns.values}")


# Encontrar pares correlacionados
correlated_pairs = find_correlated_pairs(train_df, window=WINDOW, threshold=CORR_THRESHOLD)
correlated_pairs.to_csv('data/correlated_pairs.csv', index=False)

# -- PRUEBAS DE COINTEGRACIÓN --
# Regresión OLS y prueba ADF (sobre los residuos) para confirmar estacionariedad
ols_adf_results = ols_and_adf(train_df, correlated_pairs, save_path=f'data/results_ols_adf.csv')
# Run Johansen test (solo para activos con residuos estacionarios)
johansen_results = run_johansen_test(train_df, ols_adf_results, save_path=f'data/results_johansen.csv')


# -- SELECCIÓN DEL PAR --
# Obtain cointegrated pairs
pair1_df = extract_pair(train_df, johansen_results, index=0)
pair2_df = extract_pair(train_df, johansen_results, index=1)
#plot_normalized_prices(pair1_df)
#plot_normalized_prices(pair2_df)

# -- FILTROS DE KALMAN --
# Kalman Filter 1: Dynamic Hedge Ratio (for individual cointegrated pairs found)
kalman1_pair1 = run_kalman_on_pair(pair1_df, q=1e-6, r=1e-1)
kalman1_pair2 = run_kalman_on_pair(pair2_df, q=1e-6, r=1e-1)
#plot_hedge_ratios(kalman1_pair1, kalman1_pair2)
#plot_spreads(kalman1_pair1, kalman1_pair2)
#plot_kalman_fits(kalman1_pair1, kalman1_pair2)


# Kalman Filter 2: Signal generation
# --- KALMAN 2: Generación de Señales (nuevo paso) ---
kalman2_pair1 = run_kalman2_vecm(kalman1_pair1, johansen_results, q=1e-6, r=1e-1, theta=THETA, window=100)
kalman2_pair2 = run_kalman2_vecm(kalman1_pair2, johansen_results, q=1e-6, r=1e-1, theta=THETA, window=100)
plot_vecm_mean(kalman2_pair1)
plot_vecm_mean(kalman2_pair2)
plot_dynamic_eigenvectors(kalman2_pair1)
plot_dynamic_eigenvectors(kalman2_pair2)
plot_vecm_signals(kalman2_pair1)
plot_vecm_signals(kalman2_pair2)
