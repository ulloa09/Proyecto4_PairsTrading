import pandas as pd
from matplotlib import pyplot as plt

from backtest import backtest
from graphs import plot_splits, plot_normalized_prices, plot_spreads, \
    plot_hedge_ratios, plot_kalman_fits, plot_vecm_mean, plot_dynamic_eigenvectors, plot_vecm_signals
from kalman_hedge import run_kalman_on_pair, KalmanFilterReg
from kalman_spread import run_kalman2_vecm
from pairs_search import find_correlated_pairs, ols_and_adf, run_johansen_test, extract_pair
from utils import clean_prices, split_dfs

CORR_THRESHOLD = 0.6
THETA = 0.5
WINDOW = 252
Q = 1e-7
R = 1e-3

# Pre-processing data
data = pd.read_csv('data/raw_prices.csv')
data = clean_prices(data)
train_df, test_df, val_df = split_dfs(data, 60, 20, 20)


# Encontrar pares correlacionados
correlated_pairs = find_correlated_pairs(train_df, window=252, threshold=CORR_THRESHOLD)
correlated_pairs.to_csv('data/correlated_pairs.csv', index=False)

# -- PRUEBAS DE COINTEGRACIÓN --
# Regresión OLS y prueba ADF (sobre los residuos) para confirmar estacionariedad
ols_adf_results = ols_and_adf(train_df, correlated_pairs, save_path=f'data/results_ols_adf.csv')
# Run Johansen test (solo para activos con residuos estacionarios)
johansen_results = run_johansen_test(train_df, ols_adf_results, save_path=f'data/results_johansen.csv')


# -- SELECCIÓN DEL PAR --
# Obtain cointegrated pairs
#pair1_df = extract_pair(train_df, johansen_results, 0)
pair2_df = extract_pair(data, johansen_results, 1)
pair2_train_df = extract_pair(train_df, johansen_results, 1)
pair2_test_df = extract_pair(test_df, johansen_results, 1)
pair2_val_df = extract_pair(val_df, johansen_results, 1)


# BACKTESTING
cash, port_value = backtest(pair2_df, window_size=WINDOW, theta=THETA, q=Q, r=R)
print(f"💰 Capital final: {port_value[-1]:,.2f}")
print(f"💵 Cash restante: {cash:,.2f}\n")


plt.figure(figsize=(12,6))
plt.plot(port_value, label='Portfolio Value')
plt.grid()
plt.show()
