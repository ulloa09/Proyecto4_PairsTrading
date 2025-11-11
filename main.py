import pandas as pd
from matplotlib import pyplot as plt

from backtest import backtest
from graphs import plot_splits, plot_normalized_prices, plot_spreads, \
    plot_hedge_ratios, plot_kalman_fits, plot_vecm_mean, plot_dynamic_eigenvectors, plot_vecm_signals
from kalman_hedge import run_kalman_on_pair, KalmanFilterReg
from kalman_spread import run_kalman2_vecm
from pairs_search import find_correlated_pairs, ols_and_adf, run_johansen_test, extract_pair
from utils import clean_prices, split_dfs, extract_pairs_all

CORR_THRESHOLD = 0.6
THETA = 2
WINDOW = 20
Q = 1e-7
R = 3e-2

# Pre-processing data
data = pd.read_csv('data/raw_prices.csv')
data = clean_prices(data)
train_df, test_df, val_df = split_dfs(data, 60, 20, 20)
#plot_splits(train_df, test_df, val_df)
print(f"Tickers utilizados:\n{data.columns.values}")


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
pair1_df = extract_pair(train_df, johansen_results, 0)
pair2_df = extract_pair(train_df, johansen_results, 1)
pair1_train_df, pair2_train_df, pair1_test_df, pair2_test_df, pair1_val_df, pair2_val_df = extract_pairs_all(train_df, test_df, val_df, johansen_results, [0,1])
#plot_normalized_prices(pair1_df)
#plot_normalized_prices(pair2_df)


# PRUEBA BACKTESTING
#cash_p1_train, last_value_p1_train = backtest(pair1_train_df, window_size=WINDOW, theta=THETA, q=Q, r=R)
cash, port_value = backtest(pair2_df, window_size=WINDOW, theta=THETA, q=Q, r=R)
print(f"💰 Capital final: {port_value[-1]:,.2f}")
#print(f"📊 Valor final portafolio: {last_value_p1_train:,.2f}")

plt.figure(figsize=(12,6))
plt.plot(port_value, label='Portfolio Value')
plt.grid()
plt.show()


'''
# -- FILTROS DE KALMAN --
# Kalman Filter 1: Dynamic Hedge Ratio (for individual cointegrated pairs found)
kalman1_pair1 = run_kalman_on_pair(pair1_df, q=1e-8, r=1e-3)
kalman1_pair2 = run_kalman_on_pair(pair2_df, q=1e-8, r=1e-3)
#plot_hedge_ratios(kalman1_pair1, kalman1_pair2)
#plot_spreads(kalman1_pair1, kalman1_pair2)
#plot_kalman_fits(kalman1_pair1, kalman1_pair2)


# Kalman Filter 2: Signal generation
# --- KALMAN 2: Generación de Señales (nuevo paso) ---
kalman2_pair1 = run_kalman2_vecm(kalman1_pair1, johansen_results, q=1e-6, r=1e-1, theta=THETA, window=WINDOW)
kalman2_pair2 = run_kalman2_vecm(kalman1_pair2, johansen_results, q=1e-6, r=1e-1, theta=THETA, window=WINDOW)
plot_vecm_mean(kalman2_pair1)
plot_vecm_mean(kalman2_pair2)
plot_dynamic_eigenvectors(kalman2_pair1)
plot_dynamic_eigenvectors(kalman2_pair2)
plot_vecm_signals(kalman2_pair1)
plot_vecm_signals(kalman2_pair2)
'''

