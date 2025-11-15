"""Main script for the Pairs Trading project.

This script orchestrates the workflow for identifying and backtesting pairs trading strategies.
It includes data preprocessing, correlation-based pair filtering, cointegration testing using
OLS, ADF, and Johansen tests, application of Kalman filters for spread regression, and
backtesting the trading strategy based on the identified pairs.

"""

import pandas as pd
from matplotlib import pyplot as plt

from backtest import backtest
from graphs import plot_spread_regression
from pairs_search import find_correlated_pairs, ols_and_adf, run_johansen_test, extract_pair
from utils import clean_prices, split_dfs, show_cointegration_summary

# Configuration parameters for the pairs trading strategy
#
# CORR_THRESHOLD : float
#     Minimum correlation coefficient threshold to consider a pair as correlated.
# THETA : float
#     Threshold parameter for the trading signal in the strategy.
# WINDOW : int
#     Window size (in trading days) used for rolling calculations and backtesting.
# Q : float
#     Process noise covariance parameter for Kalman filter in spread regression.
# R : float
#     Measurement noise covariance parameter for Kalman filter in spread regression.
CORR_THRESHOLD = 0.6
THETA = 0.7
WINDOW = 252
Q = 1e-7
R = 1e-3

# Pre-processing data
data = pd.read_csv('data/raw_prices.csv')
data = clean_prices(data)
train_df, test_df, val_df = split_dfs(data, 60, 20, 20)


# Find correlated pairs
corr_matrix, correlated_pairs = find_correlated_pairs(train_df, window=252, threshold=CORR_THRESHOLD)
correlated_pairs.to_csv('data/correlated_pairs.csv', index=False)

# -- COINTEGRATION TESTS --
# OLS regression and ADF test (on residuals) to confirm stationarity
ols_adf_results = ols_and_adf(train_df, correlated_pairs, save_path=f'data/results_ols_adf.csv')
# Run Johansen test (only for assets with stationary residuals)
johansen_results = run_johansen_test(train_df, ols_adf_results, save_path=f'data/results_johansen.csv')
show_cointegration_summary(ols_adf_results, johansen_results)

# -- PAIR SELECTION --
# Obtain cointegrated pairs
#pair1_df = extract_pair(train_df, johansen_results, 0)
pair2_df = extract_pair(data, johansen_results, 1)
pair2_train_df = extract_pair(train_df, johansen_results, 1)
pair2_test_df = extract_pair(test_df, johansen_results, 1)
pair2_val_df = extract_pair(val_df, johansen_results, 1)


# BACKTESTING
cash, port_value, metrics = backtest(pair2_df, window_size=WINDOW, theta=THETA, q=Q, r=R)
print(f"💰 Final portfolio value: {port_value[-1]:,.2f}")
print(f"💵 Remaining cash: {cash:,.2f}\n")
