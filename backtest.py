"""Backtesting module for pairs trading strategy using Kalman Filters and VECM.

This module implements a backtesting framework for a pairs trading strategy based on
cointegration and Kalman filter estimation. It leverages statistical techniques such
as the Johansen cointegration test and Vector Error Correction Model (VECM) to identify
trading signals. Two Kalman filters are used: one for estimating hedge ratios and spreads,
and another for dynamically updating the VECM parameters.

The backtest simulates trading operations including opening and closing long and short
positions, accounting for transaction costs, borrowing costs, and portfolio value evolution.
It produces various diagnostic plots and performance metrics to evaluate strategy efficacy.

Components:
- Johansen cointegration test for dynamic cointegration relationship identification.
- KalmanFilterReg for hedge ratio and spread estimation.
- KalmanFilterVecm for VECM parameter and signal estimation.
- Trading logic based on normalized VECM signals crossing thresholds.
- Cost accounting for commissions and borrowing rates.
- Portfolio valuation and performance metric generation.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen


from functions import get_portfolio_value
from graphs import plot_dynamic_eigenvectors, plot_vecm_signals, plot_spread_evolution, plot_portfolio_evolution, \
    plot_spread_vs_vecm, plot_normalized_prices, plot_trade_returns_distribution, plot_hedge_ratio_evolution
from kalman_filters import KalmanFilterReg, KalmanFilterVecm
from metrics import generate_metrics
from objects import Operation


def backtest(df: pd.DataFrame, window_size:int,
             theta:float, q: float, r:float):
    """
    Executes a backtest of a pairs trading strategy using Kalman filters and VECM.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing price series of two assets to trade. Must have at least two columns.
    window_size : int
        Size of the rolling window used for cointegration estimation and normalization.
    theta : float
        Threshold parameter for opening and closing trading positions based on normalized VECM signals.
    q : float
        Process noise covariance parameter for Kalman filters (currently unused in initialization).
    r : float
        Measurement noise covariance parameter for Kalman filters (currently unused in initialization).

    Returns:
    --------
    cash : float
        Final cash value after completing the backtest.
    portfolio_value : list[float]
        Time series of portfolio values throughout the backtest.
    metrics : dict
        Dictionary of performance metrics generated from the backtest results.

    Description:
    ------------
    The function simulates a pairs trading strategy by iterating over price data. It uses two Kalman filters:
    - KalmanFilterReg estimates the hedge ratio and spread between the two assets.
    - KalmanFilterVecm estimates VECM parameters and normalized signals to generate trading signals.

    Trading decisions to open or close positions are based on the normalized VECM signals crossing positive or negative
    thresholds defined by `theta`. Positions are opened with fixed capital allocation and closed when signals revert
    near zero. Transaction costs and borrowing costs for shorts are accounted for dynamically. The portfolio value is
    updated and tracked over time. Various diagnostic plots and performance metrics are generated at the conclusion.
    """

    # Make a copy of the input DataFrame to avoid modifying original data
    df = df.copy()

    # Constants for backtest simulation
    DAYS = 252  # Number of trading days in a year
    COM = 0.00125  # Commission rate per trade
    BORROW_RATE = 0.25/100 / DAYS  # Daily borrowing rate for short positions
    cash = 1_000_000  # Starting cash capital

    # Lists to store backtest outputs and track open positions
    portfolio_value = []  # Portfolio value history
    active_long_ops: list[Operation] = []  # Currently active long positions
    active_short_ops: list[Operation] = []  # Currently active short positions
    pnl_history = []  # Profit and loss history for closed trades
    entry_long_idx, entry_short_idx, exit_idx = [], [], []  # Indices of trade entries and exits
    comisiones_totales = 0.0  # Total commissions paid
    borrows_totales = 0.0  # Total borrowing costs paid

    # Extract asset names from DataFrame columns
    asset1, asset2 = df.columns[:2]

    # Initialize Kalman filter for hedge ratio and spread estimation
    # KalmanFilterReg parameters: process noise and measurement noise (currently hardcoded)
    kalman_1 = KalmanFilterReg(1e-6, 0.05)  # q and r parameters are placeholders here
    hedge_ratio_list, spreads_list = [], []  # Store hedge ratio and spread estimates over time

    # Initialize Kalman filter for VECM parameter estimation
    kalman_2 = KalmanFilterVecm(q=1e-7, r=0.3)  # q and r parameters are placeholders here
    e1_hat_list, e2_hat_list, vecms_hat_list, vecms_hatnorm_list = [], [], [], []  # Store VECM estimates

    # Print initial backtest configuration
    print(f"\nStarting backtest:")
    print(f"Cash: {cash}")
    print(f"Asset 1 (y): {asset1}, Asset 2 (x): {asset2}\n")

    # Iterate over each row of price data for backtesting
    for i, row in enumerate(df.itertuples(index=True)):
        p1 = getattr(row, asset1)  # Price of asset 1 at current timestep
        p2 = getattr(row, asset2)  # Price of asset 2 at current timestep

        # ----------------------------
        # Update Kalman Filter 1 (hedge ratio and spread estimation)
        # ----------------------------
        x_t = p2
        y_t = p1

        kalman_1.predict()
        alpha_t, beta_t, spread_t = kalman_1.update(y_t, x_t)
        w0, w1 = alpha_t, beta_t
        hedge_ratio = w1  # Hedge ratio estimate from Kalman filter

        # Store hedge ratio and spread for analysis
        hedge_ratio_list.append(hedge_ratio)
        spreads_list.append(spread_t)

        # ----------------------------
        # Update Kalman Filter 2 (VECM parameter and signal estimation)
        # ----------------------------
        if i > window_size:
            # Extract rolling window data for cointegration estimation
            window_data = df.iloc[i - window_size:i,:]
            eig = coint_johansen(window_data, det_order=0, k_ar_diff=1)  # Johansen cointegration test
            v = eig.evec[:, 0].astype(float)  # First eigenvector (cointegration vector)
            e1, e2 = v
            # Calculate observed VECM value with current prices
            vecm = e1 * y_t + e2 * x_t

            # Update KalmanFilterVecm with observed VECM and current prices
            kalman_2.predict()
            e1_hat, e2_hat, vecm_hat = kalman_2.update(y_t, x_t, vecm)

            # Store VECM parameter estimates and signal
            e1_hat_list.append(e1_hat)
            e2_hat_list.append(e2_hat)
            vecms_hat_list.append(vecm_hat)

            # Normalize vecm_hat using rolling window mean and std deviation if enough history available
            if len(vecms_hat_list) > window_size:
                vecms_sample = vecms_hat_list[-window_size:]
                mu = np.mean(vecms_sample)
                std = np.std(vecms_sample)
                vecm_norm = (vecm_hat - mu) / (std)
                vecms_hatnorm_list.append(vecm_norm)
            else:
                vecm_norm = 0.0
                vecms_hatnorm_list.append(vecm_norm)

        else:
            # For initial timesteps before window_size, append zeros as placeholders
            e1_hat_list.append(0.0)
            e2_hat_list.append(0.0)
            vecms_hat_list.append(0.0)
            vecm_norm = 0.0
            vecms_hatnorm_list.append(0.0)

        # ----------------------------
        # Close positions if normalized VECM signal reverts near zero
        # ----------------------------

        # Close long positions if absolute normalized VECM signal is below threshold
        for position in active_long_ops.copy():
            if abs(vecm_norm) < 0.05:
                exit_idx.append(i)
                if position.ticker == asset1:
                    # Calculate PnL and update cash after closing long position on asset1
                    pnl = (p1 - position.open_price) * position.n_shares
                    cash += p1 * position.n_shares * (1 - COM)  # Deduct commission on sale
                    position.close_price = p1
                    pnl_history.append(pnl)
                if position.ticker == asset2:
                    # Calculate PnL and update cash after closing long position on asset2
                    pnl = (p2 - position.open_price) * position.n_shares
                    cash += p2 * position.n_shares * (1 - COM)
                    position.close_price = p2
                    pnl_history.append(pnl)
                # Remove closed position from active list
                active_long_ops.remove(position)

        # Close short positions if absolute normalized VECM signal is below threshold
        for position in active_short_ops.copy():
            if abs(vecm_norm) < 0.05:
                exit_idx.append(i)
                if position.ticker == asset1:
                    # Calculate PnL and update cash after closing short position on asset1
                    pnl = (position.open_price - p1) * position.n_shares
                    commission = p1 * position.n_shares * COM
                    cash += pnl - commission  # Deduct commission from proceeds
                    position.close_price = p1
                    pnl_history.append(pnl)

                if position.ticker == asset2:
                    # Calculate PnL and update cash after closing short position on asset2
                    pnl = (position.open_price - p2) * position.n_shares
                    commission = p2 * position.n_shares * COM
                    cash += pnl - commission
                    position.close_price = p2
                    pnl_history.append(pnl)
                # Remove closed position from active list
                active_short_ops.remove(position)

        # ----------------------------
        # Charge daily borrowing cost for active short positions
        # ----------------------------
        for position in active_short_ops.copy():
            if position.type == 'short' and position.ticker == asset1:
                borr_cost = p1 * position.n_shares * BORROW_RATE
                cash -= borr_cost
            elif position.type == 'short' and position.ticker == asset2:
                borr_cost = p2 * position.n_shares * BORROW_RATE
                cash -= borr_cost
            borrows_totales += borr_cost

        # ----------------------------
        # Open new positions based on normalized VECM signal exceeding positive threshold (long asset1, short asset2)
        # ----------------------------
        if (vecm_norm > theta) and (len(active_long_ops) == 0) and (len(active_short_ops) == 0):
            entry_long_idx.append(i)
            available = cash * 0.4  # Allocate 40% of cash for this trade

            # Calculate number of shares to buy for asset1 (long)
            n_shares_long = available // (p1 * (1 + COM))
            costo = n_shares_long * p1 * (1 + COM)  # Total cost including commission

            # Calculate number of shares to short for asset2, scaled by hedge ratio
            n_shares_short = int(n_shares_long * abs(hedge_ratio))
            cost_short = p2 * n_shares_short * COM  # Commission cost for shorting asset2

            # Accumulate commissions for both legs
            comisiones_totales += cost_short + (n_shares_long * p1 * COM)

            # Execute long buy on asset1 if sufficient funds available
            if available >= costo:
                cash -= costo
                long_op = Operation(ticker=asset1, type='long',
                                    n_shares=n_shares_long, open_price=p1,
                                    close_price=0, date=row.Index)
                active_long_ops.append(long_op)

                # Execute short sell on asset2
                cash -= cost_short  # Deduct commission cost only for short leg
                short_op = Operation(ticker=asset2, type='short',
                                     n_shares=n_shares_short, open_price=p2,
                                     close_price=0, date=row.Index)
                active_short_ops.append(short_op)

        # ----------------------------
        # Open new positions based on normalized VECM signal below negative threshold (long asset2, short asset1)
        # ----------------------------
        if (vecm_norm < -theta) and (len(active_long_ops) == 0) and (len(active_short_ops) == 0):
            entry_short_idx.append(i)
            available = cash * 0.4  # Allocate 40% of cash for this trade

            # Calculate number of shares to short for asset1 (now considered expensive)
            n_shares_short = available // (p1 * (1 + COM))
            cost_short = n_shares_short * p1 * COM  # Commission on short leg

            # Calculate number of shares to long for asset2, scaled by hedge ratio
            n_shares_long = int(n_shares_short * abs(hedge_ratio))
            costo = p2 * n_shares_long * (1 + COM)  # Total cost including commission on long leg

            # Accumulate commissions for both legs
            comisiones_totales += cost_short + (p2 * n_shares_long * COM)

            # Execute long buy on asset2 if sufficient funds available
            if available >= costo:
                cash -= costo
                long_op = Operation(ticker=asset2, type='long',
                                    n_shares=n_shares_long, open_price=p2,
                                    close_price=0, date=row.Index)
                active_long_ops.append(long_op)

                # Execute short sell on asset1
                cash -= cost_short  # Deduct commission cost only for short leg
                short_op = Operation(ticker=asset1, type='short',
                                     n_shares=n_shares_short, open_price=p1,
                                     close_price=0, date=row.Index)
                active_short_ops.append(short_op)

        # ----------------------------
        # Update portfolio value with current prices and active positions
        # ----------------------------
        portfolio_value.append(get_portfolio_value(cash, active_long_ops, active_short_ops,
                                                   x_ticker=asset1, y_ticker=asset2, p1=p1, p2=p2))

    # Compile results into a DataFrame indexed by the corresponding timestamps
    results_df = pd.DataFrame({
        'spread': spreads_list,
        'e1_hat': e1_hat_list,
        'e2_hat': e2_hat_list,
        'hedge_ratio': hedge_ratio_list,
        'vecm_hat': vecms_hat_list,
        'vecm_norm': vecms_hatnorm_list,
        'portfolio_value': portfolio_value,
    }, index=df.index[-len(e1_hat_list):])

    # Create a Series for portfolio value indexed appropriately
    portfolio_series = pd.Series(portfolio_value, index=df.index[-len(portfolio_value):])

    # Generate diagnostic plots for analysis
    plot_hedge_ratio_evolution(results_df)
    plot_dynamic_eigenvectors(results_df)
    plot_normalized_prices(df)
    plot_spread_evolution(results_df, asset2, asset1)
    plot_vecm_signals(results_df, entry_long_idx, entry_short_idx, exit_idx, theta)
    plot_portfolio_evolution(portfolio_series)
    plot_spread_vs_vecm(results_df)
    plot_trade_returns_distribution(pnl_history)

    # Generate performance metrics from portfolio and trade results
    metrics = generate_metrics(portfolio_series, pnl_history, borrows_totales, comisiones_totales)

    return cash, portfolio_value, metrics
