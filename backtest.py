import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen


from functions import get_portfolio_value
from graphs import plot_dynamic_eigenvectors, plot_vecm_signals, plot_spread_evolution, plot_portfolio_evolution, \
    plot_spread_vs_vecm, plot_normalized_prices, plot_hedge_ratio_evolution, plot_trade_returns_distribution
from kalman_filters import KalmanFilterReg, KalmanFilterVecm
from metrics import generate_metrics
from objects import Operation


def backtest(df: pd.DataFrame, window_size:int,
             theta:float, q: float, r:float):


    # Copias por seguridad
    df = df.copy()

    # Condiciones Iniciales
    DAYS = 252
    COM = 0.00125
    BORROW_RATE = 0.25/100 / DAYS
    cash = 1_000_000



    # Listas para almacenar
    portfolio_value = []
    active_long_ops: list[Operation] = []
    active_short_ops: list[Operation] = []
    pnl_history = []
    entry_long_idx, entry_short_idx, exit_idx = [], [], []

    # Obtener nombres de activos
    asset1, asset2 = df.columns[:2]

    # Definir KALMANs
    # Kalman 1
    kalman_1 = KalmanFilterReg(1e-6, 0.05)#q=q, r=r)
    hedge_ratio_list, spreads_list = [], []

    # Kalman 2
    kalman_2 = KalmanFilterVecm(q=1e-7,r=0.3)#q=q, r=r)
    e1_hat_list, e2_hat_list, vecms_hat_list, vecms_hatnorm_list = [], [], [], []

    print(f"\nIniciando backtesting con:")
    print(f"Cash: {cash}")
    print(f"Activo 1(y):{asset1}, Activo 2(x):{asset2}\n")

    for i, row in enumerate(df.itertuples(index=True)):
        p1 = getattr(row, asset1)
        p2 = getattr(row, asset2)

        # ACTUALIZAR KALMAN 1
        x_t = p2
        y_t = p1

        kalman_1.predict()
        alpha_t, beta_t, spread_t = kalman_1.update(y_t, x_t)
        w0, w1 = alpha_t, beta_t
        hedge_ratio = w1  # <- hedge ratio (redundante)

        hedge_ratio_list.append(hedge_ratio)
        spreads_list.append(spread_t)

        # ACTUALIZAR KALMAN 2
        if i > window_size:
            # Cointegración móvil
            window_data = df.iloc[i - window_size:i,:]
            eig = coint_johansen(window_data, det_order=0, k_ar_diff=1)
            # Eigenvector sin normalizar
            v = eig.evec[:, 0].astype(float)
            e1, e2 = v
            # VECM observado con datos originales
            vecm = e1 * y_t + e2 * x_t
            # Actualizar Kalman 2 con precios y vecm observado
            kalman_2.predict()
            e1_hat, e2_hat, vecm_hat = kalman_2.update(y_t, x_t, vecm)
            # Guardar resultados
            e1_hat_list.append(e1_hat)
            e2_hat_list.append(e2_hat)
            vecms_hat_list.append(vecm_hat)
            # Normalizar vecm_hat (solo si hay suficiente historia)
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
            e1_hat_list.append(0.0)
            e2_hat_list.append(0.0)
            vecms_hat_list.append(0.0)
            vecm_norm = 0.0
            vecms_hatnorm_list.append(0.0)

        ## CERRAR OPERACIONES/POSICIONES
        # LARGAS
        for position in active_long_ops.copy():
            if abs(vecm_norm) < 0.05:
                exit_idx.append(i)
                if position.ticker == asset1:
                    pnl = (p1 - position.open_price) * position.n_shares
                    cash += p1 * position.n_shares * (1 - COM)
                    position.close_price = p1
                    pnl_history.append(pnl)
                if position.ticker == asset2:
                    pnl = (p2 - position.open_price) * position.n_shares
                    cash += p2 * position.n_shares * (1 - COM)
                    position.close_price = p2
                    pnl_history.append(pnl)
                # Quitar posición porque ya se cerró
                active_long_ops.remove(position)

        for position in active_short_ops.copy():
            if abs(vecm_norm) < 0.05:
                exit_idx.append(i)
                if position.ticker == asset1:
                    pnl = (position.open_price - p1) * position.n_shares
                    commission = p1 * position.n_shares * COM
                    cash += pnl - commission
                    position.close_price = p1
                    pnl_history.append(pnl)

                if position.ticker == asset2:
                    pnl = (position.open_price - p2) * position.n_shares
                    commission = p2 * position.n_shares * COM
                    cash += pnl - commission
                    position.close_price = p2
                    pnl_history.append(pnl)
                # Quitar posición porque ya se cerró
                active_short_ops.remove(position)

        # COBRAR BORROW RATE PARA SHORTS DIARIO
        for position in active_short_ops.copy():
            if position.type == 'short' and position.ticker == asset1:
                borr_cost = p1 * position.n_shares * BORROW_RATE
                cash -= borr_cost
            elif position.type == 'short' and position.ticker == asset2:
                borr_cost = p2 * position.n_shares * BORROW_RATE
                cash -= borr_cost



        #  APERTURA OPERACIÓN ( VECM NORM > THETA ) LONG ASSET1 / SHORT ASSET 2
        if (vecm_norm > theta) and (len(active_long_ops) == 0) and (len(active_short_ops) == 0):
            entry_long_idx.append(i)
            available = cash * 0.4
            # ASSET1 es el activo barato, se hace LONG
            n_shares_long = available // (p1 * (1 + COM))
            costo = n_shares_long * p1 * (1+COM)
            # ASSET2 es el activo caro, se hace SHORT
            n_shares_short = int(n_shares_long * abs(hedge_ratio))
            cost_short = p2 * n_shares_short * COM

            ## COMPRA DEL ACTIVO 1
            if available >= costo:
                cash -= costo
                long_op = Operation(ticker=asset1, type='long',
                                    n_shares=n_shares_long, open_price=p1,
                                    close_price=0, date=row.Index)
                active_long_ops.append(long_op)

            ## SHORT DEL ACTIVO 2
                cash -= cost_short ## NO SE DEBE SUMAR NADA
                short_op = Operation(ticker=asset2, type='short',
                                     n_shares=n_shares_short, open_price=p2,
                                     close_price=0, date=row.Index)
                active_short_ops.append(short_op)



        # APERTURA OPERACIÓN ( VECM NORM < -THETA )
        if (vecm_norm < -theta) and (len(active_long_ops) == 0) and (len(active_short_ops) == 0):
            entry_short_idx.append(i)
            ## COMPRA DEL ACTIVO 2
            available = cash * 0.4
            # Ahora P1 es el activo caro, se hace SHORT
            n_shares_short = available // (p1 * (1+COM))
            cost_short = n_shares_short * p1 * (COM)
            # P2 es el activo barato, se hace LONG
            n_shares_long = int(n_shares_short * abs(hedge_ratio))
            costo = p2 * n_shares_long * (1+COM)

            ## COMPRA DEL ACTIVO 2
            if available >= costo:
                cash -= costo
                long_op = Operation(ticker=asset2, type='long',
                                    n_shares=n_shares_long, open_price=p2,
                                    close_price=0, date=row.Index)
                active_long_ops.append(long_op)

            ## SHORT DEL ACTIVO 1
                cash -= cost_short
                short_op = Operation(ticker=asset1, type='short',
                                     n_shares=n_shares_short, open_price=p1,
                                     close_price=0, date=row.Index)
                active_short_ops.append(short_op)

        portfolio_value.append(get_portfolio_value(cash, active_long_ops, active_short_ops,
                                                   x_ticker=asset1, y_ticker=asset2, p1=p1, p2=p2))





    results_df = pd.DataFrame({
        'spread': spreads_list,
        'e1_hat': e1_hat_list,
        'e2_hat': e2_hat_list,
        'hedge_ratio': hedge_ratio_list,
        'vecm_hat': vecms_hat_list,
        'vecm_norm': vecms_hatnorm_list,
        'portfolio_value': portfolio_value,
    }, index=df.index[-len(e1_hat_list):])

    portfolio_series = pd.Series(portfolio_value, index=df.index[-len(portfolio_value):])

    print(pnl_history)
    #plot_hedge_ratio_evolution(results_df)
    #plot_dynamic_eigenvectors(results_df)
    #plot_normalized_prices(df)
    #plot_spread_evolution(results_df, asset2, asset1)
    #plot_vecm_signals(results_df, entry_long_idx, entry_short_idx, exit_idx, theta)
    #plot_portfolio_evolution(portfolio_series)
    #plot_spread_vs_vecm(results_df)
    plot_trade_returns_distribution(pnl_history)

    metrics = generate_metrics(portfolio_series, pnl_history)


    return cash, portfolio_value, metrics
