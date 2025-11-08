import numpy as np
import pandas as pd
from matplotlib.dates import DAYS_PER_YEAR
from matplotlib.style.core import available
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from functions import get_portfolio_value
from kalman_hedge import KalmanFilterReg
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
    signals = []
    pnl_history = []

    # Obtener nombres de activos
    asset1, asset2 = df.columns[:2]

    # Definir KALMANs
    # Kalman 1
    kalman_1 = KalmanFilterReg(q=q, r=r)
    hedge_ratio_list, spreads_list = [], []

    # Kalman 2
    e1_hat_list,e2_hat_list, vecms_hat = [], [], []
    # Inicialización
    init_window = df.iloc[:window_size, [0,1]]
    init_johansen = coint_johansen(init_window, det_order=0, k_ar_diff=1)
    w = init_johansen.evec[:, 0].reshape(-1, 1)
    P = np.eye(2) * 10  # Covarianza inicial
    Q = np.eye(2) * q  # Ruido proceso
    R = np.eye(2) * r  # Ruido observación
    vecm_norm = 0.0



    for i, row in enumerate(df.itertuples(index=True)):
        p1 = getattr(row, asset1)
        p2 = getattr(row, asset2)

        # ACTUALIZAR KALMAN 1
        y_t = p1
        x_t = p2

        kalman_1.predict()
        alpha_t, beta_t, spread_t = kalman_1.update(y_t, x_t)
        w0, w1 = alpha_t, beta_t
        hedge_ratio = w1  # <- hedge ratio (redundante)

        hedge_ratio_list.append(hedge_ratio)
        spreads_list.append(spread_t)


        # ACTUALIZAR KALMAN 2
        if i >= window_size:
            window = df.iloc[i-window_size:i, [0,1]].copy()
            x = window[asset1].values
            y = window[asset2].values

            # OBSERVACIÓN
            johansen = coint_johansen(window, det_order=0, k_ar_diff=1)
            eigenvector = johansen.evec[:, 0].reshape(-1, 1) # Estado inicial es el EIGENVECTOR

            # PREDICCIÓN
            w_pred = w                    # w_t = w_{t-1} + n_t
            P_pred = P + Q

            # ACTUALIZACIÓN
            innovation = eigenvector - w_pred
            S = P_pred + R                   # Covarianza de innovación
            K = P_pred @ np.linalg.pinv(S)    # Ganancia de KALMAN
            w = w_pred + K @ innovation      # Eigenvector filtrado
            P = (np.eye(2) - K) @ P_pred

            # GUARDADO RESULTADOS
            e1_hat, e2_hat = float(w[0]), float(w[1])
            e1_hat_list.append(e1_hat)
            e2_hat_list.append(e2_hat)

            # VECM filtrado (hat)
            vecm_hat = e1_hat * y_t + e2_hat * x_t
            vecms_hat.append(vecm_hat)
            vecms_sample = vecms_hat[-252:]

            ### NORMALIZACIÓN DEL VECM
            if len(vecms_sample) >= 252:
                mu = np.mean(vecms_sample)
                std = np.std(vecms_sample)
                vecm_norm = (vecm_hat - mu) / (std)
            else:
                vecm_norm = 0

            # GENERACIÓN DE SEÑALES
            if vecm_norm > theta:
                signal = -1
                signals.append(signal)
            elif vecm_norm < -theta:
                signal = 1
                signals.append(signal)
            else:
                signal = 0


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

            available = cash * 0.4
            # ASSET1 es el activo barato, se hace LONG
            n_shares_long = available // (p1 * (1+COM))
            costo = n_shares_long * (p1 * (1+COM))
            # ASSET2 es el activo caro, se hace SHORT
            n_shares_short = available * abs(hedge_ratio)
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
            ## COMPRA DEL ACTIVO 2
            available = cash * 0.4
            # Ahora P1 es el activo caro, se hace SHORT
            n_shares_short = available // (p1 * (1+COM))
            cost_short = n_shares_short * (p1 * (1+COM))
            # P2 es el activo barato, se hace LONG
            n_shares_long = available * abs(hedge_ratio)
            costo = p2 * n_shares_long * COM

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
                                                   x_ticker=asset1, y_ticker=asset2))



    ## CERRAR OPERACIONES/POSICIONES
        # LARGAS
        for position in active_long_ops.copy():
            if abs(vecm_norm) < 0.05:
                if position.ticker == asset1:
                    pnl = (p1 - position.open_price)
                    cash += p1 * position.n_shares * (1-COM)
                    position.close_price = p1
                    pnl_history.append(pnl)
                if position.ticker == asset2:
                    pnl = (p2 - position.open_price)
                    cash += p2 * position.n_shares * (1-COM)
                    position.close_price = p2
                    pnl_history.append(pnl)
                # Quitar posición porque ya se cerró
                active_long_ops.remove(position)

        for position in active_short_ops.copy():
            if abs(vecm_norm) < 0.05:
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
                #Quitar posición porque ya se cerró
                active_short_ops.remove(position)

    return cash, portfolio_value[-1], active_long_ops, active_short_ops









