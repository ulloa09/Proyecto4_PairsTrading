import pandas as pd
from matplotlib.style.core import available
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from functions import get_portfolio_value
from kalman_hedge import KalmanFilterReg
from objects import Operation

theta = [0.5, 2.0]
q = 1e-5
r = 1e-2

def backtest(df: pd.DataFrame, kalman2df: pd.DataFrame,
             window_size:int, theta:float, q: float, r:float):

    # Copias por seguridad
    df = df.copy()
    kalman2df = kalman2df.copy()

    # Condiciones Iniciales
    COM = 0.125
    BORROW_RATE = 0.25/100
    cash = 1_000_000

    # Listas para almacenar
    vecms_hat = []
    portfolio_value = []
    active_long_ops: list[Operation] = []
    active_short_ops: list[Operation] = []

    # Definir KALMANs
    kalman_1 = KalmanFilterReg(q=q, r=r)
    hedge_ratio_list = []
    spreads_list = []

    kalman_2 = kalman(...)

    # Obtener nombres de activos
    asset1, asset2 = df.columns[0], df.columns[1]

    for i, row in df.itertuples():
        p1 = getattr(row, asset1)
        p2 = getattr(row, asset2)

        # ACTUALIZAR KALMAN 1
        y_t = p1
        x_t = p2

        kalman_1.predict()
        alpha_t, beta_t, spread_t = kalman_1.update(y_t, x_t)
        w0, w1 = alpha_t, beta_t
        hedge_ratio = w1  # <- hedge ratio


        # ACTUALIZAR KALMAN 2
        x1 = p1
        x2 = p2
        eigenvector = coint_johansen(df.iloc[i-252:i,:])
        e1, e2 = eigenvector
        vecm = e1 * x1 + e2 * x2
        kalman_2.predict()
        kalman_2.update(x1, x2, vecm)
        e1_hat, e2_hat = kalman_2.params
        vecm_hat = e1_hat * x1 + e2_hat * x2
        vecms_hat.append(vecm_hat)
        vecms_sample = vecms_hat[-252:]
        ### AQUÍ SE NORMALIZA EL VECM PARA OBTENER LA COMPARACIÓN Y SACAR SEÑAL
        vecm_norm =


        # COBRAR BORROW RATE PARA SHORTS DIARIO
        for position in active_short_ops.copy():
            if position.type == 'short' and position.ticker == asset1:
                borr_cost = row.asset1 * position.n_shares * BORROW_RATE
                cash -= borr_cost
            elif position.type == 'short' and position.ticker == asset2:
                borr_cost = row.asset2 * position.n_shares * BORROW_RATE
                cash += borr_cost



        #  APERTURA OPERACIÓN ( VECM NORM > THETA )
        if vecm_norm > theta and active_long_ops is None and active_short_ops is None:

            available = cash * 0.4
            # P1 es el activo barato, se hace LONG
            n_shares_long = available // (p1 * (1+COM))
            costo = n_shares_long * (p1 * (1+COM))
            # P2 es el activo caro, se hace SHORT
            n_shares_short = available * hedge_ratio
            cost_short = p2 * n_shares_short * COM

            ## COMPRA DEL ACTIVO 1
            if available >= costo:
                cash -= costo
                long_op = Operation(ticker=asset1, type='long',
                                    n_shares=n_shares_long, open_price=p1,
                                    close_price=0, date=row.index)
                active_long_ops.append(long_op)

            ## SHORT DEL ACTIVO 2
                cash -= cost_short ## NO SE DEBE SUMAR NADA
                short_op = Operation(ticker=asset2, type='short',
                                     n_shares=n_shares_short, open_price=p2,
                                     close_price=0, date=row.index)
                active_short_ops.append(short_op)


        # APERTURA OPERACIÓN ( VECM NORM < -THETA )
        if vecm_norm < -theta and active_long_ops is None and active_short_ops is None:
            ## COMPRA DEL ACTIVO 2
            available = cash * 0.4
            # Ahora P1 es el activo caro, se hace SHORT
            n_shares_short = available // (p1 * (1+COM))
            cost_short = n_shares_short * (p1 * (1+COM))
            # P2 es el activo barato, se hace LONG
            n_shares_long = available *
            costo = p2 * n_shares_long * COM

            ## COMPRA DEL ACTIVO 2
            if available >= costo:
                cash -= costo
                long_op = Operation(ticker=asset2, type='long',
                                    n_shares=n_shares_long, open_price=p2,
                                    close_price=0, date=row.index)
                active_long_ops.append(long_op)

            ## SHORT DEL ACTIVO 1
                cash -= cost_short
                short_op = Operation(ticker=asset2, type='short',
                                     n_shares=n_shares_short, open_price=p1,
                                     close_price=0, date=row.index)
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
                if position.ticker == asset2:
                    pnl = (p2 - position.open_price)
                    cash += p2 * position.n_shares * (1-COM)
                    position.close_price = p2
            # Quitar posición porque ya se cerró
            active_long_ops.remove(position)

        for position in active_short_ops.copy():
            if abs(vecm_norm) < 0.05:
                if position.ticker == asset1:
                    pnl = (position.open_price - p1) * position.n_shares
                    comission = p1 * position.n_shares * COM
                    cash = pnl - comission
                    position.close_price = p1

                if position.ticker == asset2:
                    pnl = (position.open_price - p2) * position.n_shares
                    comission = p2 * position.n_shares * COM
                    cash = pnl - comission
                    position.close_price = p2
            #Quitar posición porque ya se cerró
            active_short_ops.remove(position)

    return cash, portfolio_value, active_long_ops, active_short_ops









