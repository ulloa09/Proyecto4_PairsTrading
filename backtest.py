from matplotlib.style.core import available
from statsmodels.tsa.vector_ar.vecm import coint_johansen

theta = [0.5, 2.0]
COM = 0.125
BORROW_RATE = 0.25/100

kalman_1 = kalman(...)
kalman_2 = kalman(...)

vecms_hat = []

asset1, asset2 = df.columns[0], df.columns[1]

for i, row in data.iterrows():
    p1 = row.asset1
    p2 = row.asset2

    # ACTUALIZAR KALMAN 1
    y = p1
    x = p2

    kalman_1.update(x, y)
    w0, w1 = kalman_1.params
    hedge_ratio = w1

    # ACTUALIZAR KALMAN 2
    x1 = p1
    x2 = p2
    eigenvector = coint_johansen(df.iloc[i-252:i,:])
    e1, e2 = eigenvector
    vecm = e1 * x1 + e2 * x2
    kalman_2.update(x1, x2, vecm)
    e1_hat, e2_hat = kalman_2.params
    vecm_hat = e1_hat * x1 + e2_hat * x2
    vecms_hat.append(vecm_hat)
    vecms_sample = vecms_hat[-252:]
    vecm_norm = ### AQUÍ SE NORMALIZA EL VECM PARA OBTENER LA COMPARACIÓN Y SACAR SEÑAL


    #  APERTURA OPERACIÓN ( VECM NORM supera THETA )
    if vecm_norm > theta and active_long_positions is None and active_short_positions is None:
        ## COMPRA DEL ACTIVO 1
        available = cash * 0.4
        n_shares_long = available // (p1 * (1+COM))
        if available >= n_shares_long * (p1*(1+COM)):
            cash -=

        ## SHORT DEL ACTIVO 2
        n_shares_short = available * hedge_ratio
        cost = p2 * n_shares_short * COM
        cash -=   ## NO SE DEBE SUMAR NADA


    # CIERRE DE POSICIONES
    if abs(vecm_norm) < 0.05

