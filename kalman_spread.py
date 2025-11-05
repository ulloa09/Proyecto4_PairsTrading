# kalman_spread.py
import numpy as np
import pandas as pd
from numpy.ma.core import inner

from kalman_hedge import KalmanFilterReg  # misma clase base

def run_kalman_signal(
    kalman1_df: pd.DataFrame,
    johansen_results: pd.DataFrame,
    theta_input: float = 2.0,
    q2: float = 1e-5,
    r2: float = 1e-3,
    window_z: int = 60,
    save: bool = True,
) -> pd.DataFrame:
    """
    Segundo filtro de Kalman (Kalman 2)
    -----------------------------------
    Genera señales de entrada/salida a partir del spread del filtro 1 y los eigenvectores del test de Johansen.
    Reutiliza la clase KalmanFilterReg (la misma del filtro 1).

    Entradas
    ---------
    kalman1_df : DataFrame
        Contiene las columnas [Activo_1, Activo_2, beta_t_est, spread_t].
        Las dos primeras son los precios de los activos.
    johansen_results : DataFrame
        Contiene ['Asset_1','Asset_2','Eigenvector_1','Eigenvector_2'].
    q2 : float, default=1e-5
        Varianza del ruido de proceso.
    r2 : float, default=1e-3
        Varianza del ruido de observación.
    window_z : int, default=60
        Ventana rolling para el z-score.
    save : bool, default=True
        Si True, guarda el CSV en data/kalman/.

    Retorna
    --------
    DataFrame con columnas:
        ['signal_t','theta_t','E·P_t','z_t','spread_t','beta_t_est']
    """

    # 1) Identificar nombres de los activos directamente
    activo1, activo2 = kalman1_df.columns[0], kalman1_df.columns[1]

    # 2) Obtener eigenvectores correspondientes
    mask = (
        ((johansen_results["Asset_1"] == activo1) & (johansen_results["Asset_2"] == activo2)) |
        ((johansen_results["Asset_1"] == activo2) & (johansen_results["Asset_2"] == activo1))
    )
    row = johansen_results.loc[mask]
    if row.empty:
        raise ValueError(f"No se encontró eigenvector para {activo1}-{activo2} en johansen_results.")

    e1, e2 = float(row["Eigenvector_1"].iloc[0]), float(row["Eigenvector_2"].iloc[0])
    w_init = np.array([e1, e2]).reshape(-1, 1)
    p_init = np.diag([abs(e1), abs(e2)])

    # 3) Variables de observación
    P1, P2 = kalman1_df[activo1].values, kalman1_df[activo2].values
    S_t = P1 * e1 + P2 * e2     # x_t = E·P_t
    y_t = kalman1_df["spread_t"].values  # y_t = spread_t

    # 4) Instancia del mismo filtro
    kf = KalmanFilterReg(q=q2, r=r2, w_init=w_init, p_init=p_init)

    # 5) Ciclo predict/update
    w_hist = np.zeros((len(kalman1_df), 2))
    for i, (x, y) in enumerate(zip(S_t, y_t)):
        kf.predict()
        kf.update(x_t=float(x), y_t=float(y))
        w0, w1 = kf.params
        w_hist[i, 0], w_hist[i, 1] = w0, w1

    # 6) Z-score rolling
    S_series = pd.Series(S_t, index=kalman1_df.index, name="E·P_t")
    mu = S_series.rolling(window=window_z, min_periods=window_z).mean()
    sigma = S_series.rolling(window=window_z, min_periods=window_z).std(ddof=0)
    z = ((S_series - mu) / sigma).rename("z_t")

    # 7) Política de decisión con theta fija ingresada
    theta_t = pd.Series(theta_input, index=kalman1_df.index, name="theta_t")
    signal = pd.Series(0, index=kalman1_df.index, dtype=int, name="signal_t")
    signal[z > theta_input] = -1  # short
    signal[z < -theta_input] = 1  # long

    # 8) Resultado final
    out = pd.DataFrame({
        activo1: P1,
        activo2: P2,
        "spread_t": kalman1_df["spread_t"].astype(float),
        "beta_t_est": kalman1_df["beta_t_est"].astype(float),
        "signal_t": signal,
        "theta_t": theta_t,  # todas iguales al theta_input
        "E·P_t": S_series,
        "z_t": z
    })


    # 9) Guardar CSV (carpeta ya existe)
    out.to_csv(f"data/kalman/kalman2_{activo1}_{activo2}.csv")
    print(f"💾 Kalman2 results saved to: data/kalman/kalman2_{activo1}_{activo2}.csv")

    return out