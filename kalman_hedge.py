import numpy as np
import pandas as pd
from pathlib import Path

class KalmanFilterReg:
    """
    Kalman filter for a dynamic linear regression:
        y_t = α_t + β_t * x_t + v_t
    with state evolution:
        w_t = w_{t-1} + η_t
    where w_t = [α_t, β_t]^T.

    Parameters
    ----------
    q : float
        Process noise variance (Q = q * I_2).
    r : float
        Observation noise variance (R = r).
    w_init : np.ndarray, optional
        Initial state vector [α_0, β_0].
    p_init : np.ndarray, optional
        Initial covariance matrix (2x2).
    """

    def __init__(self, q: float = 1e-5, r: float = 1e-2,
                 w_init=None, p_init=None):
        # Estado inicial (α_0, β_0)
        self.w = np.array([[0.0], [0.0]]) if w_init is None else w_init.reshape(-1, 1)
        # Covarianza inicial del estado (P_0)
        self.P = np.eye(2) * 1.0 if p_init is None else p_init
        # Matriz de transición del estado (identidad: paseo aleatorio)
        self.A = np.eye(2)
        # Varianzas del proceso y observación
        self.q = q
        self.r = r

    def predict(self):
        """Predicción del estado y su covarianza."""
        self.w = self.A @ self.w
        self.P = self.A @ self.P @ self.A.T + self.q * np.eye(2)

    def update(self, y_t: float, x_t: float):
        """
        Actualiza el estado dado un nuevo par (x_t, y_t).
        Devuelve la estimación instantánea (α_t, β_t) y el spread.
        """
        # Vector de observación
        C_t = np.array([[1.0, x_t]])
        # Predicción del precio
        y_pred = float(C_t @ self.w)
        # Innovación (error)
        e_t = y_t - y_pred
        # Varianza de la innovación
        S_t = C_t @ self.P @ C_t.T + self.r
        # Ganancia de Kalman
        K_t = self.P @ C_t.T / S_t
        # Actualización del estado
        self.w = self.w + K_t * e_t
        # Actualización de la covarianza
        self.P = (np.eye(2) - K_t @ C_t) @ self.P

        alpha_t, beta_t = float(self.w[0]), float(self.w[1])
        spread_t = y_t - (alpha_t + beta_t * x_t)
        return alpha_t, beta_t, spread_t


def run_kalman_on_pair(pair_df: pd.DataFrame, q: float = 1e-5, r: float = 1e-2):
    """
    Ejecuta el primer filtro de Kalman sobre un par cointegrado.

    Parámetros
    ----------
    csv_path : str
        Ruta al archivo CSV con precios de dos activos (e.g. 'data/KO_DUK_pair.csv').
    q, r : float
        Parámetros de ruido de proceso y observación.

    Devuelve
    --------
    df_result : pd.DataFrame
        DataFrame con columnas [Asset_1, Asset_2, alpha_t, beta_t, spread_t].
    También guarda el resultado en data/kalman/kalman1_<Asset_1>_<Asset_2>.csv
    """

    df = pair_df.copy()
    df.columns = [col.strip() for col in df.columns]  # limpieza ligera
    asset1, asset2 = df.columns[:2]
    y, x = df[asset1].values, df[asset2].values

    kf = KalmanFilterReg(q=q, r=r)

    results = []
    for y_t, x_t in zip(y, x):
        kf.predict()
        alpha_t, beta_t, spread_t = kf.update(y_t, x_t)
        y_pred_t = alpha_t + beta_t * x_t
        results.append((alpha_t, beta_t, spread_t, y_pred_t))

    df_res = pd.DataFrame(results, columns=['alpha', 'beta', 'spread', 'y_pred'], index=df.index)
    df_res[asset1] = df[asset1]
    df_res[asset2] = df[asset2]
    df_res = df_res[[asset1, asset2, 'alpha', 'beta', 'y_pred', 'spread']]

    # Guarda resultado en data/kalman/
    output_path = Path("data/kalman") / f"kalman1_{asset1}_{asset2}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(output_path, index=True)

    return df_res