"""
Module for applying Kalman filters in the pairs trading project.

This module implements two key Kalman filter models to support dynamic pairs trading strategies:
1. Kalman Filter 1 (KalmanFilterReg): Estimates the dynamic hedge ratio between two cointegrated assets via a time-varying linear regression model.
2. Kalman Filter 2 (KalmanFilterVecm and run_kalman2_vecm): Models the dynamic eigenvectors of a cointegration vector error correction model (VECM) with explicit stationary noise, enabling robust estimation of cointegration relationships over time.

These filters facilitate sequential decision modeling by providing updated parameters and spread estimates, which can be integrated with backtesting frameworks to evaluate trading signals and strategy performance.

The module supports loading price data, running the filters, and saving results for subsequent analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path

class KalmanFilterReg:
    """
    Kalman filter for dynamic linear regression modeling a time-varying hedge ratio.

    The model assumes the observation equation:
        y_t = α_t + β_t * x_t + v_t,
    where v_t ~ N(0, r), and the state evolution:
        w_t = w_{t-1} + η_t,
    where w_t = [α_t, β_t]^T and η_t ~ N(0, q * I_2).

    Parameters
    ----------
    q : float
        Process noise variance (Q = q * I_2), controlling state evolution smoothness.
    r : float
        Observation noise variance (R = r), controlling measurement noise.
    w_init : np.ndarray, optional
        Initial state vector of shape (2,) or (2,1), representing [α_0, β_0].
        If None, initialized to zeros.
    p_init : np.ndarray, optional
        Initial state covariance matrix of shape (2, 2).
        If None, initialized to identity matrix.

    Attributes
    ----------
    w : np.ndarray
        Current state estimate vector (2x1).
    P : np.ndarray
        Current state covariance matrix (2x2).
    A : np.ndarray
        State transition matrix (identity).
    q : float
        Process noise variance.
    r : float
        Observation noise variance.
    """

    def __init__(self, q: float = 1e-5, r: float = 5e-3,
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
        """Perform the prediction step of the Kalman filter.

        Propagates the state estimate and covariance forward one time step.

        Updates
        -------
        self.w : np.ndarray
            Predicted state estimate.
        self.P : np.ndarray
            Predicted state covariance.
        """
        self.w = self.A @ self.w
        self.P = self.A @ self.P @ self.A.T + self.q * np.eye(2)

    def update(self, y_t: float, x_t: float):
        """
        Perform the update step of the Kalman filter with a new observation.

        Parameters
        ----------
        y_t : float
            Observation at time t (dependent variable).
        x_t : float
            Predictor value at time t (independent variable).

        Returns
        -------
        alpha_t : float
            Estimated intercept at time t.
        beta_t : float
            Estimated slope (hedge ratio) at time t.
        spread_t : float
            Residual spread at time t: y_t - (alpha_t + beta_t * x_t).
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


class KalmanFilterVecm:
    """
    Kalman filter for dynamic estimation of eigenvectors in a cointegration VECM.

    The observation model is:
        ε_t = e1_t * p1_t + e2_t * p2_t + ν_t,
    where ν_t ~ N(0, r), and the state vector w_t = [e1_t, e2_t]^T evolves as a random walk.

    This model enables tracking time-varying cointegration eigenvectors with explicit stationary noise.

    Parameters
    ----------
    q : float, optional
        Process noise variance (default 1e-6).
    r : float, optional
        Observation noise variance (default 2e-1).

    Attributes
    ----------
    w : np.ndarray
        Current eigenvector estimate (2x1).
    P : np.ndarray
        Current covariance matrix (2x2).
    A : np.ndarray
        State transition matrix (identity).
    q : float
        Process noise variance.
    r : float
        Observation noise variance.
    """

    def __init__(self, q=1e-6, r=2e-1):
        self.w = np.array([[-1,1]]).T
        self.P = np.eye(2)
        self.q = q
        self.r = r
        self.A = np.eye(2)

    def predict(self):
        """Perform the prediction step of the Kalman filter.

        Propagates the state estimate and covariance forward one time step.
        """
        self.w = self.A @ self.w
        self.P = self.A @ self.P @ self.A.T + self.q * np.eye(2)

    def update(self, p1_t: float, p2_t: float, eps_t: float):
        """
        Perform the update step of the Kalman filter with a new observation.

        Parameters
        ----------
        p1_t : float
            Price of asset 1 at time t.
        p2_t : float
            Price of asset 2 at time t.
        eps_t : float
            Observed residual (spread) at time t.

        Returns
        -------
        e1_t : float
            Estimated eigenvector component for asset 1 at time t.
        e2_t : float
            Estimated eigenvector component for asset 2 at time t.
        eps_hat_t : float
            Estimated residual at time t based on current eigenvectors.
        """
        # Vector de observación (precios)
        C_t = np.array([[p1_t, p2_t]])
        # Predicción del VECM
        eps_pred = float(C_t @ self.w)
        # Innovación
        e_t = eps_t - eps_pred
        # Varianza de la innovación
        S_t = C_t @ self.P @ C_t.T + self.r
        # Ganancia de Kalman
        K_t = self.P @ C_t.T / S_t
        # Actualización del estado
        self.w = self.w + K_t * e_t
        self.P = (np.eye(2) - K_t @ C_t) @ self.P

        e1_t, e2_t = float(self.w[0]), float(self.w[1])
        eps_hat_t = p1_t * e1_t + p2_t * e2_t
        return e1_t, e2_t, eps_hat_t



def run_kalman_on_pair(pair_df: pd.DataFrame, q: float = 1e-5, r: float = 1e-2):
    """
    Run the first Kalman filter (dynamic regression) on a cointegrated asset pair.

    This function estimates the time-varying hedge ratio (alpha, beta) and spread
    between two assets using a Kalman filter regression model.

    Parameters
    ----------
    pair_df : pd.DataFrame
        DataFrame containing price series of two assets as columns.
    q : float, optional
        Process noise variance for the Kalman filter (default 1e-5).
    r : float, optional
        Observation noise variance for the Kalman filter (default 1e-2).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the original prices, estimated alpha, beta,
        predicted values, and spread residuals. Columns include:
        [Asset_1, Asset_2, alpha, beta, y_pred, spread].

    Notes
    -----
    The result is also saved to 'data/kalman/kalman1_<Asset_1>_<Asset_2>.csv'.
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

def run_kalman2_vecm(kalman1_df: pd.DataFrame,
                           johansen_df: pd.DataFrame,
                           q: float = 1e-6,
                           r: float = 1e-2,
                           p0: float = 10.0,
                           theta: float = 1.8,
                           window: int = 100):
    """
    Run the second Kalman filter for dynamic VECM eigenvector estimation with stationary noise.

    This filter models the residual spread as a linear combination of asset prices with
    time-varying eigenvectors, incorporating explicit stationary noise to prevent collapse.

    Parameters
    ----------
    kalman1_df : pd.DataFrame
        Output DataFrame from the first Kalman filter containing alpha, beta, spread, etc.
    johansen_df : pd.DataFrame
        DataFrame containing initial eigenvectors for asset pairs with columns
        ['Asset_1', 'Asset_2', 'Eigenvector_1', 'Eigenvector_2'].
    q : float, optional
        Process noise variance (default 1e-6).
    r : float, optional
        Observation noise variance (default 1e-2).
    p0 : float, optional
        Initial state covariance scalar (default 10.0).
    theta : float, optional
        Z-score threshold for signal generation (default 1.8).
    window : int, optional
        Rolling window size for z-score normalization (default 100).

    Returns
    -------
    pd.DataFrame
        DataFrame extending kalman1_df with columns:
        ['v1_t', 'v2_t', 'vecm_t', 'z_t', 'signal_t'], representing dynamic eigenvectors,
        estimated VECM residual, normalized z-score, and trading signals.

    Notes
    -----
    The result is saved to 'data/kalman/kalman2_<Asset_1>_<Asset_2>.csv'.
    """
    # --- Identificación del par y precios ---
    asset1, asset2 = kalman1_df.columns[:2]
    p1 = kalman1_df[asset1].values.astype(float)
    p2 = kalman1_df[asset2].values.astype(float)
    n = len(p1)

    # --- Eigenvector inicial (Johansen) ---
    row = johansen_df[(johansen_df["Asset_1"] == asset1) &
                      (johansen_df["Asset_2"] == asset2)]
    if row.empty:
        raise ValueError(f"No se encontró eigenvector para {asset1}-{asset2} en johansen_df.")
    v1_0, v2_0 = row[["Eigenvector_1", "Eigenvector_2"]].values[0]
    w = np.array([[v1_0], [v2_0]])  # estado inicial

    # --- Inicialización de matrices ---
    P = np.eye(2) * p0
    Q = np.eye(2) * q
    R = r

    v1_list, v2_list, vecm_list = [], [], []

    # --- Bucle Kalman (modelo cointegrante con ruido estacionario) ---
    for t in range(n):
        # Predicción
        w_pred = w
        P_pred = P + Q

        # Observación real: epsilon_t (residuo del spread observado)
        # Para empezar, usamos el spread del Kalman 1 como proxy de ε_t
        epsilon_t = kalman1_df["spread"].values[t]
        C_t = np.array([[p1[t], p2[t]]])  # 1x2

        # Innovación: residuo entre observación real y estimada
        innovation = epsilon_t - float(C_t @ w_pred)
        S_t = float(C_t @ P_pred @ C_t.T + R)
        K_t = (P_pred @ C_t.T) / S_t

        # Actualización del estado
        w = w_pred + K_t * innovation
        P = (np.eye(2) - K_t @ C_t) @ P_pred

        # Guardar resultados
        v1_t, v2_t = float(w[0]), float(w[1])
        v1_list.append(v1_t)
        v2_list.append(v2_t)
        vecm_list.append(p1[t] * v1_t + p2[t] * v2_t)

    # --- Normalización rolling (Z-score del VECM) ---
    vecm = np.array(vecm_list)
    mu_t = pd.Series(vecm).rolling(window).mean().values
    sd_t = pd.Series(vecm).rolling(window).std().values
    z_t = (vecm - mu_t) / (sd_t + 1e-12)

    # --- Señales ---
    signal_t = np.where(z_t > theta, -1,
                 np.where(z_t < -theta, +1, 0))

    # --- Ensamblar salida ---
    df_res = kalman1_df.copy()
    df_res["v1_t"] = v1_list
    df_res["v2_t"] = v2_list
    df_res["vecm_t"] = vecm
    df_res["z_t"] = np.nan_to_num(z_t)
    df_res["signal_t"] = signal_t

    df_res = df_res[[asset1, asset2, 'alpha', 'beta', 'y_pred', 'spread',
                     'v1_t', 'v2_t', 'vecm_t', 'z_t', 'signal_t']]

    # --- Guardar ---
    output_path = f"data/kalman/kalman2_{asset1}_{asset2}.csv"
    df_res.to_csv(output_path, index=True)

    return df_res
