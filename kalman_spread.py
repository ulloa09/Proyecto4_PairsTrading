import numpy as np
import pandas as pd

from kalman_hedge import KalmanFilterReg

def run_kalman2_vecm(kalman1_df: pd.DataFrame,
                            johansen_df: pd.DataFrame,
                            q: float = 1e-6,
                            r: float = 1e-2,
                            p0: float = 10.0,
                            theta: float = 1.8,
                            window: int = 100):
    """
    Segundo filtro (anclado): v1_t = 1 y se estima gamma_t = v2_t con Kalman escalar
    imponiendo 0 ≈ P1_t + gamma_t * P2_t. Luego normaliza VECM con rolling window
    y genera señales {-1,0,+1} por umbral |z_t| > theta.

    Parámetros
    ----------
    kalman1_df : DataFrame
        Salida del primer filtro con columnas [Asset_1, Asset_2, alpha, beta, y_pred, spread].
    johansen_df : DataFrame
        Debe contener ['Asset_1','Asset_2','Eigenvector_1','Eigenvector_2'] para inicializar gamma_0.
    q, r : float
        Ruido de proceso (suavidad de gamma_t) y de observación (ajuste de la medición).
    p0 : float
        Varianza inicial del estado gamma_0.
    theta : float
        Umbral para el z-score (señales).
    window : int
        Ventana para media y desviación móviles.

    Devuelve
    --------
    df_res : DataFrame
        Columnas: [Asset_1, Asset_2, alpha, beta, y_pred, spread,
                   v1_t, v2_t, vecm_t, z_t, signal_t]
        y guarda en data/kalman/kalman2_<Asset_1>_<Asset_2>.csv
    """

    # --- Identificación del par y precios ---
    asset1, asset2 = kalman1_df.columns[:2]
    p1 = kalman1_df[asset1].values.astype(float)
    p2 = kalman1_df[asset2].values.astype(float)

    # --- Eigenvector inicial (Johansen) para gamma_0 = v2/v1 ---
    row = johansen_df[(johansen_df["Asset_1"] == asset1) &
                      (johansen_df["Asset_2"] == asset2)]
    if row.empty:
        raise ValueError(f"No se encontró eigenvector para {asset1}-{asset2} en johansen_df.")
    v1_0, v2_0 = row[["Eigenvector_1", "Eigenvector_2"]].values[0]
    denom = v1_0 if abs(v1_0) > 1e-12 else (1e-12 if v1_0 == 0 else np.sign(v1_0)*1e-12)
    gamma = float(v2_0 / denom)  # estado inicial (v1_t ≡ 1)

    # --- Inicialización Kalman escalar para gamma_t ---
    P = float(p0)  # varianza del estado

    gamma_list = []
    vecm_list  = []

    # Bucle Kalman: modelo observación  y_t = -P1_t  ≈  gamma_t * P2_t
    #   => innovación = y_t - H_t * gamma_pred  con H_t = P2_t
    for y_t, H_t in zip(-p1, p2):
        # Predict
        gamma_pred = gamma
        P_pred = P + q

        # Update
        S_t = H_t * P_pred * H_t + r
        K_t = (P_pred * H_t) / (S_t + 1e-12)
        innovation = y_t - H_t * gamma_pred

        gamma = gamma_pred + K_t * innovation
        P = (1.0 - K_t * H_t) * P_pred

        gamma_list.append(float(gamma))
        vecm_list.append(float(p1[len(gamma_list)-1] + gamma * H_t))

    # --- Normalización por ventana móvil ---
    vecm = np.array(vecm_list)
    mu_t = pd.Series(vecm).rolling(window).mean().values
    sd_t = pd.Series(vecm).rolling(window).std().values
    z_t = (vecm - mu_t) / (sd_t + 1e-12)

    # Señales (long cuando está “barato”, short cuando “caro”)
    signal_t = np.where(z_t >  theta, -1,
                 np.where(z_t < -theta,  +1, 0))

    # Rellena los primeros window-1 NaN de z con 0 para no generar señales espurias
    z_t = np.nan_to_num(z_t, nan=0.0)
    signal_t = np.where(np.isnan(mu_t) | np.isnan(sd_t), 0, signal_t)

    # --- Salida: conserva Kalman1 y añade VECM + señales ---
    df_res = kalman1_df.copy()
    df_res["v1_t"]     = 1.0
    df_res["v2_t"]     = gamma_list
    df_res["vecm_t"]   = vecm
    df_res["z_t"]      = z_t
    df_res["signal_t"] = signal_t

    df_res = df_res[[asset1, asset2, 'alpha', 'beta', 'y_pred', 'spread',
                     'v1_t', 'v2_t', 'vecm_t', 'z_t', 'signal_t']]

    # Guardar (mismo estilo que Kalman 1)
    output_path = f"data/kalman/kalman2_{asset1}_{asset2}.csv"
    df_res.to_csv(output_path, index=True)

    return df_res