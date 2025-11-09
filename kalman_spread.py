import numpy as np
import pandas as pd

class KalmanFilterVecm:
    """
    Kalman filter para la estimación dinámica de los eigenvectores (e1_t, e2_t)
    del modelo cointegrante:
        ε_t = e1_t * p1_t + e2_t * p2_t + ν_t
    """
    def __init__(self, q=1e-5, r=1e-2):
        self.w = np.zeros((2,1))
        self.P = np.eye(2)
        self.q = q
        self.r = r
        self.A = np.eye(2)

    def predict(self):
        self.w = self.A @ self.w
        self.P = self.A @ self.P @ self.A.T + self.q * np.eye(2)

    def update(self, p1_t: float, p2_t: float, eps_t: float):
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

def run_kalman2_vecm(kalman1_df: pd.DataFrame,
                           johansen_df: pd.DataFrame,
                           q: float = 1e-6,
                           r: float = 1e-2,
                           p0: float = 10.0,
                           theta: float = 1.8,
                           window: int = 100):
    """
    Segundo filtro (2D con ruido estacionario explícito):
    ε_t = [P1_t, P2_t]·v_t + ν_t,  con ν_t ~ N(0, r).
    Esto permite que el VECM no colapse a 0, manteniendo estacionariedad con ruido.

    Parámetros
    ----------
    kalman1_df : DataFrame
        Salida del primer filtro (alpha, beta, etc.)
    johansen_df : DataFrame
        Con eigenvectores iniciales (Eigenvector_1, Eigenvector_2).
    q, r : float
        Ruido de proceso y de observación.
    p0 : float
        Covarianza inicial del estado.
    theta : float
        Umbral de señalización con z-score.
    window : int
        Ventana para normalización rolling.

    Devuelve
    --------
    df_res : DataFrame con eigenvectores dinámicos, vecm_t, z_t, signal_t.
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