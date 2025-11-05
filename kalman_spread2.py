# kalman_spread.py
# -*- coding: utf-8 -*-
"""
===============================================================================
Pairs Trading – Segundo Filtro (Señales) con baseline y VECM (versión simple)
===============================================================================

Modelo cointegrante (Johansen, rank=1)
--------------------------------------
Sea P_t = (P_{1,t}, P_{2,t})' el vector de precios. Con Johansen (r=1) se
obtiene el vector cointegrante β = (β1, β2)' tal que:
    ε_t = β1 P_{1,t} + β2 P_{2,t}  ~ I(0)

Baseline: S_t = E · P_t
-----------------------
Con el primer autovector de Johansen E = (e1, e2)' se define:
    S_t = e1 P_{1,t} + e2 P_{2,t}
Sobre S_t se calcula un z-score rolling de ventana 'window_z'.

VECM (z-score sin look-ahead)
-----------------------------
El z-score VECM se define a partir de ε_t:
    z_t^{(vecm)} = (ε_t - μ_ε) / σ_ε
donde μ_ε y σ_ε se estiman únicamente en la submuestra de train (índice que
se pasa desde main.py) y se aplican sin recalibrar fuera de muestra. Si no se
proporciona train_index, se estiman en tooooodo el sample tal cual.

Política de señales (versión simple, vectorizada)
-------------------------------------------------
Para cualquier canal con z_t:
- +1 si z_t < -θ
- -1 si z_t >  θ
-  0 en otro caso
(Se evita repintado al calcularse una sola vez por barra. No hay cooldown ni
cierre por cruce a 0 en esta versión simple; eso se puede aplicar fuera.)

Columnas de salida (CSV)
------------------------
['Date','Asset_1','Asset_2',
 'P1','P2','beta_t_est','spread_t',
 'E_dot_P_t','z_t','theta_t','signal_t',
 'z_vecm_t','theta_vecm_t','signal_vecm_t']

Notas
-----
- La detección de activos usa los dos primeros nombres de columna de kalman1_df.
- Los precios deben venir en prices_df con esos mismos nombres de columna.
- Guardado SIEMPRE en data/kalman/kalman2_{Asset_1}_{Asset_2}.csv
===============================================================================
"""

import os
import numpy as np
import pandas as pd


def _rolling_zscore(x: pd.Series, window: int) -> pd.Series:
    mu = x.rolling(window).mean()
    sd = x.rolling(window).std(ddof=1)
    return (x - mu) / sd


def _signals_from_z(z: pd.Series, theta: float) -> pd.Series:
    # Vectorizado: +1 si z<-theta, -1 si z>theta, 0 en otro caso
    sig = np.where(z < -theta, 1, np.where(z > theta, -1, 0))
    return pd.Series(sig, index=z.index)


def run_kalman_spread(
    kalman1_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    johansen_df: pd.DataFrame,
    theta_input: float = 1.8,
    window_z: int = 252,
    train_index: pd.Index | None = None,
    output_dir: str = "data/kalman",
) -> pd.DataFrame:
    """
    Genera señales de pares por dos rutas:
      (i) Baseline: S_t = E·P_t con z-score rolling(window_z)
     (ii) VECM:     z-score de ε_t = β1 P1 + β2 P2 con μ,σ estimados en train

    Parámetros
    ----------
    kalman1_df : DataFrame con columnas [Asset_1, Asset_2, 'beta_t_est', 'spread_t']
                 (en tu proyecto los dos primeros nombres de columna son los tickers)
    prices_df  : DataFrame con columnas de precios de esos dos tickers
    johansen_df: DataFrame con columnas ['Asset_1','Asset_2','beta1','beta2','e1','e2']
    theta_input: Umbral θ para señales (+/- θ)
    window_z   : Ventana rolling para z-score baseline
    train_index: Índice (fechas) de train ya definido en main.py. Si None,
                 μ_ε y σ_ε se estiman en toodo el sample.
    output_dir : Carpeta de salida (siempre data/kalman por defecto)

    Returns
    -------
    DataFrame final y guarda CSV en data/kalman/kalman2_{Asset_1}_{Asset_2}.csv
    """

    # 1) Detectar activos como los dos primeros nombres de columna en kalman1_df
    asset_1, asset_2 = kalman1_df.columns[:2].tolist()

    # 2) Preparar índice de fechas (usar el índice actual tal cual)
    idx = kalman1_df.index if 'Date' not in kalman1_df.columns else pd.to_datetime(kalman1_df['Date'])
    kalman1 = kalman1_df.copy()
    kalman1.index = idx

    # 3) Precios alineados
    P = prices_df[[asset_1, asset_2]].copy()
    P.index = pd.to_datetime(P.index)
    P = P.sort_index().ffill().bfill()
    # Reindexar al universo de kalman1
    P = P.reindex(kalman1.index).ffill().bfill()
    P.columns = ['P1', 'P2']

    # 4) Extraer β y E del Johansen y del primer filtro (ajustado a tus columnas)
    asset_1, asset_2 = kalman1_df.columns[:2]

    # Encontrar fila correspondiente en johansen_results
    row = johansen_df[
        (johansen_df['Asset_1'] == asset_1) &
        (johansen_df['Asset_2'] == asset_2)
        ].iloc[0]

    # En tu archivo, los eigenvectors vienen como 'Eigenvector_1' y 'Eigenvector_2'
    e1, e2 = float(row['Eigenvector_1']), float(row['Eigenvector_2'])

    # Como no hay β en el CSV, usamos los hedge ratios del primer filtro
    # Tomamos el promedio de beta_t_est como proxy de β2/β1
    beta_ratio = kalman1_df['beta_t_est'].mean()
    beta1, beta2 = 1.0, -beta_ratio

    # 5) Baseline: S_t = E·P_t y z rolling
    S_t = e1 * P['P1'] + e2 * P['P2']
    z_t = _rolling_zscore(S_t, window_z)
    signal_t = _signals_from_z(z_t, theta_input)

    # 6) VECM: ε_t = β1 P1 + β2 P2; z con μ,σ de train (si se pasa)
    epsilon = beta1 * P['P1'] + beta2 * P['P2']
    if train_index is not None:
        mu_eps = epsilon.loc[train_index].mean()
        sd_eps = epsilon.loc[train_index].std(ddof=1)
    else:
        mu_eps = epsilon.mean()
        sd_eps = epsilon.std(ddof=1)
    z_vecm_t = (epsilon - mu_eps) / sd_eps
    signal_vecm_t = _signals_from_z(z_vecm_t, theta_input)

    # 7) Ensamble de salida (sin validaciones adicionales)
    out = pd.DataFrame(index=P.index)
    out['Date'] = out.index
    out['Asset_1'] = asset_1
    out['Asset_2'] = asset_2
    out['P1'] = P['P1']
    out['P2'] = P['P2']
    out['beta_t_est'] = kalman1['beta_t_est']
    out['spread_t'] = kalman1['spread_t']
    out['E_dot_P_t'] = S_t
    out['z_t'] = z_t
    out['theta_t'] = theta_input
    out['signal_t'] = signal_t
    out['z_vecm_t'] = z_vecm_t
    out['theta_vecm_t'] = theta_input
    out['signal_vecm_t'] = signal_vecm_t

    # 8) Guardado SIEMPRE en data/kalman/
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"kalman2_{asset_1}_{asset_2}.csv")
    out.to_csv(csv_path, index=False)

    # 9) Resumen simple en consola (validación rápida)
    n_long_base = int((signal_t == 1).sum())
    n_short_base = int((signal_t == -1).sum())
    n_sig_base = int((signal_t != 0).sum())
    n_long_vecm = int((signal_vecm_t == 1).sum())
    n_short_vecm = int((signal_vecm_t == -1).sum())
    n_sig_vecm = int((signal_vecm_t != 0).sum())
    pct_sig_base = 100.0 * n_sig_base / len(out)
    pct_sig_vecm = 100.0 * n_sig_vecm / len(out)

    print(f"[{asset_1}-{asset_2}] guardado en {csv_path}")
    print(f"Baseline  -> longs: {n_long_base} | shorts: {n_short_base} | %barras con señal: {pct_sig_base:.2f}%")
    print(f"VECM      -> longs: {n_long_vecm} | shorts: {n_short_vecm} | %barras con señal: {pct_sig_vecm:.2f}%")

    return out