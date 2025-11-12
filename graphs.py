import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_normalized_prices(df: pd.DataFrame, title: str = "Precios normalizados"):
    """
    Grafica los precios normalizados de dos activos para visualizar cruces.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con índice de fechas y dos columnas (precios de los activos).
    title : str, opcional
        Título del gráfico.
    """

    # Asegurar que el índice sea datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)

    # Normalizar precios: (precio / precio inicial)
    normalized_df = df / df.iloc[0]

    # Graficar
    plt.figure(figsize=(10, 5))
    for col in normalized_df.columns:
        plt.plot(normalized_df.index, normalized_df[col], label=col)

    plt.title(title)
    plt.xlabel("Fecha")
    plt.ylabel("Precio normalizado (base 1)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_dynamic_eigenvectors(df):
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["e1_hat"], label="v₁ₜ", color='teal')
    plt.plot(df.index, df["e2_hat"], label="v₂ₜ", color='orange')
    plt.title("Eigenvectores dinámicos estimados (Kalman 2)")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def plot_vecm_signals(results_df: pd.DataFrame,
                      entry_long_idx: list[int],
                      entry_short_idx: list[int],
                      exit_idx: list[int],
                      theta: float):
    """
    Grafica el VECM normalizado y marca las señales de entrada/salida generadas durante el backtest.

    Parámetros
    ----------
    results_df : pd.DataFrame
        DataFrame con columna 'vecm_norm' e índice temporal.
    entry_long_idx : list[int]
        Índices (enteros) donde se abrieron operaciones LONG.
    entry_short_idx : list[int]
        Índices (enteros) donde se abrieron operaciones SHORT.
    exit_idx : list[int]
        Índices donde se cerraron operaciones.
    theta : float
        Umbral usado para la generación de señales.
    """

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(results_df.index, results_df["vecm_norm"],
            color="steelblue", lw=1.4, label="VECM Normalizado")

    # --- Líneas de umbral ---
    ax.axhline(theta,  color="gray", linestyle="--", lw=1, label=f"+θ = {theta}")
    ax.axhline(-theta, color="gray", linestyle="--", lw=1, label=f"-θ = {-theta}")
    ax.axhline(0, color="black", linestyle=":", lw=1)

    # --- Mapeo de índices a fechas ---
    def _idx_to_index_vals(idx_list, df_index):
        return [df_index[i] for i in idx_list if 0 <= i < len(df_index)]

    x_long  = _idx_to_index_vals(entry_long_idx,  results_df.index)
    x_short = _idx_to_index_vals(entry_short_idx, results_df.index)
    x_exit  = _idx_to_index_vals(exit_idx,        results_df.index)

    y_long  = [results_df["vecm_norm"].iloc[i] for i in entry_long_idx  if 0 <= i < len(results_df)]
    y_short = [results_df["vecm_norm"].iloc[i] for i in entry_short_idx if 0 <= i < len(results_df)]
    y_exit  = [results_df["vecm_norm"].iloc[i] for i in exit_idx        if 0 <= i < len(results_df)]

    # --- Puntos de señal ---
    ax.scatter(x_long,  y_long,  color="red",  marker="v", s=90, label="Entrada LONG")
    ax.scatter(x_short, y_short, color="green",    marker="^", s=90, label="Entrada SHORT")
    ax.scatter(x_exit,  y_exit,  color="black",  marker="x", s=70, label="Cierre")

    # --- Personalización ---
    ax.set_title("Señales de Trading — VECM Normalizado (Kalman 2)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Fecha", fontsize=11)
    ax.set_ylabel("VECM Normalizado (z-score)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()

    try:
        plt.gcf().autofmt_xdate()
    except Exception:
        pass

    plt.show()


def plot_spread_evolution(results_df: pd.DataFrame, asset1: str, asset2: str):
    """
    Grafica la evolución temporal del spread dinámico estimado por el primer Filtro de Kalman.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame con al menos la columna 'spread' y un índice temporal.
    asset1 : str
        Nombre del primer activo (p1, el activo dependiente).
    asset2 : str
        Nombre del segundo activo (p2, el activo independiente).
    """

    if "spread" not in results_df.columns:
        raise ValueError("La columna 'spread' no existe en results_df.")

    spread_series = results_df["spread"]

    plt.figure(figsize=(12, 5))
    plt.plot(spread_series, color="steelblue", linewidth=1.8, label="Spread estimado (Kalman 1)", alpha = 0.6)

    # Media y desviación estándar
    mean_spread = spread_series.mean()
    std_spread = spread_series.std()

    plt.axhline(mean_spread, color="orange", linestyle="--", linewidth=1.5, label="Media del spread")
    plt.fill_between(
        spread_series.index,
        mean_spread + std_spread,
        mean_spread - std_spread,
        color="orange",
        alpha=0.15,
        label="±1 desviación estándar"
    )
    # --- Bandas de ±2 desviaciones estándar ---
    plt.fill_between(
        spread_series.index,
        mean_spread + 2 * std_spread,
        mean_spread - 2 * std_spread,
        color="red",
        alpha=0.08,
        label="±2 desviaciones estándar"
    )

    plt.title(f"Evolución del Spread Dinámico - {asset1} vs {asset2}", fontsize=13, weight='bold')
    plt.xlabel("Fecha")
    plt.ylabel("Spread (P1 - β_t * P2)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_portfolio_evolution(portfolio_series: pd.Series, split_ratios=(0.6, 0.2, 0.2)):
    """
    Grafica la evolución del valor del portafolio, mostrando visualmente las fases
    de entrenamiento, prueba y validación (60/20/20).

    Parameters
    ----------
    portfolio_series : pd.Series
        Serie temporal con el valor total del portafolio.
    split_ratios : tuple
        Porcentaje de división temporal (por defecto (0.6, 0.2, 0.2)).
    """

    n = len(portfolio_series)
    train_end = int(n * split_ratios[0])
    test_end = int(n * (split_ratios[0] + split_ratios[1]))

    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_series, color="steelblue", linewidth=1.8, label="Valor del Portafolio")

    # Zonas de fondo
    plt.axvspan(portfolio_series.index[0], portfolio_series.index[train_end],
                color='green', alpha=0.08, label='Train (60%)')
    plt.axvspan(portfolio_series.index[train_end], portfolio_series.index[test_end],
                color='gold', alpha=0.12, label='Test (20%)')
    plt.axvspan(portfolio_series.index[test_end], portfolio_series.index[-1],
                color='red', alpha=0.08, label='Validation (20%)')

    # Gráfica y formato
    plt.title("Evolución del Valor del Portafolio (Train / Test / Validation)", fontsize=13, weight='bold')
    plt.xlabel("Fecha")
    plt.ylabel("Valor del Portafolio ($)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_spread_vs_vecm(results_df: pd.DataFrame):
    """
    Grafica la comparación entre el spread estimado por el Kalman 1
    y el VECM normalizado estimado por el Kalman 2.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame que contiene las columnas 'spread' y 'vecm_norm'.
    """
    # Validar columnas requeridas
    required_cols = ["spread", "vecm_norm"]
    for col in required_cols:
        if col not in results_df.columns:
            raise ValueError(f"Falta la columna requerida: '{col}' en results_df")

    plt.figure(figsize=(12, 5))
    plt.plot(results_df.index, results_df["spread"],
             color="steelblue", linewidth=1.8, label="Spread (Kalman 1)")
    plt.plot(results_df.index, results_df["vecm_norm"],
             color="orange", linewidth=1.5, alpha=0.8, label="VECM Normalizado (Kalman 2)")

    plt.title("Comparación: Spread vs VECM Normalizado", fontsize=13, weight='bold')
    plt.xlabel("Fecha")
    plt.ylabel("Valor estimado")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_hedge_ratio_evolution(results_df: pd.DataFrame):
    """
    Grafica la evolución temporal de la razón de cobertura (hedge ratio) estimada por el Filtro de Kalman 1.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame con al menos la columna 'hedge_ratio' y un índice temporal.
    """

    if "hedge_ratio" not in results_df.columns:
        raise ValueError("La columna 'hedge_ratio' no existe en results_df.")

    plt.figure(figsize=(12, 5))
    plt.plot(results_df.index, results_df["hedge_ratio"],
             color="purple", linewidth=1.8, label="Razón de cobertura βₜ (Kalman 1)", alpha=0.8)

    # Media y bandas ±1σ para visualizar estabilidad
    mean_beta = results_df["hedge_ratio"].mean()
    std_beta = results_df["hedge_ratio"].std()
    plt.axhline(mean_beta, color="orange", linestyle="--", linewidth=1.5, label="Media de βₜ")

    plt.title("Evolución de la Razón de Cobertura Dinámica (βₜ)", fontsize=13, weight='bold')
    plt.xlabel("Fecha")
    plt.ylabel("Hedge Ratio (βₜ)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_trade_returns_distribution(pnl_history: list[float]):
    """
    Grafica la distribución de rendimientos por operación (histograma y estadísticos).

    Parameters
    ----------
    pnl_history : list[float]
        Lista con los rendimientos o ganancias/pérdidas por operación cerrada.
    """

    if not pnl_history:
        print("⚠️ No hay operaciones cerradas para analizar la distribución de rendimientos.")
        return



    # Convertir a array
    pnl_array = np.array(pnl_history)
    mean_pnl = np.mean(pnl_array)
    median_pnl = np.median(pnl_array)
    std_pnl = np.std(pnl_array)
    win_rate = np.sum(pnl_array > 0) / len(pnl_array)

    # Crear histograma + densidad
    plt.figure(figsize=(10, 5))
    sns.histplot(pnl_array, bins=30, kde=True, color="steelblue", alpha=0.7)

    # Líneas de referencia
    plt.axvline(mean_pnl, color="orange", linestyle="--", lw=1.5, label=f"Media = {mean_pnl:.2f}")
    plt.axvline(median_pnl, color="green", linestyle="--", lw=1.5, label=f"Mediana = {median_pnl:.2f}")
    plt.axvline(0, color="black", linestyle=":", lw=1.2)

    # Título y ejes
    plt.title("Distribución de Rendimientos por Operación", fontsize=13, weight="bold")
    plt.xlabel("PnL por operación ($)")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Mostrar estadísticas resumidas
    print("\n=== Estadísticas de Trading ===")
    print(f"📊 Número de operaciones: {len(pnl_array)}")
    print(f"✅ Tasa de aciertos (Win Rate): {win_rate*100:.2f}%")
    print(f"💵 Media PnL: {mean_pnl:.2f}")
    print(f"📈 Mediana PnL: {median_pnl:.2f}")
    print(f"📉 Desviación estándar: {std_pnl:.2f}")
    print(f"📊 Pérdida promedio: {np.mean(pnl_array[pnl_array<0]):.2f}")
    print(f"📊 Ganancia promedio: {np.mean(pnl_array[pnl_array>0]):.2f}")