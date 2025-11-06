import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import alpha


def plot_splits(train_df, test_df, val_df):
    plt.figure(figsize=(10,4))
    plt.plot(train_df.index, train_df.iloc[:, 0], label='Train')
    plt.plot(test_df.index, test_df.iloc[:, 0], label='Test')
    plt.plot(val_df.index, val_df.iloc[:, 0], label='Validation')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_hedge_ratios(kalman_pair1: pd.DataFrame,
                      kalman_pair2: pd.DataFrame | None = None,
                      title: str | None = None):
    """
    Grafica la evolución del hedge ratio (βₜ) en el tiempo para uno o dos pares de activos.

    Parámetros
    ----------
    kalman_pair1 : pd.DataFrame
        DataFrame de resultados del primer filtro de Kalman con columnas
        [asset1, asset2, alpha, beta, y_pred, spread].
    kalman_pair2 : pd.DataFrame, opcional
        Segundo DataFrame (otro par) con la misma estructura.
        Si no se proporciona, solo se grafica el primero.
    title : str, opcional
        Título personalizado. Si no se indica, se genera automáticamente.

    Retorna
    -------
    matplotlib.figure.Figure
        Objeto de figura generado.
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    # --- Primer par ---
    df1 = kalman_pair1.copy()
    asset1_a, asset2_a = df1.columns[:2]
    label1 = f"βₜ ({asset1_a}-{asset2_a})"
    ax.plot(df1.index, df1["beta"], label=label1, linewidth=1.8, color="royalblue")

    # --- Segundo par (opcional) ---
    if kalman_pair2 is not None:
        df2 = kalman_pair2.copy()
        asset1_b, asset2_b = df2.columns[:2]
        label2 = f"βₜ ({asset1_b}-{asset2_b})"
        ax.plot(df2.index, df2["beta"], label=label2, linewidth=1.8, color="darkorange")

    # --- Personalización ---
    if title is None:
        if kalman_pair2 is None:
            title = f"Hedge Ratio (βₜ) — {asset1_a}-{asset2_a}"
        else:
            title = f"Hedge Ratios (βₜ) — {asset1_a}-{asset2_a} vs {asset1_b}-{asset2_b}"

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Fecha", fontsize=12)
    ax.set_ylabel("βₜ (hedge ratio dinámico)", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()

    return fig


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

def plot_spreads(pair1, pair2):
    plt.figure(figsize=(12, 6))
    asset1_p1, asset2_p2 = pair1.columns[0], pair1.columns[1]
    asset1_p2, asset2_p1 = pair2.columns[0], pair2.columns[1]

    plt.plot(pair1.index, pair1['spread'], label=f'Spread{asset1_p1}-{asset2_p1}', color='black', alpha=0.2)
    plt.plot(pair2.index, pair2['spread'], label=f'Spread{asset1_p2}-{asset2_p2}', color='darkgreen', alpha=0.7)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel('Fecha')
    plt.legend()
    plt.show()


def plot_kalman_fits(df_kalman1: pd.DataFrame,
                     df_kalman2: pd.DataFrame | None = None,
                     title: str | None = None):
    """
    Grafica la comparación entre los precios observados y estimados (y_pred)
    de uno o dos pares de activos procesados con el filtro de Kalman.

    Parámetros
    ----------
    df_kalman1 : pd.DataFrame
        DataFrame devuelto por run_kalman_on_pair para el primer par.
    df_kalman2 : pd.DataFrame, opcional
        Segundo DataFrame devuelto por run_kalman_on_pair (otro par).
    title : str, opcional
        Título personalizado del gráfico. Si no se especifica, se genera automáticamente.

    Retorna
    -------
    matplotlib.figure.Figure
        Objeto de figura generado.
    """

    # --- Primer par ---
    asset1_a, asset2_a = df_kalman1.columns[:2]
    label_obs_1 = f"{asset1_a} (observado)"
    label_pred_1 = f"{asset1_a} (estimado | {asset2_a})"

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(df_kalman1.index, df_kalman1[asset1_a],
            label=label_obs_1, color='steelblue', linewidth=1.5)
    ax.plot(df_kalman1.index, df_kalman1['y_pred'],
            label=label_pred_1, color='orange', linewidth=1.3, linestyle='--')

    # --- Segundo par (opcional) ---
    if df_kalman2 is not None:
        asset1_b, asset2_b = df_kalman2.columns[:2]
        label_obs_2 = f"{asset1_b} (observado)"
        label_pred_2 = f"{asset1_b} (estimado | {asset2_b})"

        ax.plot(df_kalman2.index, df_kalman2[asset1_b],
                label=label_obs_2, color='seagreen', linewidth=1.5)
        ax.plot(df_kalman2.index, df_kalman2['y_pred'],
                label=label_pred_2, color='darkred', linewidth=1.3, linestyle='--')

    # --- Personalización general ---
    if title is None:
        if df_kalman2 is None:
            title = f"{asset1_a} vs estimado | Predictor: {asset2_a}"
        else:
            title = f"Comparación de fits Kalman — {asset1_a}-{asset2_a} y {asset1_b}-{asset2_b}"

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel("Fecha", fontsize=11)
    ax.set_ylabel("Precio", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.show()

    return fig