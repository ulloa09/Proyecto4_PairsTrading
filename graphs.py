import pandas as pd
import matplotlib.pyplot as plt

def plot_splits(train_df, test_df, val_df):
    plt.figure(figsize=(10,4))
    plt.plot(train_df.index, train_df.iloc[:, 0], label='Train')
    plt.plot(test_df.index, test_df.iloc[:, 0], label='Test')
    plt.plot(val_df.index, val_df.iloc[:, 0], label='Validation')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_hedge_ratio(kalman_pair: pd.DataFrame, title: str | None = None, save: bool = False, path_prefix: str = "data/"):
    """
    Plots the dynamic hedge ratio (beta_t_est) over time from a Kalman results DataFrame.

    Parameters
    ----------
    kalman_pair : pd.DataFrame
        DataFrame with datetime index and columns [asset1, asset2, beta_t_est, spread_t].
    title : str, optional
        Custom title for the plot. If None, derived from asset names.
    save : bool, optional
        If True, saves the figure as a PNG file (default=False).
    path_prefix : str, optional
        Folder to save the PNG (default='data/').

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object.
    """

    df = kalman_pair.copy()

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")

    # Detect asset names
    assets = [c for c in df.columns if c not in ["beta_t_est", "spread_t"]]
    if len(assets) >= 2:
        asset1, asset2 = assets[:2]
        pair_name = f"{asset1}-{asset2}"
    else:
        pair_name = "Pair"

    # Title
    if title is None:
        title = f"Hedge Ratio (βₜ) Over Time — {pair_name}"

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df["beta_t_est"], color="royalblue", linewidth=1.8, label="βₜ (hedge ratio)")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("βₜ (hedge ratio)", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend()

    # Save
    if save:
        filename = f"{path_prefix}hedge_ratio_{pair_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"💾 Saved plot to: {filename}")

    plt.show()
    return fig



def plot_zscore_with_theta(kalman2_df: pd.DataFrame, title: str | None = None, save: bool = False):
    """
    Grafica el z-score a lo largo del tiempo con las bandas ±theta seleccionadas.

    Parámetros
    ----------
    kalman2_df : pd.DataFrame
        Resultado del filtro Kalman 2. Debe contener columnas ['z_t', 'theta_t'].
    title : str, opcional
        Título del gráfico.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(kalman2_df.index, kalman2_df['z_t'], label='Z-score', color='blue')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)

    theta = kalman2_df['theta_t'].iloc[0]  # es constante (theta_input)
    plt.axhline(theta, color='red', linestyle='--', label=f'+θ = {theta}')
    plt.axhline(-theta, color='red', linestyle='--', label=f'-θ = {theta}')

    plt.title(title or f"Z-score y Umbrales ±θ ({theta})")
    plt.xlabel('Fecha')
    plt.ylabel('Z-score normalizado')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.show()


def plot_signals_on_zscore(kalman2_df, title=None):
    """
    Muestra el z-score a lo largo del tiempo y marca las señales long/short generadas.
    """
    plt.figure(figsize=(12, 6))

    # Línea del z-score
    plt.plot(kalman2_df.index, kalman2_df['z_t'], label='Z-score', color='steelblue', alpha=0.8)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)

    # Bandas de theta
    theta = kalman2_df['theta_t'].iloc[0]
    plt.axhline(theta, color='red', linestyle='--', label=f'+θ = {theta}')
    plt.axhline(-theta, color='red', linestyle='--', label=f'-θ = {theta}')

    # Señales: +1 = long (verde), -1 = short (rojo)
    longs = kalman2_df[kalman2_df['signal_t'] == 1]
    shorts = kalman2_df[kalman2_df['signal_t'] == -1]

    plt.scatter(longs.index, longs['z_t'], marker='^', color='green', label='Compra (Long)', s=60, alpha=0.8)
    plt.scatter(shorts.index, shorts['z_t'], marker='v', color='darkred', label='Venta (Short)', s=60, alpha=0.8)

    plt.title(title or f"Z-score y señales generadas (θ = {theta})")
    plt.xlabel('Fecha')
    plt.ylabel('Z-score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.show()


def plot_signals_on_spread(kalman2_df: pd.DataFrame, title: str | None = None):
    """
    Grafica el spread y marca las señales de compra y venta generadas.
    """
    a, b = kalman2_df.columns[0], kalman2_df.columns[1]
    title = f"Señales sobre spread (Par{a},{b})"
    plt.figure(figsize=(12, 6))
    plt.plot(kalman2_df.index, kalman2_df['spread_t'], label='Spread', color='gray', alpha=0.7)
    longs = kalman2_df[kalman2_df['signal_t'] == 1]
    shorts = kalman2_df[kalman2_df['signal_t'] == -1]
    plt.scatter(longs.index, longs['spread_t'], color='green', marker='^', label='Compra (Long)', s=60)
    plt.scatter(shorts.index, shorts['spread_t'], color='red', marker='v', label='Venta (Short)', s=60)
    plt.title(title or "Spread con señales generadas")
    plt.xlabel('Fecha')
    plt.ylabel('Spread')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()