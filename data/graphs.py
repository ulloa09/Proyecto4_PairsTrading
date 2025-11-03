import pandas as pd
import matplotlib.pyplot as plt


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