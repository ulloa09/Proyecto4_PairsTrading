import numpy as np
import pandas as pd

class KalmanFilterSpread:
    """
    Kalman filter for estimating the equilibrium level (mean) of the spread.
    s_t = mu_t + v_t
    mu_t = mu_{t-1} + w_t
    """

    def __init__(self, q=1e-5, r=1e-3, mu_init=0.0, p_init=1.0):
        self.mu = mu_init
        self.P = p_init
        self.Q = q
        self.R = r

    def step(self, s_t):
        # Prediction
        mu_pred = self.mu
        P_pred = self.P + self.Q

        # Innovation
        e_t = s_t - mu_pred
        S_t = P_pred + self.R

        # Kalman gain
        K_t = P_pred / S_t

        # Update
        self.mu = mu_pred + K_t * e_t
        self.P = (1 - K_t) * P_pred

        return self.mu, np.sqrt(S_t), e_t


def run_kalman_spread(kalman1_pair: pd.DataFrame,
                      q=1e-5, r=1e-3,
                      save: bool = True,
                      path_prefix: str = "data/kalman/",
                      label: str | None = None):
    """
    Applies Kalman 2 on the spread estimated from Kalman 1 results
    to estimate dynamic mean and z-score.

    Parameters
    ----------
    kalman1_pair : pd.DataFrame
        DataFrame from run_kalman_on_pair(), must contain 'spread_t'.
    q, r : float
        Process and observation noise.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['spread_t', 'mu_t', 'sigma_t', 'z_t', 'pair_name']
    """

    if 'spread_t' not in kalman1_pair.columns:
        raise ValueError("Input DataFrame must contain column 'spread_t' from Kalman 1 output.")

    # Detect tickers automatically from columns (excluding Date and spread_t)
    tickers = [c for c in kalman1_pair.columns if c not in ['Date', 'spread_t', 'beta_t', 'alpha_t']]
    pair_name = f"{tickers[0]}_{tickers[1]}" if len(tickers) >= 2 else "UnknownPair"

    spread_series = kalman1_pair['spread_t']
    kf = KalmanFilterSpread(q=q, r=r)

    mu_list, sigma_list, z_list = [], [], []

    for s_t in spread_series:
        mu_t, sigma_t, e_t = kf.step(s_t)
        mu_list.append(mu_t)
        sigma_list.append(sigma_t)
        z_list.append(e_t / sigma_t)

    out = pd.DataFrame({
        "spread_t": spread_series.values,
        "mu_t": mu_list,
        "sigma_t": sigma_list,
        "z_t": z_list,
        "pair_name": pair_name
    }, index=kalman1_pair.index)

    # Save if needed
    if save:
        if label is None:
            filename = f"kalman2_{pair_name}.csv"
        else:
            filename = f"kalman2_{label}.csv"
        out.to_csv(f"{path_prefix}{filename}")
    print(f"💾 Kalman 2 results saved to: {path_prefix}{filename}")

    return out