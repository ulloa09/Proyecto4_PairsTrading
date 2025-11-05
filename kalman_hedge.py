import numpy as np
import pandas as pd


class KalmanFilterReg:
    """
    Kalman filter for a dynamic linear regression:
        y_t = (w0_t) + (w1_t) * x_t + v_t
    with state evolution:
        w_t = w_{t-1} + w_noise
    This matches the style used in class.
    """
    def __init__(self, q: float = 1e-5, r: float = 1e-2,
                 w_init = None,
                 p_init = None):
        # Initial parameters (w0, w1)
        if w_init is None:
            # same idea as notebook: start close to the true but not exact
            self.w = np.array([0.0, 0.0]).reshape(-1, 1)
        else:
            self.w = w_init.reshape(-1, 1)

        # State transition (random walk)
        self.A = np.eye(2)

        # Process noise (how much I let betas move)
        self.Q = np.eye(2) * q

        # Observation noise (how noisy is y_t)
        self.R = np.array([[r]])

        # Error covariance
        if p_init is None:
            self.P = np.eye(2) * 10.0   # big uncertainty at start
        else:
            self.P = p_init

    def predict(self):
        """
        Prediction step: w_{t|t-1}, P_{t|t-1}
        """
        self.w = self.A @ self.w
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, x_t: float, y_t: float):
        """
        Update step with one observation (x_t, y_t)
        """
        # Observation matrix C_t = [1  x_t]
        C = np.array([[1.0, x_t]])  # shape (1,2)

        # Innovation covariance
        S = C @ self.P @ C.T + self.R  # shape (1,1)

        # Kalman gain
        K = self.P @ C.T @ np.linalg.inv(S)  # shape (2,1)

        # Update state
        # y_t - C w_{t|t-1}
        innovation = y_t - (C @ self.w)[0, 0]
        self.w = self.w + K * innovation

        # Update covariance
        self.P = (np.eye(2) - K @ C) @ self.P

    @property
    def params(self):
        # Return w0_t, w1_t
        return self.w[0, 0], self.w[1, 0]


def run_kalman_on_pair(df_pair: pd.DataFrame,
                       q: float = 1e-5,
                       r: float = 1e-2,
                       save: bool = True,
                       path_prefix: str = "data/kalman/",
                       label: str | None = None) -> pd.DataFrame:
    """
    Runs the classroom-style Kalman filter on a 2-column pair DataFrame.

    Parameters
    ----------
    df_pair : pd.DataFrame
        DataFrame with Date index and exactly 2 price columns, e.g. ['KO', 'DUK'].
    q : float
        Process noise scale (how fast betas can move).
    r : float
        Observation noise scale.
    save : bool
        If True, saves to CSV as 'kalman1_<asset1>_<asset2>.csv'
    path_prefix : str
        Folder to save the output.
    label : str | None
        If you want to force a custom name; if None it's built from columns.

    Returns
    -------
    pd.DataFrame
        Original pair + beta0_t, beta1_t, spread_t
    """

    # Make a safe copy
    df = df_pair.copy()

    # Ensure datetime index
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")

    # We expect exactly two assets here
    asset1, asset2 = df.columns[:2]

    y_vals = df[asset1].values
    x_vals = df[asset2].values

    # Initialize Kalman as in class
    kf = KalmanFilterReg(q=q, r=r)

    beta0_list = []
    beta1_list = []
    spread_list = []

    for y_t, x_t in zip(y_vals, x_vals):
        # 1. predict
        kf.predict()

        # 2. update with current observation
        kf.update(x_t, y_t)

        # 3. store params
        b0, b1 = kf.params
        beta0_list.append(b0)
        beta1_list.append(b1)

        # current spread using dynamic beta
        spread_list.append(y_t - (b0 + b1 * x_t))

    # Build compact output
    out = pd.DataFrame({
        asset1: df[asset1].values,
        asset2: df[asset2].values,
        "beta_t_est": beta1_list,
        "spread_t": spread_list
    }, index=df.index)

    # Save if needed
    if save:
        if label is None:
            filename = f"kalman1_{asset1}_{asset2}.csv"
        else:
            filename = f"kalman1_{label}.csv"
        out.to_csv(f"{path_prefix}{filename}")
        print(f"💾 Kalman results saved to: {path_prefix}{filename}")

    return out