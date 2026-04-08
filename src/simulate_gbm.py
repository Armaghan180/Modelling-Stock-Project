import numpy as np
import pandas as pd


def simulate_gbm_paths(
    s0: float,
    mu: float,
    sigma: float,
    days: int = 252,
    n_paths: int = 10000,
    seed: int = 42
) -> np.ndarray:
    """
    Simulate stock price paths using Geometric Brownian Motion.

    Formula:
    S(t+1) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

    Parameters:
    - s0: initial stock price
    - mu: annual drift
    - sigma: annual volatility
    - days: number of trading days to simulate
    - n_paths: number of Monte Carlo paths
    - seed: random seed

    Returns:
    - NumPy array of shape (n_paths, days + 1)
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0

    z = rng.standard_normal((n_paths, days))
    increments = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z

    log_paths = np.cumsum(increments, axis=1)
    paths = s0 * np.exp(log_paths)

    # Add initial price as day 0
    initial_column = np.full((n_paths, 1), s0)
    paths = np.hstack([initial_column, paths])

    return paths


def simulate_all_stocks(
    prices: pd.DataFrame,
    params: pd.DataFrame,
    days: int = 252,
    n_paths: int = 10000
) -> dict[str, np.ndarray]:
    """
    Simulate all stocks in the dataset.

    Returns:
    - dict of {ticker: simulated_paths}
    """
    latest_prices = prices.groupby("ticker")["close"].last().to_dict()
    param_map = params.set_index("ticker").to_dict("index")

    simulations = {}

    for ticker, s0 in latest_prices.items():
        mu = float(param_map[ticker]["mu_annual"])
        sigma = float(param_map[ticker]["sigma_annual"])

        simulations[ticker] = simulate_gbm_paths(
            s0=s0,
            mu=mu,
            sigma=sigma,
            days=days,
            n_paths=n_paths,
            seed=abs(hash(ticker)) % 100000
        )

    return simulations