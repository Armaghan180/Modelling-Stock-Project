import numpy as np
import pandas as pd


def add_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Add daily log returns for each ticker.
    """
    df = prices.copy()
    df["log_return"] = df.groupby("ticker")["close"].transform(lambda s: np.log(s / s.shift(1)))
    df = df.dropna(subset=["log_return"]).reset_index(drop=True)
    return df


def estimate_parameters(returns_df: pd.DataFrame, trading_days: int = 252) -> pd.DataFrame:
    """
    Estimate annualized drift and volatility from daily log returns.

    Returns columns:
    [ticker, mu_daily, sigma_daily, mu_annual, sigma_annual]
    """
    grouped = returns_df.groupby("ticker")["log_return"]

    mu_daily = grouped.mean()
    sigma_daily = grouped.std()

    params = pd.DataFrame({
        "mu_daily": mu_daily,
        "sigma_daily": sigma_daily,
        "mu_annual": mu_daily * trading_days,
        "sigma_annual": sigma_daily * np.sqrt(trading_days)
    }).reset_index()

    return params