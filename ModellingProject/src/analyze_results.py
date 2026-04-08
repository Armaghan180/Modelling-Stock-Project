import numpy as np
import pandas as pd


def summarize_paths(paths: np.ndarray, initial_price: float) -> dict:
    """
    Calculate summary statistics from simulated paths.
    """
    final_prices = paths[:, -1]
    returns = (final_prices / initial_price) - 1.0

    summary = {
        "expected_final_price": float(np.mean(final_prices)),
        "median_final_price": float(np.median(final_prices)),
        "min_final_price": float(np.min(final_prices)),
        "max_final_price": float(np.max(final_prices)),
        "p05_final_price": float(np.percentile(final_prices, 5)),
        "p95_final_price": float(np.percentile(final_prices, 95)),
        "expected_return": float(np.mean(returns)),
        "probability_of_loss": float(np.mean(returns < 0))
    }

    return summary


def build_summary_table(prices: pd.DataFrame, simulations: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Build a results table for all tickers.
    """
    latest_prices = prices.groupby("ticker")["close"].last().to_dict()
    rows = []

    for ticker, paths in simulations.items():
        s0 = float(latest_prices[ticker])
        stats = summarize_paths(paths, s0)
        stats["ticker"] = ticker
        stats["initial_price"] = s0
        rows.append(stats)

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df[
        [
            "ticker",
            "initial_price",
            "expected_final_price",
            "median_final_price",
            "p05_final_price",
            "p95_final_price",
            "expected_return",
            "probability_of_loss",
            "min_final_price",
            "max_final_price",
        ]
    ]

    return summary_df.sort_values("expected_return", ascending=False).reset_index(drop=True)