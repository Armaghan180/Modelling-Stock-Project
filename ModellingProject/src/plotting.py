from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_sample_paths(paths: np.ndarray, ticker: str, output_folder: str, n_display: int = 50) -> None:
    """
    Plot a subset of simulated stock price paths.
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for i in range(min(n_display, paths.shape[0])):
        plt.plot(paths[i], linewidth=0.8)

    plt.title(f"Monte Carlo Simulated Price Paths - {ticker}")
    plt.xlabel("Trading Day")
    plt.ylabel("Stock Price")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_folder) / f"{ticker}_simulated_paths.png", dpi=300)
    plt.close()


def plot_final_price_histogram(paths: np.ndarray, ticker: str, output_folder: str) -> None:
    """
    Plot histogram of final simulated prices.
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    final_prices = paths[:, -1]

    plt.figure(figsize=(10, 6))
    plt.hist(final_prices, bins=40, edgecolor="black")
    plt.title(f"Distribution of Final Simulated Prices - {ticker}")
    plt.xlabel("Final Simulated Price")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(output_folder) / f"{ticker}_final_price_histogram.png", dpi=300)
    plt.close()