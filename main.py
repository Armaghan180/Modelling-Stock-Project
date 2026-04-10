from pathlib import Path

from src.load_clean import load_prices_from_folder
from src.estimate_params import add_log_returns, estimate_parameters
from src.simulate_gbm import simulate_all_stocks
from src.analyze_results import build_summary_table
from src.plotting import plot_sample_paths, plot_final_price_histogram


TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"]

DATA_FOLDER = "data"
OUTPUT_TABLES = "outputs/tables"
OUTPUT_FIGURES = "outputs/figures"

SIMULATION_DAYS = 252
N_PATHS = 10000


def main():
    Path(OUTPUT_TABLES).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_FIGURES).mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    prices = load_prices_from_folder(DATA_FOLDER, TICKERS)

    print("Computing returns...")
    returns_df = add_log_returns(prices)

    print("Estimating parameters...")
    params_df = estimate_parameters(returns_df)

    params_df.to_csv(f"{OUTPUT_TABLES}/estimated_parameters.csv", index=False)

    print("Running Monte Carlo simulations...")
    simulations = simulate_all_stocks(
        prices=prices,
        params=params_df,
        days=SIMULATION_DAYS,
        n_paths=N_PATHS
    )

    print("Building summary table...")
    summary_df = build_summary_table(prices, simulations)
    summary_df.to_csv(f"{OUTPUT_TABLES}/simulation_summary.csv", index=False)

    print("\nSimulation Summary:")
    print(summary_df)

    print("\nGenerating plots...")
    for ticker, paths in simulations.items():
        plot_sample_paths(paths, ticker, OUTPUT_FIGURES)
        plot_final_price_histogram(paths, ticker, OUTPUT_FIGURES)

    print("\nDone.")
    print(f"Tables saved to: {OUTPUT_TABLES}")
    print(f"Figures saved to: {OUTPUT_FIGURES}")


if __name__ == "__main__":
    main()