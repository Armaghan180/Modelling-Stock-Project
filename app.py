from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from src.load_clean import load_prices_from_folder
from src.estimate_params import add_log_returns, estimate_parameters
from src.simulate_gbm import simulate_all_stocks
from src.analyze_results import build_summary_table

DATA_FOLDER = "data"
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META"]
N_PATHS = 10000

st.set_page_config(page_title="Tech Stock Simulation Dashboard", layout="wide")

COLUMN_NAMES = {
    "ticker": "Ticker",
    "initial_price": "Initial Price ($)",
    "expected_final_price": "Expected Price",
    "median_final_price": "Median Price ($)",
    "p05_final_price": "5th Percentile ($)",
    "p95_final_price": "95th Percentile ($)",
    "expected_return": "Expected Return",
    "probability_of_loss": "Probability of Loss",
    "min_final_price": "Minimum Price ($)",
    "max_final_price": "Maximum Price ($)",
    "mu_daily": "Daily Return (μ)",
    "sigma_daily": "Daily Volatility (σ)",
    "mu_annual": "Annual Return (μ)",
    "sigma_annual": "Annual Volatility (σ)"
}

summary_display_cols = [
    "ticker",
    "initial_price",
    "expected_final_price",
    "median_final_price",
    "p05_final_price",
    "p95_final_price",
    "expected_return",
    "probability_of_loss",
    "min_final_price",
    "max_final_price"
]

stock_summary_display_cols = [
    "ticker",
    "initial_price",
    "expected_final_price",
    "median_final_price",
    "p05_final_price",
    "p95_final_price",
    "expected_return",
    "probability_of_loss"
]

params_display_cols = [
    "ticker",
    "mu_daily",
    "sigma_daily",
    "mu_annual",
    "sigma_annual"
]


@st.cache_data
def load_and_prepare_data():
    prices = load_prices_from_folder(DATA_FOLDER, TICKERS)
    returns_df = add_log_returns(prices)
    params_df = estimate_parameters(returns_df)
    return prices, params_df


@st.cache_data
def run_simulation(simulation_days: int):
    prices, params_df = load_and_prepare_data()
    simulations = simulate_all_stocks(
        prices=prices,
        params=params_df,
        days=simulation_days,
        n_paths=N_PATHS
    )
    summary_df = build_summary_table(prices, simulations)
    return prices, params_df, simulations, summary_df


def format_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    price_cols = [
        "initial_price",
        "expected_final_price",
        "median_final_price",
        "p05_final_price",
        "p95_final_price",
        "min_final_price",
        "max_final_price"
    ]
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"{x:,.2f}")

    percent_cols = ["expected_return", "probability_of_loss"]
    for col in percent_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"{x:.2%}")

    return df.rename(columns=COLUMN_NAMES)


def format_params_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "mu_daily" in df.columns:
        df["mu_daily"] = df["mu_daily"].map(lambda x: f"{x:.2%}")
    if "sigma_daily" in df.columns:
        df["sigma_daily"] = df["sigma_daily"].map(lambda x: f"{x:.2%}")
    if "mu_annual" in df.columns:
        df["mu_annual"] = df["mu_annual"].map(lambda x: f"{x:.2%}")
    if "sigma_annual" in df.columns:
        df["sigma_annual"] = df["sigma_annual"].map(lambda x: f"{x:.2%}")

    return df.rename(columns=COLUMN_NAMES)


def plot_simulated_paths(paths: np.ndarray, ticker: str, n_display: int = 50):
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(min(n_display, paths.shape[0])):
        ax.plot(paths[i], linewidth=0.9)
    ax.set_title(f"Monte Carlo Simulated Price Paths - {ticker}")
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Stock Price")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_final_price_distribution(paths: np.ndarray, ticker: str):
    final_prices = paths[:, -1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(final_prices, bins=40, edgecolor="black")
    ax.set_title(f"Distribution of Final Simulated Prices - {ticker}")
    ax.set_xlabel("Final Simulated Price")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


st.title("Tech Stock Monte Carlo Simulation Dashboard")
st.write(
    "This dashboard displays summary metrics and Monte Carlo simulation results for selected major technology stocks."
)

# --- Simulation Controls ---
control_col1, control_col2 = st.columns(2)

with control_col1:
    horizon_option = st.selectbox(
        "Simulation Horizon",
        ["1 Year", "2 Years", "3 Years"]
    )

HORIZON_MAP = {
    "1 Year": 252,
    "2 Years": 504,
    "3 Years": 756
}

simulation_days = HORIZON_MAP[horizon_option]

with control_col2:
    st.metric("Trading Days", simulation_days)

prices, params_df, simulations, summary_df = run_simulation(simulation_days)

tickers = summary_df["ticker"].tolist()
selected_ticker = st.selectbox("Select a stock", tickers)

st.subheader("Simulation Summary Table")
summary_display = format_summary_df(summary_df[summary_display_cols])
st.dataframe(summary_display, use_container_width=True, hide_index=True)

st.subheader(f"Details for {selected_ticker}")

col1, col2 = st.columns(2)

with col1:
    st.write("### Summary Metrics")
    stock_summary = summary_df[summary_df["ticker"] == selected_ticker][stock_summary_display_cols]
    stock_summary_display = format_summary_df(stock_summary)
    st.dataframe(stock_summary_display, use_container_width=True, hide_index=True)

with col2:
    st.write("### Estimated Parameters")
    stock_params = params_df[params_df["ticker"] == selected_ticker][params_display_cols]
    stock_params_display = format_params_df(stock_params)
    st.dataframe(stock_params_display, use_container_width=True, hide_index=True)

st.subheader(f"Plots for {selected_ticker}")

selected_paths = simulations[selected_ticker]

plot_col1, plot_col2 = st.columns(2)

with plot_col1:
    st.write("### Simulated Price Paths")
    fig1 = plot_simulated_paths(selected_paths, selected_ticker, n_display=50)
    st.pyplot(fig1)

with plot_col2:
    st.write("### Final Price Distribution")
    fig2 = plot_final_price_distribution(selected_paths, selected_ticker)
    st.pyplot(fig2)

