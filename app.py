from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

TABLES_DIR = Path("outputs/tables")
FIGURES_DIR = Path("outputs/figures")

st.set_page_config(page_title="Tech Stock Monte Carlo Simulation Dashboard", layout="wide")

COLUMN_NAMES = {
    "ticker": "Ticker",
    "initial_price": "Initial Price ($)",
    "expected_final_price": "Expected Price (1 Year)",
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


st.title("Tech Stock Monte Carlo Simulation Dashboard")
st.write(
    "This dashboard displays summary metrics and Monte Carlo simulation results for selected major technology stocks over a one-year horizon."
)

summary_file = TABLES_DIR / "simulation_summary.csv"
params_file = TABLES_DIR / "estimated_parameters.csv"

if not summary_file.exists() or not params_file.exists():
    st.error("Missing output files. Run `python3 main.py` first to generate the simulation results.")
    st.stop()

summary_df = pd.read_csv(summary_file)
params_df = pd.read_csv(params_file)

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

path_plot = FIGURES_DIR / f"{selected_ticker}_simulated_paths.png"
hist_plot = FIGURES_DIR / f"{selected_ticker}_final_price_histogram.png"

plot_col1, plot_col2 = st.columns(2)

with plot_col1:
    st.write("### Simulated Price Paths")
    if path_plot.exists():
        img = mpimg.imread(path_plot)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img)
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.warning(f"Missing plot: {path_plot.name}")

with plot_col2:
    st.write("### Final Price Distribution")
    if hist_plot.exists():
        img = mpimg.imread(hist_plot)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img)
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.warning(f"Missing plot: {hist_plot.name}")
