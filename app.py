from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

TABLES_DIR = Path("outputs/tables")
FIGURES_DIR = Path("outputs/figures")

st.set_page_config(page_title="Tech Stock Simulation Dashboard", layout="wide")

st.title("Tech Stock Monte Carlo Simulation Dashboard")
st.write(
    "This dashboard displays summary metrics and simulation plots for the selected technology stocks."
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
st.dataframe(summary_df, use_container_width=True)

st.subheader(f"Details for {selected_ticker}")

col1, col2 = st.columns(2)

with col1:
    st.write("### Summary Metrics")
    stock_summary = summary_df[summary_df["ticker"] == selected_ticker]
    st.dataframe(stock_summary, use_container_width=True)

with col2:
    st.write("### Estimated Parameters")
    stock_params = params_df[params_df["ticker"] == selected_ticker]
    st.dataframe(stock_params, use_container_width=True)

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

st.subheader("Quick Observations")
st.write(
    """
- The dashboard shows the simulated future stock price outcomes based on Geometric Brownian Motion.
- The summary table includes expected return, probability of loss, and percentile-based risk measures.
- The plots help visualize uncertainty in future stock price behavior.
"""
)