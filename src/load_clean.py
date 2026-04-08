from pathlib import Path
import pandas as pd


POSSIBLE_DATE_COLS = ["Date", "date", "Datetime", "datetime", "timestamp"]
POSSIBLE_TICKER_COLS = ["Ticker", "ticker", "Symbol", "symbol", "stock_symbol"]
POSSIBLE_CLOSE_COLS = ["Close", "close", "Adj Close", "adj_close", "AdjClose"]


def find_column(columns, possible_names):
    """Return the first matching column name from a list of possibilities."""
    for col in possible_names:
        if col in columns:
            return col
    return None


def load_prices_from_folder(data_folder: str, selected_tickers: list[str]) -> pd.DataFrame:
    """
    Load stock prices from CSV files inside a folder.

    Supports:
    1. One combined CSV with ticker column
    2. Multiple CSV files, one per ticker

    Returns standardized dataframe with columns:
    [date, ticker, close]
    """
    folder = Path(data_folder)
    csv_files = list(folder.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in folder: {data_folder}")

    all_frames = []

    for file_path in csv_files:
        df = pd.read_csv(file_path)
        cols = list(df.columns)

        date_col = find_column(cols, POSSIBLE_DATE_COLS)
        ticker_col = find_column(cols, POSSIBLE_TICKER_COLS)
        close_col = find_column(cols, POSSIBLE_CLOSE_COLS)

        if date_col is None or close_col is None:
            continue

        # If ticker column doesn't exist, try to infer ticker from filename
        inferred_ticker = None
        if ticker_col is None:
            file_upper = file_path.stem.upper()
            for ticker in selected_tickers:
                if ticker.upper() in file_upper:
                    inferred_ticker = ticker.upper()
                    break

        use_cols = [date_col, close_col]
        if ticker_col is not None:
            use_cols.append(ticker_col)

        temp = df[use_cols].copy()
        temp.rename(columns={date_col: "date", close_col: "close"}, inplace=True)

        if ticker_col is not None:
            temp.rename(columns={ticker_col: "ticker"}, inplace=True)
        else:
            if inferred_ticker is None:
                continue
            temp["ticker"] = inferred_ticker

        all_frames.append(temp)

    if not all_frames:
        raise ValueError("Could not parse any CSV files. Check your column names.")

    prices = pd.concat(all_frames, ignore_index=True)

    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")

    prices = prices.dropna(subset=["date", "ticker", "close"])
    prices = prices[prices["ticker"].isin([t.upper() for t in selected_tickers])]
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    return prices