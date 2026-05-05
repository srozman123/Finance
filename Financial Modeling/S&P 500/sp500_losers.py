"""
S&P 500 Biggest Losers
Fetches S&P 500 constituents and identifies the top 10 biggest losers
for 1-day, 1-week, and 1-month timeframes.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    import pandas as pd
    import requests
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install yfinance pandas requests")
    sys.exit(1)


def fetch_sp500_tickers() -> list[str]:
    """Scrape S&P 500 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; sp500-losers-script/1.0)"}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        tables = pd.read_html(response.text)
        df = tables[0]
        tickers = df["Symbol"].tolist()
        # yfinance uses '-' instead of '.' for some tickers (e.g. BRK.B -> BRK-B)
        tickers = [t.replace(".", "-") for t in tickers]
        return tickers
    except Exception as e:
        print(f"Failed to fetch S&P 500 tickers: {e}")
        sys.exit(1)


def fetch_price_data(tickers: list[str], period: str = "1mo") -> pd.DataFrame:
    """
    Download historical close prices for all tickers.
    Returns a DataFrame with dates as index and tickers as columns.
    """
    try:
        data = yf.download(
            tickers,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if data.empty:
            raise ValueError("No price data returned.")
        # Extract 'Close' prices
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]
        else:
            close = data[["Close"]]
        return close
    except Exception as e:
        print(f"Error downloading price data: {e}")
        sys.exit(1)


def compute_losers(close: pd.DataFrame, lookback_days: int, label: str) -> pd.DataFrame:
    """
    Given a close-price DataFrame, find the 10 biggest losers
    over the last `lookback_days` trading days.

    lookback_days=1  -> compare latest close vs previous close
    lookback_days=5  -> compare latest close vs close ~1 week ago
    lookback_days=21 -> compare latest close vs close ~1 month ago
    """
    # Drop tickers with insufficient history
    valid = close.dropna(axis=1, thresh=lookback_days + 1)

    if valid.shape[0] < lookback_days + 1:
        print(f"  Not enough trading days in data for '{label}' timeframe.")
        return pd.DataFrame()

    latest_price = valid.iloc[-1]
    past_price = valid.iloc[-(lookback_days + 1)]

    pct_change = ((latest_price - past_price) / past_price) * 100
    abs_change = latest_price - past_price

    result = pd.DataFrame({
        "Ticker": valid.columns,
        "Current Price": latest_price.values,
        "Abs Change": abs_change.values,
        "Pct Change": pct_change.values,
    })

    result = result.dropna(subset=["Pct Change"])
    result = result.sort_values("Pct Change").head(10).reset_index(drop=True)
    result.index += 1  # 1-based rank
    return result


def print_losers(df: pd.DataFrame, label: str) -> None:
    """Pretty-print the losers table to the terminal."""
    separator = "=" * 62
    print(f"\n{separator}")
    print(f"  TOP 10 BIGGEST LOSERS — {label}")
    print(separator)

    if df.empty:
        print("  No data available.")
        print(separator)
        return

    header = f"  {'Rank':<5} {'Ticker':<8} {'Price':>10} {'% Change':>10} {'$ Change':>10}"
    print(header)
    print("  " + "-" * 58)

    for rank, row in df.iterrows():
        pct = row["Pct Change"]
        abs_ch = row["Abs Change"]
        price = row["Current Price"]
        ticker = row["Ticker"]

        pct_str = f"{pct:+.2f}%"
        abs_str = f"{abs_ch:+.2f}"
        price_str = f"${price:.2f}"

        print(f"  {rank:<5} {ticker:<8} {price_str:>10} {pct_str:>10} {abs_str:>10}")

    print(separator)


def _run_sec_analysis(ticker: str) -> None:
    """Import sec_analyzer and run the full 10-K analysis for the given ticker."""
    sec_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "sec_10k_analyzer",
    )
    if sec_dir not in sys.path:
        sys.path.insert(0, sec_dir)

    try:
        from sec_analyzer import analyze_company, display_results
        import requests

        data = analyze_company(ticker)
        display_results(data)
    except ImportError as exc:
        print(f"  Could not import sec_analyzer: {exc}")
        print(f"  Expected location: {sec_dir}")
    except requests.HTTPError as exc:
        print(f"  HTTP error for {ticker}: {exc}")
    except ValueError as exc:
        print(f"  Data error for {ticker}: {exc}")
    except Exception as exc:
        print(f"  Unexpected error for {ticker}: {exc}")


def prompt_and_analyze(loser_frames: list[pd.DataFrame]) -> None:
    """
    Display an interactive numbered menu of all unique tickers from the loser
    tables and run the SEC 10-K analyzer for whichever one the user picks.
    """
    seen: set[str] = set()
    ticker_list: list[str] = []
    for df in loser_frames:
        if not df.empty:
            for t in df["Ticker"]:
                if t not in seen:
                    seen.add(t)
                    ticker_list.append(t)

    if not ticker_list:
        return

    separator = "=" * 62
    print(f"\n{separator}")
    print("  SEC 10-K ANALYZER — enter a number to deep-dive a company")
    print(separator)
    for i, t in enumerate(ticker_list, 1):
        print(f"  [{i:>2}]  {t}")
    print(f"\n  [ 0]  Exit")
    print(separator)

    while True:
        try:
            raw = input("\n  Select (number, ticker, or 0 to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not raw or raw == "0":
            break

        ticker = None
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(ticker_list):
                ticker = ticker_list[idx]
            else:
                print(f"  Please enter a number between 1 and {len(ticker_list)}.")
                continue
        else:
            upper = raw.upper()
            if upper in seen:
                ticker = upper
            else:
                print(f"  '{raw.upper()}' is not in the list above.")
                continue

        print(f"\n  Fetching SEC 10-K data for {ticker}...\n")
        _run_sec_analysis(ticker)


def main():
    print("\nFetching S&P 500 constituents...")
    tickers = fetch_sp500_tickers()
    print(f"Found {len(tickers)} tickers.")

    print("Downloading price data (last 1 month)...")
    close = fetch_price_data(tickers, period="1mo")
    print(f"Data loaded: {close.shape[0]} trading days x {close.shape[1]} tickers.\n")

    timeframes = [
        (1,  "1 DAY"),
        (5,  "1 WEEK"),
        (21, "1 MONTH"),
    ]

    loser_frames: list[pd.DataFrame] = []
    for lookback, label in timeframes:
        df = compute_losers(close, lookback, label)
        print_losers(df, label)
        loser_frames.append(df)

    print()
    prompt_and_analyze(loser_frames)


if __name__ == "__main__":
    main()
