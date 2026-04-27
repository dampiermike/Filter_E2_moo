#!/usr/bin/env python3
"""
Daily end-of-day Yahoo Finance fetch for SQQQ and ^VIX.

Downloads the last 5 trading days, saves a date-stamped snapshot to:
  <script_dir>/data/csv/daily/

Then merges any new dates into the history files:
  <script_dir>/data/csv/history/sqqq-from-yahoo.csv
  <script_dir>/data/csv/history/vix-from-yahoo.csv

Usage:
    python3 fetch_yahoo_daily.py
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
HIST_DIR    = SCRIPT_DIR / 'data' / 'csv' / 'history'
DAILY_DIR   = SCRIPT_DIR / 'data' / 'csv' / 'daily'
DAILY_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_FILES = {
    'SQQQ': HIST_DIR / 'sqqq-from-yahoo.csv',
    '^VIX': HIST_DIR / 'vix-from-yahoo.csv',
}

DATE_FMT = '%-m/%-d/%y'   # M/D/YY — matches existing file format (no zero-padding)


# ── Helpers ───────────────────────────────────────────────────────────────────

def download_ticker(symbol: str) -> pd.DataFrame:
    """Download last 5 days from Yahoo and return a flat DataFrame."""
    raw = yf.download(symbol, period='5d', auto_adjust=False, progress=False)

    # Flatten multi-level columns produced by yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [col[0] for col in raw.columns]

    raw = raw.reset_index()
    raw['Date'] = pd.to_datetime(raw['Date'])

    # Ensure Adj Close column exists (VIX and leveraged ETFs may omit it)
    if 'Adj Close' not in raw.columns:
        raw['Adj Close'] = raw['Close']

    # Reorder to match history file column order
    cols = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    raw = raw[[c for c in cols if c in raw.columns]].copy()
    return raw


def load_history(path: Path) -> pd.DataFrame:
    """Load existing history CSV, parse dates."""
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def merge_new_rows(history: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Append rows from new that are not already in history. Returns (merged_df, added_count)."""
    existing_dates = set(history['Date'])
    fresh = new[~new['Date'].isin(existing_dates)].copy()
    if fresh.empty:
        return history, 0
    merged = pd.concat([history, fresh], ignore_index=True).sort_values('Date').reset_index(drop=True)
    return merged, len(fresh)


def save_history(df: pd.DataFrame, path: Path):
    """Write history back to CSV using M/D/YY date format (no zero-padding)."""
    out = df.copy()
    out['Date'] = out['Date'].dt.strftime(DATE_FMT)
    out.to_csv(path, index=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    today = date.today()
    date_str = today.strftime('%Y-%m-%d')
    print(f"fetch_yahoo_daily  {date_str}\n")

    for symbol, hist_path in HISTORY_FILES.items():
        print(f"── {symbol} ─────────────────────────────────")

        # 1. Download
        df_new = download_ticker(symbol)
        if df_new.empty:
            print(f"  ⚠️  No data returned for {symbol} — skipping.")
            continue
        print(f"  Downloaded {len(df_new)} rows  ({df_new['Date'].min().date()} → {df_new['Date'].max().date()})")

        # 2. Save daily snapshot
        safe_sym   = symbol.replace('^', '')
        daily_path = DAILY_DIR / f"{safe_sym.lower()}_{date_str}.csv"
        snap = df_new.copy()
        snap['Date'] = snap['Date'].dt.strftime(DATE_FMT)
        snap.to_csv(daily_path, index=False)
        print(f"  Saved snapshot → {daily_path.name}")

        # 3. Merge into history
        df_hist = load_history(hist_path)
        df_merged, added = merge_new_rows(df_hist, df_new)

        if added == 0:
            print(f"  History already up to date — no new rows added.")
        else:
            save_history(df_merged, hist_path)
            print(f"  ✅  Added {added} new row(s) to {hist_path.name}  "
                  f"(now {len(df_merged)} rows, latest: {df_merged['Date'].max().date()})")

        print()


if __name__ == '__main__':
    main()
