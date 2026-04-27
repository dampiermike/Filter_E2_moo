"""
Dampier Filter E² — MOO (next-day Market-on-Open) variant.

Same indicator and state-machine logic as filter_e2.py, but trades execute at
the NEXT day's open instead of same-day market-on-close:

  - Signals and state machine: unchanged. All indicators are computed from
    close prices and rolling close-to-close returns, exactly per spec §3.
  - Position arrays are shifted forward by 1 day. The decision made at
    close[t] becomes effective at open[t+1].
  - Daily portfolio P&L uses open-to-open returns. Synthetic TQQQ/SQQQ open
    series are constructed via parallel backward chains anchored on the real
    open on the corresponding split dates (TQQQ 2010-03-31, SQQQ 2010-02-11),
    using QQQ open-to-open returns and the same decay constants.
  - Real-period SQQQ open is split-adjusted as adj_open = Open * (Adj Close /
    Close), since Yahoo's Open column is not split-adjusted.
  - The E² ATR post-processing filter still uses ATR(14) computed from close-
    based H/L/C and is applied to the shifted SQQQ position array.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))


def _find_data_dir() -> str:
    """Locate the directory containing the CSVs.

    Checks, in order: HERE/data/csv/history, HERE itself, then walks upward up
    to 6 levels (also checking each level's data/csv/history subfolder).
    """
    candidates = [
        os.path.join(HERE, "data", "csv", "history"),
        HERE,
        os.path.dirname(HERE),
    ]
    cur = HERE
    for _ in range(6):
        cur = os.path.dirname(cur)
        candidates.append(cur)
        candidates.append(os.path.join(cur, "data", "csv", "history"))
    for c in candidates:
        if os.path.exists(os.path.join(c, "qqq-from-vv.csv")):
            return c
    raise FileNotFoundError(
        "Could not locate qqq-from-vv.csv near " + HERE
    )


DATA = _find_data_dir()

QQQ_F = os.path.join(DATA, "qqq-from-vv.csv")
TQQQ_F = os.path.join(DATA, "tqqq-from-vv.csv")
SQQQ_F = os.path.join(DATA, "sqqq-from-yahoo.csv")
VV_F = os.path.join(DATA, "vectorvest-views-w3place-precision.csv")
VIX_F = os.path.join(DATA, "vix-from-yahoo.csv")

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
SPLIT_TQQQ = pd.Timestamp("2010-03-31")
SPLIT_SQQQ = pd.Timestamp("2010-02-11")
DECAY_T = (0.0086 + 0.010) / 252  # 0.00007381
DECAY_S = (0.0095 + 0.010) / 252  # 0.00007738
WARMUP = 80
CP6_START = pd.Timestamp("2000-01-03")


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y")
    return df.sort_values("Date").reset_index(drop=True)


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------
def ms(close: pd.Series, ret: pd.Series, n: int) -> pd.Series:
    return close.pct_change(n) / (ret.rolling(n).std() * np.sqrt(n))


def rma(ret: pd.Series, n: int) -> pd.Series:
    pos = ret.clip(lower=0).rolling(n).sum()
    neg = ret.clip(upper=0).abs().rolling(n).sum()
    return pos / (pos + neg)


def atr_pct(H: pd.Series, L: pd.Series, C: pd.Series, n: int = 14) -> pd.Series:
    prev_c = C.shift(1)
    tr = pd.concat([H - L, (H - prev_c).abs(), (L - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean() / C * 100.0


def rt_v6(C: pd.Series, H: pd.Series, L: pd.Series) -> pd.Series:
    """RT v6 formula for synthetic TQQQ RT (spec section 2)."""
    high59 = H.rolling(59).max()
    low59 = L.rolling(59).min()
    mid59 = (high59 + low59) / 2.0
    range59 = high59 - low59
    ma5 = C.rolling(5).mean()
    ma50 = C.rolling(50).mean()
    f1 = C / mid59
    f7 = (C - low59) / range59
    z = (
        5.9942 * (C / mid59)
        + 1.6014 * (C / ma5)
        + 0.5108 * (C / ma50)
        - 2.5453 * f7
        + 1.9755 * (f1 * f7)
        - 7.6562
    )
    return 2.0288 / (1.0 + np.exp(-z))


# --------------------------------------------------------------------------
# Synthetic TQQQ construction (spec §0.2)
# --------------------------------------------------------------------------
def _backward_chain(prices: np.ndarray, returns: np.ndarray, decay: float) -> np.ndarray:
    """Backward chain: synth[i] = synth[i+1] / (1 + 3*returns[i] - decay).

    Anchor must be pre-loaded into prices[-1]. NaN returns produce flat carry.
    """
    n = len(prices)
    out = prices.copy()
    for i in range(n - 2, -1, -1):
        r = returns[i]
        if np.isnan(r):
            out[i] = out[i + 1]
        else:
            out[i] = out[i + 1] / (1.0 + 3.0 * r - decay)
    return out


def build_synthetic_tqqq(
    qqq: pd.DataFrame, anchor_close: float, anchor_open: float
) -> pd.DataFrame:
    """Backward-chain synthetic TQQQ close AND open from real anchors."""
    qqq = qqq.copy()
    qqq["qqq_ret"] = qqq["Close"].pct_change()
    qqq["qqq_open_ret"] = qqq["Open"].pct_change()
    qqq_pre = qqq[qqq["Date"] <= SPLIT_TQQQ].copy().reset_index(drop=True)
    n = len(qqq_pre)

    # Close chain (spec §0.2)
    synth_close = np.full(n, np.nan)
    synth_close[-1] = anchor_close
    synth_close = _backward_chain(synth_close, qqq_pre["qqq_ret"].values, DECAY_T)
    qqq_pre["synth_close"] = synth_close

    # Open chain (parallel construction using QQQ open-to-open returns)
    synth_open = np.full(n, np.nan)
    synth_open[-1] = anchor_open
    synth_open = _backward_chain(
        synth_open, qqq_pre["qqq_open_ret"].values, DECAY_T
    )
    qqq_pre["synth_open"] = synth_open

    # Synthetic H/L from QQQ intraday range (spec §0.3)
    range_pct = (qqq_pre["High"] - qqq_pre["Low"]) / qqq_pre["Close"]
    qqq_pre["th"] = qqq_pre["synth_close"] * (1.0 + 1.5 * range_pct)
    qqq_pre["tl"] = qqq_pre["synth_close"] * (1.0 - 1.5 * range_pct)

    # Drop the split_date row (real data takes over there)
    return qqq_pre[qqq_pre["Date"] < SPLIT_TQQQ].copy()


# --------------------------------------------------------------------------
# State machine (spec §11)
# --------------------------------------------------------------------------
def run_state_machine(df_w: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    N = len(df_w)
    tp_arr = np.zeros(N, dtype=np.int8)
    sp_arr = np.zeros(N, dtype=np.int8)
    tw_arr = np.ones(N)
    path_arr: list[str] = [""] * N

    ms20 = df_w["ms20"].values
    ms8 = df_w["ms8"].values
    rma20 = df_w["rma20"].values
    rma12 = df_w["rma12"].values
    rma10 = df_w["rma10"].values
    rma5 = df_w["rma5"].values
    atr14 = df_w["atr14"].values
    rt5 = df_w["rt5"].values
    bsr5 = df_w["bsr5"].values
    vix_a = df_w["VIX"].values

    tp = 0
    sp = 0
    for i in range(N):
        P1 = (
            ms20[i] >= 0.50
            and rma20[i] >= 0.48
            and rt5[i] >= 0.85
            and bsr5[i] > 0.30
            and atr14[i] < 9.0
        )
        P2 = (
            ms8[i] >= 0.30
            and rma10[i] >= 0.52
            and atr14[i] < 8.0
            and bsr5[i] > 0.30
        )
        P3 = (
            rma12[i] >= 0.58
            and rt5[i] >= 0.98
            and atr14[i] < 6.5
            and bsr5[i] > 0.30
        )
        in_tqqq = P1 or P2 or P3
        t_exit = (
            ms20[i] <= -0.05
            or rma20[i] <= 0.38
            or atr14[i] >= 9.0
            or bsr5[i] <= 0.30
            or ms8[i] <= -0.15
        )
        s_entry = (not in_tqqq) and ms20[i] <= -0.15 and vix_a[i] >= 20.0
        s_exit = (
            ms20[i] >= -0.05 or vix_a[i] < 20.0 or in_tqqq or rma5[i] >= 0.52
        )

        # TQQQ — exit wins on conflict (runs after entry)
        if in_tqqq:
            tp = 1
            sp = 0
        if t_exit:
            tp = 0

        # SQQQ — exit wins on conflict
        if not sp and s_entry:
            sp = 1
        if sp and s_exit:
            sp = 0

        if tp == 1 and atr14[i] > 0:
            tw = min(1.0, 4.0 / atr14[i])
        else:
            tw = 1.0

        tp_arr[i] = tp
        sp_arr[i] = sp
        tw_arr[i] = tw
        parts = []
        if P1:
            parts.append("P1")
        if P2:
            parts.append("P2")
        if P3:
            parts.append("P3")
        path_arr[i] = "+".join(parts)

    return tp_arr, sp_arr, tw_arr, path_arr


# --------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------
def annualised_sharpe(rets: np.ndarray) -> float:
    if len(rets) == 0:
        return float("nan")
    return rets.mean() / rets.std(ddof=1) * np.sqrt(252)


def max_drawdown(rets: np.ndarray) -> float:
    if len(rets) == 0:
        return float("nan")
    eq = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def fmt_money(x: float) -> str:
    return f"${x:,.0f}"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    # Load data
    qqq = load_csv(QQQ_F)
    tqqq_real = load_csv(TQQQ_F)
    sqqq = load_csv(SQQQ_F)
    vv = load_csv(VV_F).rename(columns={"BS Ratio": "BSR"})
    vix = load_csv(VIX_F).rename(columns={"Close": "VIX"})

    anchor_close = float(tqqq_real["Close"].iloc[0])
    anchor_open = float(tqqq_real["Open"].iloc[0])

    # Synthetic TQQQ (close + open chains)
    synth = build_synthetic_tqqq(qqq, anchor_close, anchor_open)

    # full_t = synth + real (now also carries Open)
    full_t = pd.DataFrame(
        {
            "Date": pd.concat([synth["Date"], tqqq_real["Date"]]).values,
            "O": pd.concat([synth["synth_open"], tqqq_real["Open"]]).values,
            "H": pd.concat([synth["th"], tqqq_real["High"]]).values,
            "L": pd.concat([synth["tl"], tqqq_real["Low"]]).values,
            "C": pd.concat([synth["synth_close"], tqqq_real["Close"]]).values,
        }
    ).sort_values("Date").reset_index(drop=True)
    # Close-to-close return (used for indicators per spec §3)
    full_t["tqqq_ret"] = full_t["C"].pct_change()
    # Open-to-open return (used for actual P&L under MOO execution)
    full_t["tqqq_oo_ret"] = full_t["O"].pct_change()

    # RT: compute v6 on full_t, then blend real VV RT for dates >= 2010-03-31
    full_t["RT_calc"] = rt_v6(full_t["C"], full_t["H"], full_t["L"])
    rt_map = dict(zip(tqqq_real["Date"], tqqq_real["RT"]))
    full_t["RT"] = [rt_map.get(d, c) for d, c in zip(full_t["Date"], full_t["RT_calc"])]

    # Synthetic SQQQ maps (pre-split). Close-to-close map kept for completeness;
    # open-to-open is what actually drives MOO P&L during the synthetic period.
    qqq_with_ret = qqq.copy()
    qqq_with_ret["qqq_ret"] = qqq_with_ret["Close"].pct_change()
    qqq_with_ret["qqq_open_ret"] = qqq_with_ret["Open"].pct_change()
    synth_s_map = {
        row["Date"]: -3.0 * row["qqq_ret"] - DECAY_S
        for _, row in qqq_with_ret[qqq_with_ret["Date"] < SPLIT_SQQQ].iterrows()
        if not pd.isna(row["qqq_ret"])
    }
    synth_s_oo_map = {
        row["Date"]: -3.0 * row["qqq_open_ret"] - DECAY_S
        for _, row in qqq_with_ret[qqq_with_ret["Date"] < SPLIT_SQQQ].iterrows()
        if not pd.isna(row["qqq_open_ret"])
    }

    # Real SQQQ returns: close-to-close from Adj Close (kept available), and
    # open-to-open from split-adjusted open: adj_open = Open * (Adj Close / Close).
    sqqq["sqqq_ret"] = sqqq["Adj Close"].pct_change()
    sqqq.loc[sqqq["Date"] == SPLIT_SQQQ, "sqqq_ret"] = synth_s_map.get(SPLIT_SQQQ, 0.0)
    sqqq["adj_open"] = sqqq["Open"] * (sqqq["Adj Close"] / sqqq["Close"])
    sqqq["sqqq_oo_ret"] = sqqq["adj_open"].pct_change()
    sqqq.loc[sqqq["Date"] == SPLIT_SQQQ, "sqqq_oo_ret"] = synth_s_oo_map.get(
        SPLIT_SQQQ, 0.0
    )
    sqqq_map = dict(zip(sqqq["Date"], sqqq["sqqq_ret"]))
    sqqq_oo_map = dict(zip(sqqq["Date"], sqqq["sqqq_oo_ret"]))

    # Merge VV / VIX
    df = full_t.merge(vv[["Date", "BSR"]], on="Date", how="left")
    df = df.merge(vix[["Date", "VIX"]], on="Date", how="left")
    df[["BSR", "VIX"]] = df[["BSR", "VIX"]].ffill()
    df["sqqq_ret"] = df["Date"].map(
        lambda d: sqqq_map.get(d, synth_s_map.get(d, np.nan))
    )
    df["sqqq_oo_ret"] = df["Date"].map(
        lambda d: sqqq_oo_map.get(d, synth_s_oo_map.get(d, np.nan))
    )

    # Indicators on full df (BEFORE warmup slice)
    df["ms20"] = ms(df["C"], df["tqqq_ret"], 20)
    df["ms8"] = ms(df["C"], df["tqqq_ret"], 8)
    df["rma20"] = rma(df["tqqq_ret"], 20)
    df["rma12"] = rma(df["tqqq_ret"], 12)
    df["rma10"] = rma(df["tqqq_ret"], 10)
    df["rma5"] = rma(df["tqqq_ret"], 5)
    df["atr14"] = atr_pct(df["H"], df["L"], df["C"], 14)
    df["rt5"] = df["RT"].rolling(5).mean()
    df["bsr5"] = df["BSR"].rolling(5).mean()

    # Apply warmup
    df_w = df.iloc[WARMUP:].reset_index(drop=True)

    # State machine — produces TARGET positions from close-based indicators
    tgt_tp, tgt_sp_raw, tgt_tw, path_arr = run_state_machine(df_w)

    # E² ATR filter applied at DECISION time (close[t]). When ATR14[t] > 15,
    # the SQQQ target is overridden to 0; that override carries through the
    # 1-day shift, so the next-day execution naturally sits flat. This
    # preserves the "pause and resume" behavior on a delayed-execution basis.
    atr14 = df_w["atr14"].values
    target_atr_mask = (tgt_sp_raw == 1) & (atr14 > 15.0)
    tgt_sp_filtered = tgt_sp_raw.copy()
    tgt_sp_filtered[target_atr_mask] = 0

    # MOO execution: a decision made at close[t] becomes effective at open[t+1].
    # Shift target arrays forward by 1 day. Day 0 starts flat (no prior signal).
    tp_arr = np.r_[0, tgt_tp[:-1]].astype(np.int8)
    sp_arr = np.r_[0, tgt_sp_filtered[:-1]].astype(np.int8)
    tw_arr = np.r_[1.0, tgt_tw[:-1]]

    # Portfolio returns — open-to-open prices, executed positions
    t_oo = df_w["tqqq_oo_ret"].fillna(0.0).values
    s_oo = df_w["sqqq_oo_ret"].fillna(0.0).values
    port_ret = tp_arr * tw_arr * t_oo + sp_arr * (1.0 / 3.0) * s_oo
    atr_mask = target_atr_mask  # for end-of-run reporting

    df_w["tp"] = tp_arr
    df_w["sp"] = sp_arr
    df_w["weight"] = tw_arr
    df_w["port_ret"] = port_ret
    df_w["path"] = path_arr
    df_w["state"] = np.where(
        df_w["tp"] == 1, "T", np.where(df_w["sp"] == 1, "S", "F")
    )

    # CP6 equity (start $10,000 on 2000-01-03)
    cp6 = df_w[df_w["Date"] >= CP6_START].copy().reset_index(drop=True)
    cp6["equity"] = (1.0 + cp6["port_ret"].fillna(0.0)).cumprod() * 10000.0

    # Full-period equity (from 1999-10-25, all non-NaN returns)
    full_returns = df_w["port_ret"].dropna().values
    n_full_yrs = len(full_returns) / 252.0
    eq_full_final = float(np.prod(1.0 + full_returns) * 10000.0)
    cagr_full = (eq_full_final / 10000.0) ** (1.0 / n_full_yrs) - 1.0
    sharpe_full = annualised_sharpe(full_returns)
    mdd_full = max_drawdown(full_returns)

    # Real-period stats (from 2010-03-31)
    real_mask = df_w["Date"] >= SPLIT_TQQQ
    r_real = df_w.loc[real_mask, "port_ret"].fillna(0.0).values
    n_real_yrs = len(r_real) / 252.0
    eq_real_final = float(np.prod(1.0 + r_real) * 10000.0)
    cagr_real = (eq_real_final / 10000.0) ** (1.0 / n_real_yrs) - 1.0
    sharpe_real = annualised_sharpe(r_real)
    mdd_real = max_drawdown(r_real)

    # ----- Print results -----
    print("=" * 78)
    print("DAMPIER FILTER E² — BACKTEST  (MOO: next-day market-on-open)")
    print("=" * 78)
    print(f"df rows total            : {len(df):>10,}   (target 6,734)")
    print(f"df_w rows post-warmup    : {len(df_w):>10,}   (target 6,654)")
    print(f"df_w start date          : {df_w['Date'].iloc[0].date()}   (target 1999-10-25)")
    print(f"df_w end date            : {df_w['Date'].iloc[-1].date()}   (target 2026-04-01)")
    print(f"Anchor TQQQ close        : {anchor_close:.6f}    (target 0.290000)")
    print(f"Anchor TQQQ open         : {anchor_open:.6f}    (synth open chain anchor)")
    print()
    print("Targets below are the MOC same-day baseline from filter_e2.py.")
    print()
    print("-" * 78)
    print("Performance — Full period (1999-10-25 → 2026-04-01)")
    print("-" * 78)
    print(f"  CAGR     : {cagr_full * 100:>8.2f}%   MOC baseline  144.13%")
    print(f"  Sharpe   : {sharpe_full:>8.4f}    MOC baseline  3.7114")
    print(f"  MaxDD    : {mdd_full * 100:>8.2f}%   MOC baseline  -15.99%")
    print(f"  Final eq : {fmt_money(eq_full_final)}")
    print()
    print("-" * 78)
    print("Performance — Real period (2010-03-31 → 2026-04-01)")
    print("-" * 78)
    print(f"  CAGR     : {cagr_real * 100:>8.2f}%   MOC baseline  191.53%")
    print(f"  Sharpe   : {sharpe_real:>8.4f}    MOC baseline  4.1188")
    print(f"  MaxDD    : {mdd_real * 100:>8.2f}%   MOC baseline  -13.70%")
    print(f"  Final eq : {fmt_money(eq_real_final)}")
    print()

    # CP1 indicator spot checks (year 2010, 2013, 2022)
    print("-" * 78)
    print("CP1 — Indicator spot checks")
    print("-" * 78)
    cp1_dates = [
        ("2010-01-04", 1.112, 1.496, 0.636, 0.689, 1.612, 1.436, 3.45, "T", 1.000),
        ("2013-01-02", 0.494, 0.421, 0.581, 0.647, 1.058, 1.072, 4.06, "T", 0.986),
        ("2022-01-03", 0.734, 1.360, 0.605, 0.740, 1.640, 0.548, 4.65, "T", 0.860),
        ("2022-01-05", -0.550, -0.815, 0.434, 0.442, 1.496, 0.566, 5.35, "F", None),
    ]
    print(
        f"{'Date':<11}  {'MS20':>7} {'MS8':>7} {'RMA20':>6} {'RMA10':>6} "
        f"{'RT5':>6} {'BSR5':>6} {'ATR':>6} {'St':>2} {'Wt':>5}    (spec)"
    )
    for d, ms20_e, ms8_e, rma20_e, rma10_e, rt5_e, bsr5_e, atr_e, st_e, wt_e in cp1_dates:
        row = df_w[df_w["Date"] == pd.Timestamp(d)]
        if len(row) == 0:
            print(f"{d}  NOT FOUND")
            continue
        r = row.iloc[0]
        wt_str = f"{r['weight']:.3f}" if r["tp"] == 1 else "  ---"
        wt_e_str = f"{wt_e:.3f}" if wt_e is not None else "  ---"
        print(
            f"{d}  {r['ms20']:>7.3f} {r['ms8']:>7.3f} "
            f"{r['rma20']:>6.3f} {r['rma10']:>6.3f} "
            f"{r['rt5']:>6.3f} {r['bsr5']:>6.3f} "
            f"{r['atr14']:>6.2f} {r['state']:>2} {wt_str:>5}"
        )
        print(
            f"{'(spec)':<11}  {ms20_e:>7.3f} {ms8_e:>7.3f} "
            f"{rma20_e:>6.3f} {rma10_e:>6.3f} "
            f"{rt5_e:>6.3f} {bsr5_e:>6.3f} "
            f"{atr_e:>6.2f} {st_e:>2} {wt_e_str:>5}"
        )
    print()

    # CP6 year-end equity
    print("-" * 78)
    print("CP6 — Year-end equity ($10,000 start on 2000-01-03)")
    print("-" * 78)
    cp6["Year"] = cp6["Date"].dt.year
    yearly = cp6.groupby("Year").agg(
        eq=("equity", "last"),
        ann_ret=("port_ret", lambda r: (1.0 + r).prod() - 1.0),
    )
    state_pct = (
        cp6.groupby(["Year", "state"]).size().unstack(fill_value=0)
    )
    state_pct = state_pct.div(state_pct.sum(axis=1), axis=0) * 100.0
    for yr in [
        2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
        2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
        2020, 2021, 2022, 2023, 2024, 2025, 2026,
    ]:
        if yr not in yearly.index:
            continue
        row = yearly.loc[yr]
        ts = state_pct.loc[yr].get("T", 0.0)
        ss = state_pct.loc[yr].get("S", 0.0)
        fs = state_pct.loc[yr].get("F", 0.0)
        print(
            f"  {yr}  T={ts:>4.1f}%  S={ss:>4.1f}%  F={fs:>4.1f}%  "
            f"ret={row['ann_ret'] * 100:>+8.2f}%  eq={fmt_money(row['eq'])}"
        )
    print()

    # Trade count (TQQQ entries from F or S → T plus SQQQ entries from F → S)
    prior_tp = np.r_[0, df_w["tp"].values[:-1]]
    prior_sp = np.r_[0, df_w["sp"].values[:-1]]
    n_tqqq_entries = int(((df_w["tp"] == 1) & (prior_tp == 0)).sum())
    n_sqqq_entries = int(((df_w["sp"] == 1) & (prior_sp == 0)).sum())
    n_total_trades = n_tqqq_entries + n_sqqq_entries
    print("-" * 78)
    print("Trade counts (full period)")
    print("-" * 78)
    print(f"  TQQQ entries : {n_tqqq_entries:>5d}")
    print(f"  SQQQ entries : {n_sqqq_entries:>5d}")
    print(f"  Total trades : {n_total_trades:>5d}   target 529")
    print()

    # ATR filter stats
    n_filt = int(atr_mask.sum())
    print(f"E² ATR filter blocked SQQQ days: {n_filt}   target 269 (~4.0%)")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
