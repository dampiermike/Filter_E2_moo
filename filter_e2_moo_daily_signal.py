#!/usr/bin/env python3
"""
Dampier Filter E² (MOO) Daily Signal Generator
Runs after market close. Reads the latest history CSVs, replays the full
Filter E² state machine through today, then emails / iMessages the signal
for tomorrow's market-on-open execution.

Usage:
    python3 filter_e2_moo_daily_signal.py

Engine logic is imported from filter_e2_market_on_open.py so any change to
the backtester flows through automatically.
"""

import os
import sys
import smtplib
import subprocess
from datetime import date
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import numpy as np
import pandas as pd

from filter_e2_market_on_open import (
    QQQ_F, TQQQ_F, SQQQ_F, VV_F, VIX_F,
    SPLIT_SQQQ, DECAY_S, WARMUP,
    load_csv, ms, rma, atr_pct, rt_v6,
    build_synthetic_tqqq, run_state_machine,
)

# ── Config ────────────────────────────────────────────────────────────────────
GMAIL_USER = os.environ.get('GOOGLE_EMAIL', 'dampiermike@gmail.com')
GMAIL_PASS = os.environ.get('GOOGLE_APP_PASSWORD', '')
TO_EMAIL    = ['dampiermike@gmail.com', 'ddampier777@gmail.com', 'brooke.hoover@yahoo.com']
SMS_NUMBERS = ['+12256144680', '+13038818222', '+18137815601']
# Numbers that must be sent via SMS (Continuity relay through paired iPhone)
# rather than iMessage — e.g. Android/Verizon recipients where iMessage bounces.
SMS_FORCE   = {'+18137815601'}

STARTING_CAPITAL    = 100_000.0
LIVE_START          = pd.Timestamp('2026-04-27')
ATR_FILTER_THRESHOLD = 15.0  # SQQQ blocked when ATR14 > this
SQQQ_WEIGHT          = 1.0 / 3.0


# ── Build merged dataframe (mirrors filter_e2_market_on_open.main) ────────────

def build_dataframe():
    qqq       = load_csv(QQQ_F)
    tqqq_real = load_csv(TQQQ_F)
    sqqq      = load_csv(SQQQ_F)
    vv        = load_csv(VV_F).rename(columns={'BS Ratio': 'BSR'})
    vix       = load_csv(VIX_F).rename(columns={'Close': 'VIX'})

    anchor_close = float(tqqq_real['Close'].iloc[0])
    anchor_open  = float(tqqq_real['Open'].iloc[0])
    synth        = build_synthetic_tqqq(qqq, anchor_close, anchor_open)

    full_t = pd.DataFrame({
        'Date': pd.concat([synth['Date'], tqqq_real['Date']]).values,
        'O':    pd.concat([synth['synth_open'],  tqqq_real['Open']]).values,
        'H':    pd.concat([synth['th'],          tqqq_real['High']]).values,
        'L':    pd.concat([synth['tl'],          tqqq_real['Low']]).values,
        'C':    pd.concat([synth['synth_close'], tqqq_real['Close']]).values,
    }).sort_values('Date').reset_index(drop=True)
    full_t['tqqq_ret']    = full_t['C'].pct_change()
    full_t['tqqq_oo_ret'] = full_t['O'].pct_change()

    full_t['RT_calc'] = rt_v6(full_t['C'], full_t['H'], full_t['L'])
    rt_map = dict(zip(tqqq_real['Date'], tqqq_real['RT']))
    full_t['RT'] = [rt_map.get(d, c) for d, c in zip(full_t['Date'], full_t['RT_calc'])]

    qqq_with_ret = qqq.copy()
    qqq_with_ret['qqq_ret']      = qqq_with_ret['Close'].pct_change()
    qqq_with_ret['qqq_open_ret'] = qqq_with_ret['Open'].pct_change()
    synth_s_map = {
        row['Date']: -3.0 * row['qqq_ret'] - DECAY_S
        for _, row in qqq_with_ret[qqq_with_ret['Date'] < SPLIT_SQQQ].iterrows()
        if not pd.isna(row['qqq_ret'])
    }
    synth_s_oo_map = {
        row['Date']: -3.0 * row['qqq_open_ret'] - DECAY_S
        for _, row in qqq_with_ret[qqq_with_ret['Date'] < SPLIT_SQQQ].iterrows()
        if not pd.isna(row['qqq_open_ret'])
    }

    sqqq['sqqq_ret'] = sqqq['Adj Close'].pct_change()
    sqqq.loc[sqqq['Date'] == SPLIT_SQQQ, 'sqqq_ret'] = synth_s_map.get(SPLIT_SQQQ, 0.0)
    sqqq['adj_open']    = sqqq['Open'] * (sqqq['Adj Close'] / sqqq['Close'])
    sqqq['sqqq_oo_ret'] = sqqq['adj_open'].pct_change()
    sqqq.loc[sqqq['Date'] == SPLIT_SQQQ, 'sqqq_oo_ret'] = synth_s_oo_map.get(SPLIT_SQQQ, 0.0)
    sqqq_map    = dict(zip(sqqq['Date'], sqqq['sqqq_ret']))
    sqqq_oo_map = dict(zip(sqqq['Date'], sqqq['sqqq_oo_ret']))

    df = full_t.merge(vv[['Date', 'BSR']],  on='Date', how='left')
    df = df.merge(vix[['Date', 'VIX']],     on='Date', how='left')
    df[['BSR', 'VIX']] = df[['BSR', 'VIX']].ffill()
    df['sqqq_ret']    = df['Date'].map(lambda d: sqqq_map.get(d, synth_s_map.get(d, np.nan)))
    df['sqqq_oo_ret'] = df['Date'].map(lambda d: sqqq_oo_map.get(d, synth_s_oo_map.get(d, np.nan)))

    # Real prices for sizing display
    df['tqqq_close_real'] = df['Date'].map(dict(zip(tqqq_real['Date'], tqqq_real['Close'])))
    df['sqqq_close_real'] = df['Date'].map(dict(zip(sqqq['Date'],      sqqq['Close'])))

    df['ms20']  = ms(df['C'], df['tqqq_ret'], 20)
    df['ms8']   = ms(df['C'], df['tqqq_ret'], 8)
    df['rma20'] = rma(df['tqqq_ret'], 20)
    df['rma12'] = rma(df['tqqq_ret'], 12)
    df['rma10'] = rma(df['tqqq_ret'], 10)
    df['rma5']  = rma(df['tqqq_ret'], 5)
    df['atr14'] = atr_pct(df['H'], df['L'], df['C'], 14)
    df['rt5']   = df['RT'].rolling(5).mean()
    df['bsr5']  = df['BSR'].rolling(5).mean()

    return df.iloc[WARMUP:].reset_index(drop=True)


# ── Engine driver ─────────────────────────────────────────────────────────────

def run_engine(df_w):
    """Returns target arrays (decisions made at each close) and executed arrays
    (positions in force during each day, = previous day's target).
    """
    tgt_tp, tgt_sp_raw, tgt_tw, path_arr = run_state_machine(df_w)

    atr14         = df_w['atr14'].values
    target_atr_mask = (tgt_sp_raw == 1) & (atr14 > ATR_FILTER_THRESHOLD)
    tgt_sp        = tgt_sp_raw.copy()
    tgt_sp[target_atr_mask] = 0

    exec_tp = np.r_[0, tgt_tp[:-1]].astype(np.int8)
    exec_sp = np.r_[0, tgt_sp[:-1]].astype(np.int8)
    exec_tw = np.r_[1.0, tgt_tw[:-1]]

    t_oo = df_w['tqqq_oo_ret'].fillna(0.0).values
    s_oo = df_w['sqqq_oo_ret'].fillna(0.0).values
    port_ret = exec_tp * exec_tw * t_oo + exec_sp * SQQQ_WEIGHT * s_oo

    return {
        'tgt_tp': tgt_tp, 'tgt_sp': tgt_sp, 'tgt_tw': tgt_tw,
        'exec_tp': exec_tp, 'exec_sp': exec_sp, 'exec_tw': exec_tw,
        'atr_mask': target_atr_mask, 'path_arr': path_arr,
        'port_ret': port_ret,
    }


# ── Signal interpretation ─────────────────────────────────────────────────────

def state_label(tp, sp):
    if tp == 1: return 'T'
    if sp == 1: return 'S'
    return 'F'


def find_entry_date(df_w, exec_arr, last_idx):
    """Walk back to find the most recent F→position transition for the array
    that's currently 1 at last_idx. Returns the date of that transition or None.
    """
    if exec_arr[last_idx] != 1:
        return None
    for j in range(last_idx, 0, -1):
        if exec_arr[j - 1] == 0:
            return df_w['Date'].iloc[j].date()
    return df_w['Date'].iloc[0].date()


def get_signal(df_w, eng):
    i         = len(df_w) - 1
    last_date = df_w['Date'].iloc[i].date()

    today_state = state_label(eng['exec_tp'][i], eng['exec_sp'][i])
    next_state  = state_label(eng['tgt_tp'][i],  eng['tgt_sp'][i])
    tw_today    = float(eng['exec_tw'][i])
    tw_next     = float(eng['tgt_tw'][i])
    atr_blocked = bool(eng['atr_mask'][i])
    path        = eng['path_arr'][i]

    rt5   = df_w['rt5'].iloc[i]
    bsr5  = df_w['bsr5'].iloc[i]
    ms20  = df_w['ms20'].iloc[i]
    ms8   = df_w['ms8'].iloc[i]
    rma20 = df_w['rma20'].iloc[i]
    rma10 = df_w['rma10'].iloc[i]
    atr14 = df_w['atr14'].iloc[i]
    vix   = df_w['VIX'].iloc[i]

    tqqq_close = df_w['tqqq_close_real'].iloc[i]
    if pd.isna(tqqq_close):
        tqqq_close = df_w['C'].iloc[i]
    sqqq_close = df_w['sqqq_close_real'].iloc[i]

    actions = []
    notes   = []
    sizing  = None  # (ticker, price, size_factor, label)

    if today_state == next_state:
        if next_state == 'T':
            entered = find_entry_date(df_w, eng['exec_tp'], i)
            actions.append(f"HOLD TQQQ (entered {entered}, current weight {tw_today:.3f})")
            sizing = ('TQQQ', float(tqqq_close), tw_next, f'TQQQ hold ({path or "—"})')
            if abs(tw_next - tw_today) > 0.005:
                actions.append(f"  REBALANCE TQQQ weight {tw_today:.3f} → {tw_next:.3f} at TOMORROW'S open")
        elif next_state == 'S':
            entered = find_entry_date(df_w, eng['exec_sp'], i)
            actions.append(f"HOLD SQQQ 1/3 (entered {entered})")
            sizing = ('SQQQ', float(sqqq_close), SQQQ_WEIGHT, 'SQQQ hold')
        else:
            actions.append("STAY FLAT — no signal today")
    else:
        if today_state == 'T':
            actions.append("SELL TQQQ at TOMORROW'S open")
        elif today_state == 'S':
            actions.append("SELL SQQQ at TOMORROW'S open")
        if next_state == 'T':
            actions.append(f"BUY TQQQ at TOMORROW'S open (weight {tw_next:.3f}, path={path or '—'})")
            sizing = ('TQQQ', float(tqqq_close), tw_next, f'TQQQ entry ({path or "—"})')
        elif next_state == 'S':
            actions.append("BUY SQQQ at TOMORROW'S open (1/3 size)")
            sizing = ('SQQQ', float(sqqq_close), SQQQ_WEIGHT, 'SQQQ entry')

    notes.append(f"State today  (executed): {today_state}  weight={tw_today:.3f}")
    notes.append(f"State tomorrow (target): {next_state}  weight={tw_next:.3f}")
    if atr_blocked:
        notes.append(f"⚠ ATR filter ACTIVE: SQQQ target blocked because ATR14={atr14:.2f}% > {ATR_FILTER_THRESHOLD:.0f}%")
    if next_state == 'T':
        notes.append(f"TQQQ entry path: {path or '(carry-over)'}")
        notes.append(f"Weight rule: min(1.0, 4.0/ATR14) = min(1.0, 4.0/{atr14:.2f}) = {tw_next:.3f}")
        notes.append("TQQQ exit (any): MS20≤−0.05 | RMA20≤0.38 | ATR14≥9.0 | BSR5≤0.30 | MS8≤−0.15")
    elif next_state == 'S':
        notes.append("SQQQ entry: not in TQQQ AND MS20≤−0.15 AND VIX≥20")
        notes.append("SQQQ exit (any): MS20≥−0.05 | VIX<20 | TQQQ entry fires | RMA5≥0.52")

    notes.append("")
    notes.append(
        f"Indicators: MS20={ms20:.3f} MS8={ms8:.3f} RMA20={rma20:.3f} RMA10={rma10:.3f} "
        f"RT5={rt5:.3f} BSR5={bsr5:.3f} ATR14={atr14:.2f}% VIX={vix:.1f}"
    )
    notes.append(f"Closes: TQQQ ${float(tqqq_close):.2f}  SQQQ ${float(sqqq_close):.2f}")

    return actions, notes, last_date, sizing


# ── Email / iMessage ──────────────────────────────────────────────────────────

def build_sms_summary(actions, data_date) -> str:
    if not actions:
        return f"FilterE2-MOO {data_date.strftime('%y-%m-%d')}: No signal"
    return f"FilterE2-MOO {data_date.strftime('%y-%m-%d')}: {actions[0]}"[:160]


def send_email(subject, body_text):
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From']    = GMAIL_USER
    msg['To']      = ', '.join(TO_EMAIL)
    msg.attach(MIMEText(body_text, 'plain'))
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(GMAIL_USER, GMAIL_PASS)
        server.sendmail(GMAIL_USER, TO_EMAIL, msg.as_string())


def send_imessage(numbers, body):
    safe = body.replace('\\', '\\\\').replace('"', '\\"')
    for num in numbers:
        service_type = 'SMS' if num in SMS_FORCE else 'iMessage'
        script = (
            'tell application "Messages"\n'
            f'  set svc to first service whose service type = {service_type}\n'
            f'  send "{safe}" to participant "{num}" of svc\n'
            'end tell'
        )
        try:
            subprocess.run(['osascript', '-e', script], check=False, timeout=30)
        except subprocess.TimeoutExpired:
            print(f"  warning: osascript send to {num} timed out after 30s")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    today = date.today()
    print(f"Filter E² (MOO) Daily Signal  {today}\n")
    print("Building dataframe...")
    df_w = build_dataframe()
    print(f"  rows: {len(df_w):,}   last bar: {df_w['Date'].iloc[-1].date()}")

    print("Running engine...")
    eng = run_engine(df_w)

    # Trade counts (entries from F → T, F → S)
    prior_tp = np.r_[0, eng['exec_tp'][:-1]]
    prior_sp = np.r_[0, eng['exec_sp'][:-1]]
    n_tqqq = int(((eng['exec_tp'] == 1) & (prior_tp == 0)).sum())
    n_sqqq = int(((eng['exec_sp'] == 1) & (prior_sp == 0)).sum())
    n_atr  = int(eng['atr_mask'].sum())

    # Live equity from LIVE_START
    live_mask    = (df_w['Date'] >= LIVE_START).values
    live_returns = eng['port_ret'][live_mask]
    equity       = STARTING_CAPITAL * float(np.prod(1.0 + live_returns))
    equity_1m    = 1_000_000.0 * float(np.prod(1.0 + live_returns))

    print(f"Total bars: {len(df_w):,}   live bars (since {LIVE_START.date()}): {int(live_mask.sum())}")
    print(f"Trade entries  : TQQQ={n_tqqq}  SQQQ={n_sqqq}  ATR-blocked SQQQ days={n_atr}")
    print(f"Simulated equity (starting ${STARTING_CAPITAL:,.0f} on {LIVE_START.date()}): ${equity:,.2f}")
    print(f"Simulated equity (starting $1,000,000 on {LIVE_START.date()}): ${equity_1m:,.2f}")

    actions, notes, data_date, sizing = get_signal(df_w, eng)

    lines = []
    lines.append("Dampier Filter E² (MOO) — Daily Signal")
    lines.append(f"Data through: {data_date.strftime('%y-%m-%d')}  |  Signal for: tomorrow's open")
    lines.append("")
    for a in actions:
        lines.append(a)
    lines.append("")
    if sizing is not None:
        ticker, price, size_factor, label = sizing
        invest_pct = size_factor * 100.0
        lines.append(f"Position Size ({label})")
        lines.append(f"  Today's close: ${price:.2f}")
        lines.append(f"  Invest: {invest_pct:.1f}% of capital")
        lines.append("")
    lines.append(f"Live equity since {LIVE_START.date()}:")
    lines.append(f"  $100K start  → ${equity:,.2f}")
    lines.append(f"  $1M  start   → ${equity_1m:,.2f}")
    lines.append("")
    lines.append("Details:")
    for n in notes:
        if n.strip():
            lines.append(f"  {n}")

    body = "\n".join(lines)
    print("\n" + body)

    if any('STAY FLAT' in a for a in actions):
        subject = f"FilterE2-MOO {data_date}: FLAT"
    elif any('HOLD' in a for a in actions):
        first = actions[0]
        if 'TQQQ' in first:
            subject = f"FilterE2-MOO {data_date}: HOLD TQQQ"
        elif 'SQQQ' in first:
            subject = f"FilterE2-MOO {data_date}: HOLD SQQQ"
        else:
            subject = f"FilterE2-MOO {data_date}: HOLD"
    else:
        head = actions[0].split('(')[0].strip()
        subject = f"FilterE2-MOO {data_date}: {head}"

    if GMAIL_PASS:
        print(f"\nSending email to {TO_EMAIL} ...")
        try:
            send_email(subject, body)
            print("Email sent.")
        except Exception as e:
            print(f"Email failed: {e}")
    else:
        print(f"\n[skipped email — GOOGLE_APP_PASSWORD not set]  subject would be: {subject}")

    if SMS_NUMBERS:
        sms = build_sms_summary(actions, data_date)
        send_imessage(SMS_NUMBERS, sms)
        print(f"iMessage sent to {SMS_NUMBERS}: {sms}")


if __name__ == '__main__':
    sys.exit(main() or 0)
