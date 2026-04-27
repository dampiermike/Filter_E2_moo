#!/usr/bin/env python3
"""
Validate that the Filter E² MOO engine trades on the day AFTER the signal.

Two assertions per signal-day:
  1. On the signal day t (close[t] generates a new target), the executed
     position has NOT changed yet — exec[t] still equals the prior target.
  2. On the next day t+1 (open[t+1] = fill time), the executed position
     equals the new target — exec[t+1] == tgt[t].

Also samples 5 entry transitions and 5 exit transitions, printing the
3-day window around each so you can eyeball the timing.
"""
import sys
import numpy as np
import pandas as pd

from filter_e2_moo_daily_signal import build_dataframe, run_engine

ATR_FILTER_THRESHOLD = 15.0  # mirrors filter_e2_market_on_open

print("Building dataframe + running engine...")
df_w = build_dataframe()
eng = run_engine(df_w)

dates  = df_w['Date'].dt.strftime('%Y-%m-%d').values
tgt_tp = eng['tgt_tp']
tgt_sp = eng['tgt_sp']
exec_tp = eng['exec_tp']
exec_sp = eng['exec_sp']
port_ret = eng['port_ret']

N = len(df_w)
print(f"Bars: {N:,}  ({dates[0]} → {dates[-1]})\n")

# Sanity 0: the shift identity must hold by construction
assert np.array_equal(exec_tp[1:], tgt_tp[:-1]), "exec_tp shift broken"
assert np.array_equal(exec_sp[1:], tgt_sp[:-1]), "exec_sp shift broken"
assert exec_tp[0] == 0 and exec_sp[0] == 0,        "day 0 must start flat"
print("[PASS] shift identity: exec[i] == tgt[i-1] for all i; day 0 flat\n")

# Build a state label (T/S/F) per day for both target and executed
def state(tp, sp):
    if tp == 1: return 'T'
    if sp == 1: return 'S'
    return 'F'

tgt_state  = np.array([state(tgt_tp[i],  tgt_sp[i])  for i in range(N)])
exec_state = np.array([state(exec_tp[i], exec_sp[i]) for i in range(N)])

# Signal days = days where the TARGET state changes vs the previous day.
# A signal at close[t] means tgt_state[t] != tgt_state[t-1].
sig_days = np.where(tgt_state[1:] != tgt_state[:-1])[0] + 1  # absolute index
print(f"Signal days (target transitions): {len(sig_days):,}")
print(f"  Of these, F→T entries: {sum(1 for t in sig_days if tgt_state[t] == 'T' and tgt_state[t-1] == 'F')}")
print(f"           F→S entries: {sum(1 for t in sig_days if tgt_state[t] == 'S' and tgt_state[t-1] == 'F')}")
print(f"           T→F exits  : {sum(1 for t in sig_days if tgt_state[t] == 'F' and tgt_state[t-1] == 'T')}")
print(f"           S→F exits  : {sum(1 for t in sig_days if tgt_state[t] == 'F' and tgt_state[t-1] == 'S')}\n")

# Assertion 1: on signal day t, exec_state[t] == tgt_state[t-1] (NOT yet the new target)
# Assertion 2: on day t+1, exec_state[t+1] == tgt_state[t] (new position now in force)
fail_1, fail_2 = 0, 0
for t in sig_days:
    if exec_state[t] != tgt_state[t-1]:
        fail_1 += 1
    if t + 1 < N and exec_state[t+1] != tgt_state[t]:
        fail_2 += 1
print(f"[{'PASS' if fail_1 == 0 else 'FAIL'}] Assertion 1: signal day exec NOT yet changed  ({fail_1} failures of {len(sig_days)})")
print(f"[{'PASS' if fail_2 == 0 else 'FAIL'}] Assertion 2: next day exec == new target     ({fail_2} failures of {len(sig_days)})\n")

# Pick 5 F→T entries and 5 T→F exits to print as eyeball examples
def pick_examples(sig_days, prev, new, n=5):
    out = [t for t in sig_days if tgt_state[t-1] == prev and tgt_state[t] == new]
    if len(out) <= n:
        return out
    # Spread across the timeline
    return [out[int(round(i * (len(out) - 1) / (n - 1)))] for i in range(n)]

def print_window(t, label):
    print(f"  ── {label}  signal day = {dates[t]}  (transition {tgt_state[t-1]} → {tgt_state[t]}) ──")
    print(f"  {'Day':<12} {'Date':<11} {'tgt':>4} {'exec':>5} {'oo_ret':>8}  note")
    for off in (-1, 0, 1, 2):
        i = t + off
        if i < 0 or i >= N:
            continue
        tag = ''
        if off == -1: tag = '(prior day — old target in force)'
        if off ==  0: tag = '(SIGNAL DAY — close generates new target; NO TRADE today)'
        if off ==  1: tag = '(EXECUTION DAY — fill at this open; new position now in force)'
        if off ==  2: tag = '(post-execution; new position still held)'
        print(f"  t{off:+d}        {dates[i]} {tgt_state[i]:>4} {exec_state[i]:>5} {port_ret[i]:>+8.4f}  {tag}")
    print()

print("─" * 78)
print("Sampled F→T entry transitions")
print("─" * 78)
for t in pick_examples(sig_days, 'F', 'T', 5):
    print_window(t, "F→T entry")

print("─" * 78)
print("Sampled T→F exit transitions")
print("─" * 78)
for t in pick_examples(sig_days, 'T', 'F', 5):
    print_window(t, "T→F exit")

print("─" * 78)
print("Sampled F→S entry transitions")
print("─" * 78)
for t in pick_examples(sig_days, 'F', 'S', 5):
    print_window(t, "F→S entry")

# Final summary
ok = (fail_1 == 0 and fail_2 == 0)
print("=" * 78)
print(f"  Result: {'PASS — MOO timing verified' if ok else 'FAIL — see assertion failures above'}")
print("=" * 78)
sys.exit(0 if ok else 1)
