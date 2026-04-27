#!/usr/bin/env python3
"""
Fetch all metrics for a ticker from VectorVest Stock Analysis page.
Usage:
    python3 fetch_vectorvest_stock.py --symbol TQQQ
    python3 fetch_vectorvest_stock.py --symbol TQQQ --out tqqq_metrics
"""
import os
import re
import json
import argparse
from pathlib import Path

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout


def login(page, email: str, password: str):
    """Log into www.vectorvest.com."""
    try:
        page.wait_for_selector("input[type='text'], input[type='email'], input[name*='Email' i]",
                               timeout=10000)
    except PWTimeout:
        print("Login form not found — may already be logged in")
        return

    print("Login form loaded")
    for sel in ["input[name*='Email' i]", "input[type='email']", "input[type='text']"]:
        try:
            el = page.locator(sel).first
            if el.count() and el.is_visible():
                el.fill(email)
                print(f"Filled username: {email}")
                break
        except Exception:
            continue

    for sel in ["input[type='password']"]:
        try:
            el = page.locator(sel).first
            if el.count() and el.is_visible():
                el.fill(password)
                print("Filled password")
                break
        except Exception:
            continue

    for sel in ["input[name*='btnLogin' i]", "input[type='submit']", "button[type='submit']"]:
        try:
            el = page.locator(sel).first
            if el.count() and el.is_visible():
                print(f"Clicking login button: {sel}")
                el.click()
                break
        except Exception:
            continue

    try:
        page.wait_for_load_state("domcontentloaded", timeout=15000)
        print(f"Login completed, current URL: {page.url}")
    except PWTimeout:
        print("Timeout after login click")


def extract_all_metrics(page) -> dict:
    """Extract all label/value pairs from the stock analysis page."""
    metrics = {}

    # ── Strategy 1: label/value table rows ────────────────────────────────────
    # VectorVest analysis pages typically have rows of <td> pairs: label | value
    rows = page.locator("table tr").all()
    for row in rows:
        cells = row.locator("td").all()
        if len(cells) >= 2:
            label = cells[0].inner_text().strip().rstrip(':')
            value = cells[1].inner_text().strip()
            if label and value and len(label) < 80:
                metrics[label] = value

    # ── Strategy 2: definition lists <dl><dt><dd> ──────────────────────────────
    dts = page.locator("dl dt").all()
    dds = page.locator("dl dd").all()
    for dt, dd in zip(dts, dds):
        label = dt.inner_text().strip().rstrip(':')
        value = dd.inner_text().strip()
        if label and value:
            metrics[label] = value

    # ── Strategy 3: span/div pairs with data-label or class containing label ──
    # Try to find any element that has a recognisable label pattern
    label_els = page.locator("[class*='label' i], [class*='field-name' i], [class*='metric' i]").all()
    for el in label_els:
        label = el.inner_text().strip().rstrip(':')
        if label and len(label) < 80:
            # Try next sibling
            try:
                parent = el.locator("xpath=..").first
                siblings = parent.locator("*").all()
                found = False
                for sib in siblings:
                    if found:
                        value = sib.inner_text().strip()
                        if value:
                            metrics.setdefault(label, value)
                        break
                    if sib == el or sib.inner_text().strip() == label:
                        found = True
            except Exception:
                pass

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="TQQQ", help="Ticker symbol (default: TQQQ)")
    ap.add_argument("--type", default="0", help="VVExpress type param (default: 0)")
    ap.add_argument("--out", default=None, help="Output file prefix (default: <symbol>_metrics)")
    ap.add_argument("--headful", action="store_true", help="Show browser window")
    ap.add_argument("--cookies-path", help="Path to browser cookie JSON (optional)")
    args = ap.parse_args()

    email = os.getenv("VECTORVEST_EMAIL")
    password = os.getenv("VECTORVEST_PASSWORD")
    if not email or not password:
        raise SystemExit("Set env vars VECTORVEST_EMAIL and VECTORVEST_PASSWORD before running.")

    symbol = args.symbol.upper()
    out_prefix = args.out or f"{symbol.lower()}_metrics"
    out_csv  = Path(f"/Users/mikedampier/.openclaw/workspace/{out_prefix}.csv")
    out_json = Path(f"/Users/mikedampier/.openclaw/workspace/{out_prefix}.json")
    out_html = Path(f"/Users/mikedampier/.openclaw/workspace/{out_prefix}_debug.html")
    out_png  = Path(f"/Users/mikedampier/.openclaw/workspace/{out_prefix}_debug.png")

    target_url = f"https://www.vectorvest.com/vvexpress/us/stockanalysis.aspx?Type={args.type}&Symbol={symbol}"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headful)
        ctx = browser.new_context()

        if args.cookies_path:
            from fetch_vectorvest_timing import load_cookies_into_context
            load_cookies_into_context(ctx, args.cookies_path)

        page = ctx.new_page()

        # Step 1: Login to www.vectorvest.com (main site)
        login_url = "https://www.vectorvest.com/vvlogin/login.aspx"
        print(f"Step 1: Logging into main site {login_url}...")
        page.goto(login_url, wait_until="domcontentloaded")
        login(page, email, password)
        page.wait_for_timeout(2000)

        # Step 2: Login to members.vectorvest.com (vvexpress auth)
        members_login = "https://members.vectorvest.com/vvexpress/us/stockanalysis.aspx"
        print(f"\nStep 2: Navigating to members.vectorvest.com...")
        page.goto(members_login, wait_until="domcontentloaded")
        page.wait_for_timeout(2000)
        print(f"  Landed: {page.url}")
        print(f"  Title:  {page.title()}")

        # If redirected to login, fill it
        if "login" in page.url.lower():
            print("  Login form detected on members subdomain — filling credentials...")
            login(page, email, password)
            page.wait_for_timeout(2000)
            print(f"  After login: {page.url}")

        # Step 3: Navigate via ApplicationFrameset with ReturnURL
        frameset_url = f"https://www.vectorvest.com/vvlogin/ApplicationFrameset.aspx?type=3&ReturnURL=%2Fvvexpress%2Fus%2Fstockanalysis.aspx%3FType%3D{args.type}%26Symbol%3D{symbol}"
        print(f"\nStep 3: Navigating via ApplicationFrameset...")
        page.goto(frameset_url, wait_until="domcontentloaded")
        page.wait_for_timeout(4000)
        print(f"  Frameset URL: {page.url}")
        print(f"  Title: {page.title()}")

        # Check for iframes
        frames = page.frames
        print(f"  Frames on page: {len(frames)}")
        for i, fr in enumerate(frames):
            print(f"    Frame {i}: {fr.url}")

        # If still on login, try direct URL
        if "login" in page.url.lower():
            print(f"\n  Still on login — trying direct URL: {target_url}")
            page.goto(target_url, wait_until="domcontentloaded")
            page.wait_for_timeout(3000)

        print(f"\nStep 4: Final URL: {page.url}")
        print(f"  Title: {page.title()}")

        print(f"Landed on: {page.url}")
        print(f"Page title: {page.title()}")

        # Save debug artifacts
        page.screenshot(path=str(out_png))
        out_html.write_text(page.content(), encoding="utf-8")
        print(f"Saved debug screenshot: {out_png.name}")

        # Count tables
        table_count = page.locator("table").count()
        print(f"Tables found: {table_count}")

        # Dump all table contents for inspection
        all_rows = []
        for t_idx in range(table_count):
            tbl = page.locator("table").nth(t_idx)
            rows = tbl.locator("tr").all()
            for row in rows:
                cells = row.locator("td, th").all()
                cell_texts = [c.inner_text().strip() for c in cells]
                if any(cell_texts):
                    all_rows.append(cell_texts)

        # Get full page text for regex-based extraction
        full_text = page.inner_text("body")

        # ── Parse structured metrics from paragraph prose ──────────────────────
        # Each metric paragraph starts with "METRIC_NAME (abbrev):" and contains
        # "TQQQ has a/an X of VALUE" or similar patterns.
        metrics = {}

        # Metric patterns: extract name + value from prose paragraphs
        metric_patterns = [
            # "TQQQ has a current Value of $43.33 per share"
            # "TQQQ has an RV of 1.00"
            # "TQQQ has a VST rating of 0.97"
            (r'(?:Value|RV|RS|RT|VST|GRT|EPS|P/E|EY|GPE|DIV|DY|DS|DG|YSG|CI|Stop)[:\s].*?'
             r'TQQQ has (?:a current |an |a )(\w[\w /\-]*?) (?:rating )?of ([\$\d\.,\-%]+)',
             None),
        ]

        # Targeted extraction for each known metric
        known_metrics = {
            'Symbol':           r'The ticker symbol for[^i]+is (\w+)\.',
            'Name':             r'VectorVest Stock Analysis of ([^\n]+?) as of',
            'Date':             r'as of ([\d/]+)',
            'Exchange':         r'TQQQ is traded on the ([^\-]+)',
            'Business':         r'Business:\s*[^\n]*?\(TQQQ\)\s*([^\n]+)',
            'Business Sector':  r'Business Sector:\s*TQQQ has been assigned to the ([^\.\n]+?) Business Sector',
            'Industry Group':   r'Industry Group:\s*TQQQ has been assigned to the ([^\.\n]+?) Industry Group',
            'Price':            r'TQQQ closed on [\d/]+ at \$([\d\.]+) per share',
            'Open':             r'TQQQ opened trading at a price of \$([\d\.]+)',
            'High':             r'TQQQ traded at a High price of \$([\d\.]+)',
            'Low':              r'TQQQ traded at a Low price of \$([\d\.]+)',
            'Close':            r'TQQQ closed trading at price \$([\d\.]+)',
            'Range':            r'TQQQ traded with a range of \$([\d\.]+)',
            '$Change':          r'TQQQ closed (?:up|down) ([\d\.]+) from the prior',
            '%PRC':             r"TQQQ's Price changed ([\-\d\.]+%)",
            'Volume':           r'TQQQ traded ([\d,]+) shares on',
            'AvgVol':           r'TQQQ has an AvgVol of ([\d,]+)',
            '%Vol':             r'TQQQ had a %Vol of ([\-\d\.]+%)',
            'Value ($)':        r'TQQQ has a current Value of \$([\d\.]+)',
            'RV':               r'TQQQ has an RV of ([\d\.]+)',
            'RS':               r'TQQQ has an RS rating of ([\d\.]+)',
            'RT':               r'TQQQ has a Relative Timing rating of ([\d\.]+)',
            'VST':              r'TQQQ has a VST rating of ([\d\.]+)',
            'REC':              r'TQQQ has a (\w+) recommendation\.',
            'Stop ($)':         r'TQQQ has a Stop of \$([\d\.]+)',
            'GRT (%)':          r'TQQQ has a forecasted Earnings Growth Rate of ([\-\d\.]+%)',
            'EPS ($)':          r'TQQQ has a forecasted EPS of \$([\d\.]+)',
            'P/E':              r'TQQQ has a P/E of ([\d\.]+)',
            'EY (%)':           r'TQQQ has an EY of ([\d\.]+) percent',
            'GPE':              r'TQQQ has a GPE rating of ([\d\.]+)',
            'DIV ($)':          r'TQQQ pays an annual dividend of \$([\d\.]+)',
            'DY (%)':           r'TQQQ has a Dividend Yield of ([\d\.]+%)',
            'DS':               r'TQQQ has a Dividend Safety of (\d+)',
            'DG (%)':           r'TQQQ has a Dividend Growth of (\d+%)',
            'YSG':              r'TQQQ has a YSG rating of ([\d\.]+)',
            'CI':               r'TQQQ has a CI rating of ([\d\.]+)',
            'Sales ($)':        r'TQQQ has annual sales of \$([\d\.]+)',
            'Sales Growth (%)': r'TQQQ has a Sales Growth of ([\-\d\.]+%)',
            'SPS ($)':          r'TQQQ has annual sales of \$([\d\.]+) per share',
            'P/S':              r'TQQQ has a P/S of ([\d\.]+)',
            'Shares':           r'TQQQ has ([\d,\.]+) shares of stock outstanding',
            'Market Cap ($)':   r'TQQQ has a Market Capitalization of \$([\d,\.]+)',
        }

        for metric_name, pattern in known_metrics.items():
            m = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
            if m:
                metrics[metric_name] = m.group(1).strip()

        print(f"\n── Extracted {len(metrics)} metrics ──")
        for k, v in metrics.items():
            print(f"  {k:25s}: {v}")

        # Save outputs
        df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
        df.to_csv(out_csv, index=False)
        out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"\nWrote {out_csv.name} ({len(df)} metrics)")
        print(f"Wrote {out_json.name}")

        ctx.close()
        browser.close()


if __name__ == "__main__":
    main()
