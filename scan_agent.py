"""
Odyssey — Scan Agent
Screens AI/chip supply chain and clean energy stocks for trading signals.
Flags unusual volume, price momentum, RSI extremes, and MA crossovers, then
scores each ticker's net lean (bullish vs bearish).

Outputs on each run:
  • scan_reports/scan_YYYY-MM-DD.md   — dated markdown report
  • dashboard.html                    — minimalist live dashboard (overwritten each run)

Usage:
    python scan_agent.py

Data source: Yahoo Finance public chart API (no third-party packages required
beyond pandas). Color convention: 🟢 = bullish lean, 🔴 = bearish lean.
"""

from __future__ import annotations
import pandas as pd
from datetime import datetime, date
import os
import sys
import json
import time
import urllib.request

# ─── WATCHLIST ────────────────────────────────────────────────────────────────
# Organized by thesis. MU is the north star.

WATCHLIST = {
    "AI Memory & Chip Supply Chain": [
        "MU",    # Micron — north star. DRAM/NAND for AI/data centers
        "NVDA",  # Nvidia — GPUs, AI inference/training
        "AMD",   # AMD — CPUs/GPUs competing with Intel/Nvidia
        "AMAT",  # Applied Materials — semiconductor equipment
        "LRCX",  # Lam Research — etch/deposition equipment
        "KLAC",  # KLA Corp — process control equipment
        "ASML",  # ASML — EUV lithography (chokepoint of the whole industry)
        "TSM",   # TSMC — foundry that makes the chips
        "MRVL",  # Marvell — data center networking chips
        "ON",    # ON Semiconductor — mixed-signal, EVs + AI edge
        "SMCI",  # Super Micro — AI server infrastructure
        "ARM",   # Arm Holdings — chip architecture IP
    ],
    "Clean Energy & Solar": [
        "TAN",   # Invesco Solar ETF — broad solar exposure
        "ENPH",  # Enphase — microinverters, residential solar
        "FSLR",  # First Solar — utility-scale solar panels
        "SEDG",  # SolarEdge — inverters + energy storage
        "NEE",   # NextEra Energy — largest renewable utility
        "PLUG",  # Plug Power — hydrogen fuel cells
        "RUN",   # Sunrun — residential solar installer
        "CSIQ",  # Canadian Solar — global solar manufacturer
    ],
}

NORTH_STAR = "MU"

# ─── THRESHOLDS ───────────────────────────────────────────────────────────────

VOLUME_SPIKE_THRESHOLD = 1.5     # Flag if today's volume > 1.5x 20-day avg
PRICE_MOVE_THRESHOLD   = 2.0     # Flag if 1-day price change > ±2%
RSI_OVERBOUGHT         = 70
RSI_OVERSOLD           = 30
LOOKBACK_DAYS          = "6mo"   # How much history to pull for calculations

# ─── DATA FETCH ───────────────────────────────────────────────────────────────

def fetch_history(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Pull daily OHLCV from Yahoo's public chart API. Returns empty df on failure."""
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
           f"?range={period}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.load(r)
            res = data["chart"]["result"][0]
            ts = res["timestamp"]
            q = res["indicators"]["quote"][0]
            df = pd.DataFrame({
                "Open": q["open"], "High": q["high"], "Low": q["low"],
                "Close": q["close"], "Volume": q["volume"],
            }, index=pd.to_datetime(ts, unit="s"))
            return df.dropna(subset=["Close"])
        except Exception as e:
            if attempt == 3:
                print(f"  ⚠ {ticker} fetch failed: {e}", file=sys.stderr)
                return pd.DataFrame()
            time.sleep(2 * (attempt + 1))
    return pd.DataFrame()

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 1)


def compute_signals(ticker: str) -> dict | None:
    """Fetch data and compute all signals for a single ticker.

    Each flag carries a direction: +1 bullish, -1 bearish, 0 neutral.
    """
    try:
        hist = fetch_history(ticker, LOOKBACK_DAYS)

        if hist.empty or len(hist) < 30:
            return None

        close   = hist["Close"]
        volume  = hist["Volume"]

        price_now   = close.iloc[-1]
        price_prev  = close.iloc[-2]
        price_1d_pct = ((price_now - price_prev) / price_prev) * 100
        price_5d_pct = ((price_now - close.iloc[-6]) / close.iloc[-6]) * 100 if len(close) >= 6 else None
        price_20d_pct = ((price_now - close.iloc[-21]) / close.iloc[-21]) * 100 if len(close) >= 21 else None

        vol_today   = volume.iloc[-1]
        vol_avg_20  = volume.iloc[-21:-1].mean()
        vol_ratio   = vol_today / vol_avg_20 if vol_avg_20 > 0 else 0

        ma_50  = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
        ma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None

        rsi = compute_rsi(close)

        high_52w = close.rolling(252).max().iloc[-1] if len(close) >= 252 else close.max()
        low_52w  = close.rolling(252).min().iloc[-1] if len(close) >= 252 else close.min()
        pct_from_high = ((price_now - high_52w) / high_52w) * 100
        pct_from_low  = ((price_now - low_52w) / low_52w) * 100

        # ── Flag conditions ──────────────────────────────────────────────────
        # Each flag is (text, direction): +1 bullish, -1 bearish, 0 neutral.
        flags = []

        if vol_ratio >= VOLUME_SPIKE_THRESHOLD:
            # Volume confirms whichever way price is moving that day.
            vdir = 1 if price_1d_pct > 0 else -1 if price_1d_pct < 0 else 0
            flags.append((f"Vol spike {vol_ratio:.1f}x avg", vdir))

        if price_1d_pct >= PRICE_MOVE_THRESHOLD:
            flags.append((f"Up {price_1d_pct:.1f}% today", 1))
        elif price_1d_pct <= -PRICE_MOVE_THRESHOLD:
            flags.append((f"Down {price_1d_pct:.1f}% today", -1))

        if rsi >= RSI_OVERBOUGHT:
            # Overbought = stretched / toppy → bearish lean.
            flags.append((f"RSI overbought ({rsi})", -1))
        elif rsi <= RSI_OVERSOLD:
            # Oversold = potential bounce → bullish lean.
            flags.append((f"RSI oversold ({rsi})", 1))

        if ma_50 and ma_200:
            prev_ma50  = close.rolling(50).mean().iloc[-2]
            prev_ma200 = close.rolling(200).mean().iloc[-2]
            if prev_ma50 < prev_ma200 and ma_50 >= ma_200:
                flags.append(("Golden cross (50MA crossed above 200MA)", 1))
            elif prev_ma50 > prev_ma200 and ma_50 <= ma_200:
                flags.append(("Death cross (50MA crossed below 200MA)", -1))

        if pct_from_high >= -3:
            flags.append((f"Near 52w high ({pct_from_high:.1f}%)", 1))
        if pct_from_low <= 5:
            flags.append((f"Near 52w low (+{pct_from_low:.1f}%)", -1))

        net = sum(d for _, d in flags)

        return {
            "ticker":        ticker,
            "price":         round(price_now, 2),
            "1d_pct":        round(price_1d_pct, 2),
            "5d_pct":        round(price_5d_pct, 2) if price_5d_pct else None,
            "20d_pct":       round(price_20d_pct, 2) if price_20d_pct else None,
            "vol_ratio":     round(vol_ratio, 2),
            "rsi":           rsi,
            "ma_50":         round(ma_50, 2) if ma_50 else None,
            "ma_200":        round(ma_200, 2) if ma_200 else None,
            "pct_from_high": round(pct_from_high, 2),
            "pct_from_low":  round(pct_from_low, 2),
            "flags":         [t for t, _ in flags],
            "flag_dirs":     flags,
            "signal_count":  len(flags),
            "net":           net,
        }

    except Exception as e:
        print(f"  ⚠ Error computing {ticker}: {e}", file=sys.stderr)
        return None


def tier(s: dict) -> str:
    """Directional tier. 🟢 = bullish lean, 🔴 = bearish lean, 🟡 = mixed, ⚪ = quiet."""
    net = s["net"]
    count = s["signal_count"]
    if count == 0:
        return "⚪ QUIET"
    if net >= 2:
        return "🟢 BULLISH"
    if net == 1:
        return "🟢 Lean Bull"
    if net == 0:
        return "🟡 MIXED"
    if net == -1:
        return "🔴 Lean Bear"
    return "🔴 BEARISH"


def tier_rank(s: dict) -> int:
    """Sort key: most decisive signals first (by absolute net, then count)."""
    return abs(s["net"]) * 10 + s["signal_count"]


def format_pct(val) -> str:
    if val is None:
        return "—"
    arrow = "▲" if val > 0 else "▼" if val < 0 else "—"
    return f"{arrow} {abs(val):.1f}%"


# ─── MARKDOWN REPORT ───────────────────────────────────────────────────────────

def build_report(results: dict) -> str:
    now = datetime.now()
    lines = []

    lines.append("# Odyssey — Scan Report")
    lines.append(f"**{now.strftime('%A, %B %d, %Y — %I:%M %p')}**")
    lines.append("")

    total_flagged = sum(1 for group in results.values() for s in group if s and s["signal_count"] > 0)
    lines.append(f"> Scanned {sum(len(v) for v in results.values())} tickers across "
                 f"{len(results)} themes. **{total_flagged} flagged.**  "
                 f"🟢 bullish lean · 🔴 bearish lean · 🟡 mixed · ⚪ quiet")
    lines.append("")

    # ── Decisive signals summary at top ──
    notable = [
        (theme, s) for theme, group in results.items()
        for s in group if s and abs(s["net"]) >= 2
    ]
    if notable:
        lines.append("## ⚡ Signals Worth a Look")
        lines.append("")
        for theme, s in sorted(notable, key=lambda x: -tier_rank(x[1])):
            star = " ⭐" if s["ticker"] == NORTH_STAR else ""
            lines.append(f"### {s['ticker']}{star}  `{tier(s)}`")
            lines.append(f"**${s['price']}** | 1d: {format_pct(s['1d_pct'])} | "
                         f"5d: {format_pct(s['5d_pct'])} | 20d: {format_pct(s['20d_pct'])}")
            lines.append(f"Vol: **{s['vol_ratio']:.1f}x** avg | RSI: **{s['rsi']}** | "
                         f"From 52w high: {s['pct_from_high']:.1f}% | From 52w low: +{s['pct_from_low']:.1f}%")
            for flag in s["flags"]:
                lines.append(f"- {flag}")
            lines.append("")

    # ── Full breakdown by theme ──
    lines.append("---")
    lines.append("")
    lines.append("## Full Breakdown")
    lines.append("")

    for theme, group in results.items():
        lines.append(f"### {theme}")
        lines.append("")
        lines.append("| Ticker | Price | 1d | 5d | 20d | Vol/Avg | RSI | Signal |")
        lines.append("|--------|-------|----|----|-----|---------|-----|--------|")

        for s in group:
            if s is None:
                continue
            star = " ⭐" if s["ticker"] == NORTH_STAR else ""
            lines.append(
                f"| **{s['ticker']}**{star} "
                f"| ${s['price']} "
                f"| {format_pct(s['1d_pct'])} "
                f"| {format_pct(s['5d_pct'])} "
                f"| {format_pct(s['20d_pct'])} "
                f"| {s['vol_ratio']:.1f}x "
                f"| {s['rsi']} "
                f"| {tier(s)} |"
            )

        lines.append("")

        flagged = [s for s in group if s and s["flags"]]
        if flagged:
            for s in flagged:
                lines.append(f"**{s['ticker']}** — " + " · ".join(s["flags"]))
            lines.append("")

    lines.append("---")
    lines.append("*Generated by Odyssey Scan Agent. Not financial advice. You make the calls.*")

    return "\n".join(lines)


# ─── HTML DASHBOARD ─────────────────────────────────────────────────────────────

def _tier_css(s: dict) -> str:
    if s["signal_count"] == 0:
        return "quiet"
    if s["net"] >= 2:
        return "bull"
    if s["net"] == 1:
        return "bull-lean"
    if s["net"] == 0:
        return "mixed"
    if s["net"] == -1:
        return "bear-lean"
    return "bear"


def build_dashboard(results: dict) -> str:
    now = datetime.now()
    total = sum(len(v) for v in results.values())
    flagged = sum(1 for g in results.values() for s in g if s and s["signal_count"] > 0)
    bull = sum(1 for g in results.values() for s in g if s and s["net"] > 0)
    bear = sum(1 for g in results.values() for s in g if s and s["net"] < 0)

    rows = []
    for theme, group in results.items():
        rows.append(f'<h2 class="theme">{theme}</h2>')
        rows.append('<table><thead><tr>'
                    '<th>Ticker</th><th>Price</th><th>1d</th><th>5d</th><th>20d</th>'
                    '<th>Vol/Avg</th><th>RSI</th><th>Signal</th><th>Flags</th>'
                    '</tr></thead><tbody>')
        for s in group:
            if s is None:
                continue
            cls = _tier_css(s)
            star = ' <span class="star">★</span>' if s["ticker"] == NORTH_STAR else ""
            label = tier(s).split(" ", 1)[1] if " " in tier(s) else tier(s)

            def cell(v):
                if v is None:
                    return '<td class="num">—</td>'
                c = "up" if v > 0 else "down" if v < 0 else ""
                arrow = "▲" if v > 0 else "▼" if v < 0 else "—"
                return f'<td class="num {c}">{arrow} {abs(v):.1f}%</td>'

            rsi_cls = "hot" if s["rsi"] >= RSI_OVERBOUGHT else "cold" if s["rsi"] <= RSI_OVERSOLD else ""
            flags = " · ".join(s["flags"]) if s["flags"] else "—"
            rows.append(
                f'<tr class="row {cls}">'
                f'<td class="tkr">{s["ticker"]}{star}</td>'
                f'<td class="num">${s["price"]:,.2f}</td>'
                f'{cell(s["1d_pct"])}{cell(s["5d_pct"])}{cell(s["20d_pct"])}'
                f'<td class="num">{s["vol_ratio"]:.1f}x</td>'
                f'<td class="num {rsi_cls}">{s["rsi"]}</td>'
                f'<td><span class="pill {cls}">{label}</span></td>'
                f'<td class="flags">{flags}</td>'
                f'</tr>'
            )
        rows.append('</tbody></table>')

    body = "\n".join(rows)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Odyssey</title>
<meta name="theme-color" content="#0f1115">
<link rel="icon" type="image/png" href="favicon.png">
<link rel="icon" type="image/svg+xml" href="icon.svg">
<link rel="apple-touch-icon" href="icon-180.png">
<link rel="manifest" href="manifest.json">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Odyssey">
<style>
  :root {{
    --bg:#0f1115; --panel:#171a21; --line:#262b36; --text:#e6e9ef;
    --muted:#8a93a3; --bull:#2ecc71; --bear:#ff5c5c; --mixed:#e0b341;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--text);
    font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  .wrap {{ max-width:1040px; margin:0 auto; padding:32px 24px 64px; }}
  h1 {{ font-size:22px; font-weight:650; margin:0 0 2px; letter-spacing:.2px; }}
  .ts {{ color:var(--muted); font-size:13px; margin-bottom:24px; }}
  .summary {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:28px; }}
  .stat {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
    padding:12px 16px; min-width:96px; }}
  .stat .n {{ font-size:22px; font-weight:650; }}
  .stat .l {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.5px; }}
  .stat.bull .n {{ color:var(--bull); }} .stat.bear .n {{ color:var(--bear); }}
  h2.theme {{ font-size:14px; font-weight:600; color:var(--muted); text-transform:uppercase;
    letter-spacing:.6px; margin:28px 0 10px; }}
  table {{ width:100%; border-collapse:collapse; background:var(--panel);
    border:1px solid var(--line); border-radius:10px; overflow:hidden; }}
  th {{ text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.5px;
    color:var(--muted); font-weight:600; padding:10px 12px; border-bottom:1px solid var(--line); }}
  td {{ padding:11px 12px; border-bottom:1px solid var(--line); font-size:14px; }}
  tr:last-child td {{ border-bottom:none; }}
  .num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
  th:nth-child(n+2):nth-child(-n+7) {{ text-align:right; }}
  .tkr {{ font-weight:650; }}
  .star {{ color:var(--mixed); }}
  .up {{ color:var(--bull); }} .down {{ color:var(--bear); }}
  .hot {{ color:var(--bear); font-weight:600; }} .cold {{ color:#5aa9ff; font-weight:600; }}
  .flags {{ color:var(--muted); font-size:12px; }}
  .row {{ border-left:3px solid transparent; }}
  .row.bull {{ border-left-color:var(--bull); }}
  .row.bull-lean {{ border-left-color:rgba(46,204,113,.5); }}
  .row.bear {{ border-left-color:var(--bear); }}
  .row.bear-lean {{ border-left-color:rgba(255,92,92,.5); }}
  .row.mixed {{ border-left-color:var(--mixed); }}
  .pill {{ display:inline-block; padding:2px 9px; border-radius:20px; font-size:12px;
    font-weight:600; white-space:nowrap; }}
  .pill.bull {{ background:rgba(46,204,113,.15); color:var(--bull); }}
  .pill.bull-lean {{ background:rgba(46,204,113,.1); color:var(--bull); }}
  .pill.bear {{ background:rgba(255,92,92,.15); color:var(--bear); }}
  .pill.bear-lean {{ background:rgba(255,92,92,.1); color:var(--bear); }}
  .pill.mixed {{ background:rgba(224,179,65,.15); color:var(--mixed); }}
  .pill.quiet {{ background:#20242d; color:var(--muted); }}
  .legend {{ color:var(--muted); font-size:12px; margin:20px 0 0; }}
  .legend b.bull {{ color:var(--bull); }} .legend b.bear {{ color:var(--bear); }}
  .legend b.mixed {{ color:var(--mixed); }}
  footer {{ color:var(--muted); font-size:12px; margin-top:32px; border-top:1px solid var(--line); padding-top:16px; }}
</style></head>
<body><div class="wrap">
  <h1>Odyssey — Scan Dashboard</h1>
  <div class="ts">{now.strftime('%A, %B %d, %Y · %I:%M %p')}</div>
  <div class="summary">
    <div class="stat"><div class="n">{total}</div><div class="l">Scanned</div></div>
    <div class="stat"><div class="n">{flagged}</div><div class="l">Flagged</div></div>
    <div class="stat bull"><div class="n">{bull}</div><div class="l">Bullish lean</div></div>
    <div class="stat bear"><div class="n">{bear}</div><div class="l">Bearish lean</div></div>
  </div>
  {body}
  <p class="legend"><b class="bull">● Green</b> = bullish lean (momentum, near highs, golden cross, oversold bounce) ·
    <b class="bear">● Red</b> = bearish lean (overbought/toppy, selling off, death cross, near lows) ·
    <b class="mixed">● Yellow</b> = mixed · ● Grey = quiet. ★ = north star (MU).</p>
  <footer>Generated by Odyssey Scan Agent. Not financial advice — signals only. You make the calls.</footer>
</div></body></html>"""


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("\n🔍 Odyssey Scan Agent starting...\n")

    results = {}
    for theme, tickers in WATCHLIST.items():
        print(f"  Scanning {theme}...")
        group = []
        for ticker in tickers:
            data = compute_signals(ticker)
            group.append(data)
            if data:
                print(f"    → {ticker}: {tier(data)}")
            else:
                print(f"    → {ticker}: (no data)")
            time.sleep(0.6)  # gentle throttle to avoid rate limiting
        results[theme] = group
        print()

    report = build_report(results)
    dashboard = build_dashboard(results)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Save dated markdown report ──
    reports_dir = os.path.join(script_dir, "scan_reports")
    os.makedirs(reports_dir, exist_ok=True)
    md_path = os.path.join(reports_dir, f"scan_{date.today().strftime('%Y-%m-%d')}.md")
    with open(md_path, "w") as f:
        f.write(report)

    # ── Save / overwrite dashboard ──
    dash_path = os.path.join(script_dir, "dashboard.html")
    with open(dash_path, "w") as f:
        f.write(dashboard)

    # ── Web app manifest (for "Add to Home Screen") ──
    manifest = {
        "name": "Odyssey",
        "short_name": "Odyssey",
        "description": "Weekly equity signal dashboard",
        "start_url": "./",
        "scope": "./",
        "display": "standalone",
        "orientation": "portrait",
        "background_color": "#0f1115",
        "theme_color": "#0f1115",
        "icons": [
            {"src": "icon-180.png", "sizes": "180x180", "type": "image/png"},
            {"src": "icon-512.png", "sizes": "512x512", "type": "image/png",
             "purpose": "any maskable"},
        ],
    }
    manifest_path = os.path.join(script_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Report saved    → {md_path}")
    print(f"✅ Dashboard saved → {dash_path}")
    print(f"✅ Manifest saved  → {manifest_path}\n")


if __name__ == "__main__":
    main()
