"""
Odyssey — Scout
Screens AI/chip supply chain and clean energy stocks for trading signals.
Flags unusual volume, price momentum, RSI extremes, and MA crossovers, then
scores each ticker's net lean (bullish vs bearish).

Outputs on each run:
  • scout_reports/scout_YYYY-MM-DD.md     — dated markdown report
  • scout_reports/scout_YYYY-MM-DD.json   — same data, structured for Dexter (Research Agent)
  • index.html                            — minimalist live dashboard (overwritten each run)

Usage:
    python scout_agent.py

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
TOP_TIER_THRESHOLD     = 2       # |net| >= this is "Signals Worth a Look" (top tier)

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


def is_top_tier(s: dict) -> bool:
    """True if this ticker belongs in Scout's 'Signals Worth a Look' tier."""
    return abs(s["net"]) >= TOP_TIER_THRESHOLD


def format_pct(val) -> str:
    if val is None:
        return "—"
    arrow = "▲" if val > 0 else "▼" if val < 0 else "—"
    return f"{arrow} {abs(val):.1f}%"


# ─── MARKDOWN REPORT ───────────────────────────────────────────────────────────

def build_report(results: dict) -> str:
    now = datetime.now()
    lines = []

    lines.append("# Odyssey — Scout Report")
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
        for s in group if s and is_top_tier(s)
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
    lines.append("*Generated by Odyssey Scout. Not financial advice. You make the calls.*")

    return "\n".join(lines)


# ─── JSON HANDOFF (for Dexter) ──────────────────────────────────────────────

def build_json(results: dict) -> dict:
    """Flat, structured version of the scan for Dexter (Research Agent) to consume.
    Includes a `top_tier` flag per ticker so Dexter knows what to research."""
    now = datetime.now()
    tickers = []
    for theme, group in results.items():
        for s in group:
            if s is None:
                continue
            tickers.append({
                "ticker":        s["ticker"],
                "theme":         theme,
                "price":         s["price"],
                "1d_pct":        s["1d_pct"],
                "5d_pct":        s["5d_pct"],
                "20d_pct":       s["20d_pct"],
                "vol_ratio":     s["vol_ratio"],
                "rsi":           s["rsi"],
                "ma_50":         s["ma_50"],
                "ma_200":        s["ma_200"],
                "pct_from_high": s["pct_from_high"],
                "pct_from_low":  s["pct_from_low"],
                "flags":         s["flags"],
                "signal_count":  s["signal_count"],
                "net":           s["net"],
                "tier":          tier(s),
                "top_tier":      is_top_tier(s),
                "is_north_star": s["ticker"] == NORTH_STAR,
            })
    return {
        "generated_at": now.isoformat(),
        "top_tier_threshold": TOP_TIER_THRESHOLD,
        "tickers": tickers,
    }


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
    """Card-grid dashboard: dark/minimal theme, emerald accent, client-side
    sort + filter (no backend needed — all tickers ship as embedded JSON and
    a small vanilla-JS layer handles re-rendering)."""
    now = datetime.now()
    total = sum(len(v) for v in results.values())
    flagged = sum(1 for g in results.values() for s in g if s and s["signal_count"] > 0)
    bull = sum(1 for g in results.values() for s in g if s and s["net"] > 0)
    bear = sum(1 for g in results.values() for s in g if s and s["net"] < 0)

    cards = []
    themes_seen = []
    for theme, group in results.items():
        if theme not in themes_seen:
            themes_seen.append(theme)
        for s in group:
            if s is None:
                continue
            cls = _tier_css(s)
            full_label = tier(s)
            label = full_label.split(" ", 1)[1] if " " in full_label else full_label
            cards.append({
                "ticker":        s["ticker"],
                "theme":         theme,
                "price":         s["price"],
                "pct1d":         s["1d_pct"],
                "pct5d":         s["5d_pct"],
                "pct20d":        s["20d_pct"],
                "vol_ratio":     s["vol_ratio"],
                "rsi":           s["rsi"],
                "pct_from_high": s["pct_from_high"],
                "pct_from_low":  s["pct_from_low"],
                "flags":         s["flags"],
                "net":           s["net"],
                "signal_count":  s["signal_count"],
                "tier_label":    label,
                "tier_class":    cls,
                "is_north_star": s["ticker"] == NORTH_STAR,
            })

    data_json = json.dumps(cards)
    themes_json = json.dumps(themes_seen)

    template = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Odyssey</title>
<meta name="theme-color" content="#0a0d0c">
<link rel="icon" type="image/png" href="favicon.png">
<link rel="icon" type="image/svg+xml" href="icon.svg">
<link rel="apple-touch-icon" href="icon-180.png">
<link rel="manifest" href="manifest.json">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Odyssey">
<style>
  :root {
    --bg:#0a0d0c; --bg-alt:#0e1311; --panel:#111614; --border:#1d2522;
    --text:#e8ece9; --muted:#7c8985; --green:#34d399; --green-soft:rgba(52,211,153,.14);
    --red:#f87171; --red-soft:rgba(248,113,113,.14); --yellow:#fbbf24; --yellow-soft:rgba(251,191,36,.14);
    --blue:#60a5fa;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
    font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  .wrap { max-width:1200px; margin:0 auto; padding:32px 20px 64px; }
  h1 { font-size:21px; font-weight:700; margin:0; letter-spacing:.2px; }
  h1 span { color:var(--green); }
  .ts { color:var(--muted); font-size:13px; margin:2px 0 22px; }
  .summary { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:20px; }
  .stat { background:var(--panel); border:1px solid var(--border); border-radius:12px;
    padding:12px 18px; min-width:90px; flex:0 1 auto; }
  .stat .n { font-size:22px; font-weight:700; font-variant-numeric:tabular-nums; }
  .stat .l { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.5px; margin-top:2px; }
  .stat.flagged .n { color:var(--green); }
  .stat.bull .n { color:var(--green); } .stat.bear .n { color:var(--red); }
  .controls { display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-bottom:22px; }
  .filter-group { display:flex; gap:3px; background:var(--panel); border:1px solid var(--border);
    border-radius:10px; padding:4px; }
  .filter-btn { padding:7px 13px; border-radius:7px; font-size:12.5px; font-weight:600;
    color:var(--muted); background:transparent; border:none; cursor:pointer; white-space:nowrap; }
  .filter-btn.active { background:var(--green-soft); color:var(--green); }
  select { background:var(--panel); border:1px solid var(--border); color:var(--text);
    border-radius:9px; padding:8px 12px; font-size:12.5px; font-weight:600; cursor:pointer; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(250px,1fr)); gap:14px; }
  .card { background:var(--panel); border:1px solid var(--border); border-left:3px solid var(--border);
    border-radius:14px; padding:16px; transition:border-color .15s, transform .15s; }
  .card:hover { transform:translateY(-2px); }
  .card.bull, .card.bull-lean { border-left-color:var(--green); }
  .card.bear, .card.bear-lean { border-left-color:var(--red); }
  .card.mixed { border-left-color:var(--yellow); }
  .card-head { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px; }
  .tkr { font-size:18px; font-weight:700; letter-spacing:.2px; }
  .star { color:var(--green); margin-left:4px; font-size:14px; }
  .theme-tag { font-size:10.5px; color:var(--muted); text-transform:uppercase;
    letter-spacing:.4px; margin-top:3px; }
  .pill { display:inline-block; padding:3px 10px; border-radius:20px; font-size:11.5px;
    font-weight:700; white-space:nowrap; }
  .pill.bull { background:var(--green-soft); color:var(--green); }
  .pill.bull-lean { background:rgba(52,211,153,.08); color:var(--green); }
  .pill.bear { background:var(--red-soft); color:var(--red); }
  .pill.bear-lean { background:rgba(248,113,113,.08); color:var(--red); }
  .pill.mixed { background:var(--yellow-soft); color:var(--yellow); }
  .pill.quiet { background:var(--bg-alt); color:var(--muted); }
  .price-row { display:flex; align-items:baseline; gap:9px; margin-bottom:12px; }
  .price { font-size:21px; font-weight:700; font-variant-numeric:tabular-nums; }
  .pct { font-size:13px; font-weight:700; }
  .pct.up { color:var(--green); } .pct.down { color:var(--red); } .pct.flat { color:var(--muted); }
  .stats-row { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; padding:10px 0;
    border-top:1px solid var(--border); }
  .stats-row + .stats-row { border-top:none; padding-top:0; }
  .stat-mini { text-align:center; }
  .stat-mini .v { font-size:13px; font-weight:700; font-variant-numeric:tabular-nums; }
  .stat-mini .v.hot { color:var(--red); } .stat-mini .v.cold { color:var(--blue); }
  .stat-mini .k { font-size:9.5px; color:var(--muted); text-transform:uppercase;
    letter-spacing:.4px; margin-top:2px; }
  .flags { display:flex; flex-wrap:wrap; gap:6px; margin-top:12px; }
  .flag-tag { font-size:11px; color:var(--muted); background:var(--bg-alt);
    border:1px solid var(--border); padding:3px 9px; border-radius:7px; }
  .empty { color:var(--muted); padding:40px 0; text-align:center; font-size:14px; grid-column:1/-1; }
  .legend { color:var(--muted); font-size:12px; margin:24px 0 0; }
  .legend b.bull { color:var(--green); } .legend b.bear { color:var(--red); } .legend b.mixed { color:var(--yellow); }
  footer { color:var(--muted); font-size:12px; margin-top:32px; border-top:1px solid var(--border); padding-top:16px; }
  @media (max-width:480px) {
    .wrap { padding:20px 14px 48px; }
    .stat { flex:1 1 40%; }
    .filter-btn { padding:7px 10px; font-size:12px; }
  }
</style></head>
<body><div class="wrap">
  <header><h1>Odyssey <span>·</span> Scout Dashboard</h1></header>
  <div class="ts">__TIMESTAMP__</div>
  <div class="summary">
    <div class="stat"><div class="n">__TOTAL__</div><div class="l">Scanned</div></div>
    <div class="stat flagged"><div class="n">__FLAGGED__</div><div class="l">Flagged</div></div>
    <div class="stat bull"><div class="n">__BULL__</div><div class="l">Bullish lean</div></div>
    <div class="stat bear"><div class="n">__BEAR__</div><div class="l">Bearish lean</div></div>
  </div>
  <div class="controls">
    <div class="filter-group" id="tierFilters">
      <button class="filter-btn active" data-tier="all">All</button>
      <button class="filter-btn" data-tier="bull">Bullish</button>
      <button class="filter-btn" data-tier="bear">Bearish</button>
      <button class="filter-btn" data-tier="mixed">Mixed</button>
      <button class="filter-btn" data-tier="quiet">Quiet</button>
    </div>
    <select id="themeFilter"><option value="all">All themes</option></select>
    <select id="sortBy">
      <option value="net">Sort: Signal strength</option>
      <option value="ticker">Sort: Ticker A–Z</option>
      <option value="pct1d">Sort: 1-day move</option>
      <option value="rsi">Sort: RSI</option>
      <option value="price">Sort: Price</option>
    </select>
  </div>
  <div class="grid" id="grid"></div>
  <p class="legend"><b class="bull">● Green</b> = bullish lean (momentum, near highs, golden cross, oversold bounce) ·
    <b class="bear">● Red</b> = bearish lean (overbought/toppy, selling off, death cross, near lows) ·
    <b class="mixed">● Yellow</b> = mixed · ● Grey = quiet. ★ = north star (MU).</p>
  <footer>Generated by Odyssey Scout. Not financial advice — signals only. You make the calls.</footer>
</div>
<script>
  const DATA = __DATA_JSON__;
  const THEMES = __THEMES_JSON__;
  const RSI_OB = __RSI_OB__, RSI_OS = __RSI_OS__;
  const state = { tier: 'all', theme: 'all', sort: 'net' };

  const themeSel = document.getElementById('themeFilter');
  THEMES.forEach(t => {
    const o = document.createElement('option');
    o.value = t; o.textContent = t;
    themeSel.appendChild(o);
  });

  function fmtArrowPct(v) {
    if (v === null || v === undefined) return '—';
    const arrow = v > 0 ? '▲' : v < 0 ? '▼' : '—';
    return `${arrow} ${Math.abs(v).toFixed(1)}%`;
  }

  function fmtPct(v) {
    if (v === null || v === undefined) return '<span class="pct flat">—</span>';
    const cls = v > 0 ? 'up' : v < 0 ? 'down' : 'flat';
    return `<span class="pct ${cls}">${fmtArrowPct(v)}</span>`;
  }

  function tierGroup(cls) {
    if (cls === 'bull' || cls === 'bull-lean') return 'bull';
    if (cls === 'bear' || cls === 'bear-lean') return 'bear';
    return cls;
  }

  function cardHtml(c) {
    const rsiCls = c.rsi >= RSI_OB ? 'hot' : c.rsi <= RSI_OS ? 'cold' : '';
    const flagsHtml = c.flags.length
      ? `<div class="flags">${c.flags.map(f => `<span class="flag-tag">${f}</span>`).join('')}</div>`
      : '';
    return `<div class="card ${c.tier_class}" data-tier="${tierGroup(c.tier_class)}" data-theme="${c.theme}">
      <div class="card-head">
        <div>
          <div class="tkr">${c.ticker}${c.is_north_star ? '<span class="star">★</span>' : ''}</div>
          <div class="theme-tag">${c.theme}</div>
        </div>
        <span class="pill ${c.tier_class}">${c.tier_label}</span>
      </div>
      <div class="price-row">
        <span class="price">$${c.price.toFixed(2)}</span>
        ${fmtPct(c.pct1d)}
      </div>
      <div class="stats-row">
        <div class="stat-mini"><div class="v">${fmtArrowPct(c.pct5d)}</div><div class="k">5D</div></div>
        <div class="stat-mini"><div class="v">${fmtArrowPct(c.pct20d)}</div><div class="k">20D</div></div>
        <div class="stat-mini"><div class="v ${rsiCls}">${c.rsi}</div><div class="k">RSI</div></div>
      </div>
      <div class="stats-row">
        <div class="stat-mini"><div class="v">${c.vol_ratio.toFixed(1)}x</div><div class="k">Vol/Avg</div></div>
        <div class="stat-mini"><div class="v">${c.pct_from_high.toFixed(1)}%</div><div class="k">From High</div></div>
        <div class="stat-mini"><div class="v">+${c.pct_from_low.toFixed(1)}%</div><div class="k">From Low</div></div>
      </div>
      ${flagsHtml}
    </div>`;
  }

  function sortData(arr) {
    const s = state.sort;
    const copy = [...arr];
    if (s === 'ticker') copy.sort((a, b) => a.ticker.localeCompare(b.ticker));
    else if (s === 'pct1d') copy.sort((a, b) => (b.pct1d ?? -999) - (a.pct1d ?? -999));
    else if (s === 'rsi') copy.sort((a, b) => b.rsi - a.rsi);
    else if (s === 'price') copy.sort((a, b) => b.price - a.price);
    else copy.sort((a, b) => (Math.abs(b.net) * 10 + b.signal_count) - (Math.abs(a.net) * 10 + a.signal_count));
    return copy;
  }

  function render() {
    let filtered = DATA.filter(c => {
      const tierOk = state.tier === 'all' || tierGroup(c.tier_class) === state.tier;
      const themeOk = state.theme === 'all' || c.theme === state.theme;
      return tierOk && themeOk;
    });
    filtered = sortData(filtered);
    const grid = document.getElementById('grid');
    grid.innerHTML = filtered.length
      ? filtered.map(cardHtml).join('')
      : '<div class="empty">No tickers match this filter.</div>';
  }

  document.getElementById('tierFilters').addEventListener('click', e => {
    const btn = e.target.closest('.filter-btn');
    if (!btn) return;
    document.querySelectorAll('#tierFilters .filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.tier = btn.dataset.tier;
    render();
  });
  themeSel.addEventListener('change', e => { state.theme = e.target.value; render(); });
  document.getElementById('sortBy').addEventListener('change', e => { state.sort = e.target.value; render(); });

  render();
</script>
</body></html>"""

    return (template
        .replace("__TIMESTAMP__", now.strftime('%A, %B %d, %Y · %I:%M %p'))
        .replace("__TOTAL__", str(total))
        .replace("__FLAGGED__", str(flagged))
        .replace("__BULL__", str(bull))
        .replace("__BEAR__", str(bear))
        .replace("__DATA_JSON__", data_json)
        .replace("__THEMES_JSON__", themes_json)
        .replace("__RSI_OB__", str(RSI_OVERBOUGHT))
        .replace("__RSI_OS__", str(RSI_OVERSOLD))
    )


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("\n🔍 Odyssey Scout starting...\n")

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
    handoff = build_json(results)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Save dated markdown report + JSON handoff for Dexter ──
    reports_dir = os.path.join(script_dir, "scout_reports")
    os.makedirs(reports_dir, exist_ok=True)
    today_str = date.today().strftime('%Y-%m-%d')
    md_path = os.path.join(reports_dir, f"scout_{today_str}.md")
    json_path = os.path.join(reports_dir, f"scout_{today_str}.json")
    with open(md_path, "w") as f:
        f.write(report)
    with open(json_path, "w") as f:
        json.dump(handoff, f, indent=2)

    # ── Save / overwrite dashboard ──
    dash_path = os.path.join(script_dir, "index.html")
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
        "background_color": "#0a0d0c",
        "theme_color": "#0a0d0c",
        "icons": [
            {"src": "icon-180.png", "sizes": "180x180", "type": "image/png"},
            {"src": "icon-512.png", "sizes": "512x512", "type": "image/png",
             "purpose": "any maskable"},
        ],
    }
    manifest_path = os.path.join(script_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    top_tier_count = sum(1 for t in handoff["tickers"] if t["top_tier"])
    print(f"✅ Report saved    → {md_path}")
    print(f"✅ JSON saved      → {json_path}  ({top_tier_count} top-tier for Dexter)")
    print(f"✅ Dashboard saved → {dash_path}")
    print(f"✅ Manifest saved  → {manifest_path}\n")


if __name__ == "__main__":
    main()
