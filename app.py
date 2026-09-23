"""BURAK CRYPTO RADAR V6.9 — OKX + BIST and read-only OKX account view.
No order placement, cancellation, transfers, or leverage execution.
"""
import base64
import hashlib
import hmac
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Burak Crypto Radar V6.8 — OKX + BIST", page_icon="📡", layout="wide")
st.markdown("""
<style>
@media (max-width: 600px) {
  .block-container {padding: 0.75rem 0.65rem 4rem; max-width: 100%;}
  h1 {font-size: 1.65rem !important; line-height: 1.2;}
  h2 {font-size: 1.35rem !important;}
  div[data-testid="stMetric"] {padding: 0.65rem; border: 1px solid rgba(128,128,128,.25); border-radius: 12px;}
  div[data-testid="stMetricValue"] {font-size: 1.25rem;}
  div[data-testid="stHorizontalBlock"] {gap: .4rem;}
  div[data-testid="stTabs"] button {white-space: nowrap;}
  div[data-testid="stDataFrame"] {max-width: 100%;}
  button[kind="primary"] {min-height: 46px;}
}
</style>
""", unsafe_allow_html=True)
def okx_private_get(path, credentials):
    """Read-only OKX v5 GET; no private POST/DELETE capabilities in this app."""
    timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    signature = base64.b64encode(hmac.new(
        credentials["secret_key"].encode("utf-8"),
        (timestamp + "GET" + path).encode("utf-8"),
        hashlib.sha256
    ).digest()).decode("ascii")
    headers = {
        "OK-ACCESS-KEY": credentials["api_key"],
        "OK-ACCESS-SIGN": signature,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": credentials["passphrase"],
        "Content-Type": "application/json",
    }
    if credentials.get("demo", False):
        headers["x-simulated-trading"] = "1"
    try:
        response = requests.get("https://www.okx.com" + path, headers=headers, timeout=18)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError("OKX bağlantısı kurulamadı; ağ erişimini ve API ayarlarını kontrol et.") from None
    if payload.get("code") != "0":
        code = str(payload.get("code", "bilinmiyor"))
        raise RuntimeError(f"OKX isteği reddetti (hata kodu: {code}). API yetkilerini, demo/canlı seçimini ve sunucu saatini kontrol et.")
    return payload.get("data", [])


def okx_account_secrets():
    try:
        cfg = st.secrets.get("okx", {})
        return {
            "api_key": str(cfg.get("api_key", "")).strip(),
            "secret_key": str(cfg.get("secret_key", "")).strip(),
            "passphrase": str(cfg.get("passphrase", "")),
            "dashboard_password": str(cfg.get("dashboard_password", "")),
            "demo": bool(cfg.get("demo", False)),
        }
    except (FileNotFoundError, KeyError):
        return {}


HEADERS = {"User-Agent": "BurakCryptoRadar/1.0", "accept": "application/json"}
STABLE = {"usdt", "usdc", "dai", "fdusd", "tusd", "usde", "usdd", "pyusd", "frax"}


def get_json(url, params=None, headers=None):
    resp = requests.get(url, params=params, headers=headers or HEADERS, timeout=18)
    resp.raise_for_status()
    return resp.json()


def technical(df, cfg=None):
    if len(df) < 205:
        return None
    close, high, low, vol = df.close, df.high, df.low, df.quote_volume
    e20, e50, e200 = [close.ewm(span=n, adjust=False).mean() for n in (20, 50, 200)]
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    up, down = high.diff(), -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr
    dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    vratio = vol.iloc[-1] / vol.iloc[-21:-1].mean() if vol.iloc[-21:-1].mean() > 0 else np.nan
    c = float(close.iloc[-1]); r = float(rsi.iloc[-1]); a = float(adx.iloc[-1]); v = float(vratio)
    if not all(np.isfinite(z) for z in (c, r, a, v)):
        return None
    score = sum([
        5 if c > e20.iloc[-1] else 0, 5 if c > e50.iloc[-1] else 0,
        5 if c > e200.iloc[-1] else 0, 5 if e20.iloc[-1] > e50.iloc[-1] else 0,
        5 if e50.iloc[-1] > e200.iloc[-1] else 0,
        5 if c > e20.iloc[-1] > e50.iloc[-1] else 0,
        10 if 50 <= r <= 65 else (6 if 65 < r <= 72 else (4 if 45 <= r < 50 else 0)),
        8 if macd.iloc[-1] > signal.iloc[-1] else 0,
        4 if hist.iloc[-1] > 0 else 0,
        3 if hist.iloc[-1] > hist.iloc[-2] else 0,
        5 if a > 20 else 0, 5 if a > 25 else 0,
        5 if plus_di.iloc[-1] > minus_di.iloc[-1] else 0,
        5 if v > 1.2 else 0, 5 if v > 1.5 else 0,
        5 if v > 2 else 0, 5 if v > 3 else 0,
    ])
    cfg = cfg or {"enabled": {k: True for k in ("EMA", "MACD", "DI", "ADX", "RSI", "Hacim")},
                  "adx_min": 20, "long_rsi": (45, 68), "short_rsi": (32, 55),
                  "volume_min": 1.2}
    long_tests = {"EMA": bool(c > e20.iloc[-1] > e50.iloc[-1]),
                  "MACD": bool(macd.iloc[-1] > signal.iloc[-1]),
                  "DI": bool(plus_di.iloc[-1] > minus_di.iloc[-1]),
                  "ADX": bool(a >= cfg["adx_min"]),
                  "RSI": bool(cfg["long_rsi"][0] <= r <= cfg["long_rsi"][1]),
                  "Hacim": bool(v >= cfg["volume_min"])}
    short_tests = {"EMA": bool(c < e20.iloc[-1] < e50.iloc[-1]),
                   "MACD": bool(macd.iloc[-1] < signal.iloc[-1]),
                   "DI": bool(minus_di.iloc[-1] > plus_di.iloc[-1]),
                   "ADX": bool(a >= cfg["adx_min"]),
                   "RSI": bool(cfg["short_rsi"][0] <= r <= cfg["short_rsi"][1]),
                   "Hacim": bool(v >= cfg["volume_min"])}
    active = [k for k, yes in cfg["enabled"].items() if yes]
    lc = sum(long_tests[k] for k in active)
    sc = sum(short_tests[k] for k in active)
    n = len(active)
    bullish = n > 0 and lc == n
    bearish = n > 0 and sc == n
    direction = "🟢 LONG" if bullish and not bearish else ("🔴 SHORT" if bearish and not bullish else "⚪ BEKLE")
    if direction != "⚪ BEKLE":
        stage, missing = direction, "—"
    elif n == 0:
        stage, missing = "⚪ BEKLE", "En az bir teknik koşul seç"
    elif lc == n - 1 and sc < n - 1:
        stage = "🟡 LONG adayı"
        missing = ", ".join(k for k in active if not long_tests[k])
    elif sc == n - 1 and lc < n - 1:
        stage = "🟠 SHORT adayı"
        missing = ", ".join(k for k in active if not short_tests[k])
    else:
        stage, missing = "⚪ BEKLE", ("LONG: " + ", ".join(k for k in active if not long_tests[k])
                                     + " | SHORT: " + ", ".join(k for k in active if not short_tests[k]))
    # The live candle may change before close.
    candle_time = df["date"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC")
    # 30 trend + 10 RSI + 15 MACD + 15 ADX + 20 volume = 90.
    return {"Sinyal": direction, "Fırsat durumu": stage, "Eksik koşul": missing,
            "LONG koşul": f"{lc}/{n}", "SHORT koşul": f"{sc}/{n}", "Sinyal mumu": candle_time,
            "Teknik skor": round(score / 90 * 100), "RSI": round(r, 1),
            "ADX": round(a, 1), "Hacim katı": round(v, 2),
            "EMA20 üstü": bool(c > e20.iloc[-1]), "EMA50 üstü": bool(c > e50.iloc[-1]),
            "EMA200 üstü": bool(c > e200.iloc[-1]), "Kapanış": c, "_long_tests": long_tests, "_short_tests": short_tests}


def fibonacci_check(df, price, side, atr_limit=1.0):
    closed = df[df["confirm"] == "1"].tail(100)
    if len(closed) < 30:
        return False, float("nan"), float("nan")
    hi, lo = float(closed.high.max()), float(closed.low.min())
    prev = closed.close.shift()
    tr = pd.concat([closed.high-closed.low, (closed.high-prev).abs(),
                    (closed.low-prev).abs()], axis=1).max(axis=1)
    atr = float(tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1])
    if hi <= lo or not np.isfinite(atr) or atr <= 0:
        return False, float("nan"), float("nan")
    levels = sorted(lo + (hi-lo)*x for x in (0, .236, .382, .5, .618, .786, 1))
    eligible = [x for x in levels if x <= price] if side == "LONG" else [x for x in levels if x >= price]
    if not eligible:
        return False, float("nan"), float("nan")
    level = max(eligible) if side == "LONG" else min(eligible)
    distance = abs(price-level)/atr
    return distance <= atr_limit, level, distance


def levels_and_risks(df, row):
    """Descriptive price zones from completed candles; not trade recommendations."""
    if df is None or len(df) < 205:
        return {}
    close = df["close"]
    high = df["high"]
    low = df["low"]
    last = float(close.iloc[-1])
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
    support = float(low.iloc[-21:-1].min())
    resistance = float(high.iloc[-21:-1].max())
    tr = pd.concat([high-low, (high-close.shift()).abs(),
                    (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1])
    mc = row.get("market_cap")
    fdv = row.get("fully_diluted_valuation")
    circulating = row.get("circulating_supply")
    total = row.get("total_supply")
    flags = []
    rsi_delta = close.diff()
    gain = rsi_delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-rsi_delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = float((100 - 100 / (1 + gain / loss.replace(0, np.nan))).iloc[-1])
    if np.isfinite(rsi) and rsi >= 70:
        flags.append("RSI ≥70: kısa vadeli aşırı alım göstergesi")
    if np.isfinite(rsi) and rsi <= 30:
        flags.append("RSI ≤30: güçlü satış baskısı göstergesi")
    if pd.notna(mc) and mc > 0 and pd.notna(fdv) and fdv / mc >= 2:
        flags.append("FDV/MC ≥2: arz seyrelmesi açısından incele")
    if pd.notna(circulating) and pd.notna(total) and total > 0 and circulating / total < .5:
        flags.append("Dolaşımdaki arz toplam arzın <%50'si")
    if last < ema200:
        flags.append("Fiyat EMA200 altında")
    if pd.notna(row.get("total_volume")) and row["total_volume"] < 10e6:
        flags.append("24s küresel hacim <$10m")
    return {"EMA20 ($)": ema20, "EMA50 ($)": ema50, "EMA200 ($)": ema200,
            "20 mum destek ($)": support, "20 mum direnç ($)": resistance,
            "ATR14 ($)": atr, "ATR14 %": 100 * atr / last if last > 0 else np.nan,
            "Dolaşım %": 100 * circulating / total if pd.notna(circulating)
            and pd.notna(total) and total > 0 else np.nan,
            "Risk notları": " | ".join(flags) if flags else "Tanımlı risk eşiği tetiklenmedi"}



def atr_scenario(indicators, levels, stop_mult, target_mult):
    direction = indicators.get("Sinyal")
    entry = float(indicators.get("Kapanış", np.nan))
    atr = float(levels.get("ATR14 ($)", np.nan))
    blank = {"Referans giriş ($)": np.nan, "Stop ($)": np.nan, "Hedef ($)": np.nan,
             "Stop uzaklık %": np.nan, "Hedef uzaklık %": np.nan, "Risk/Ödül": np.nan}
    if direction not in ("🟢 LONG", "🔴 SHORT") or not np.isfinite(entry) or not np.isfinite(atr) or entry <= 0 or atr <= 0:
        return blank
    sign = 1 if direction == "🟢 LONG" else -1
    stop = entry - sign * stop_mult * atr
    target = entry + sign * target_mult * atr
    if stop <= 0 or target <= 0:
        return blank
    return {"Referans giriş ($)": entry, "Stop ($)": stop, "Hedef ($)": target,
            "Stop uzaklık %": 100 * stop_mult * atr / entry,
            "Hedef uzaklık %": 100 * target_mult * atr / entry,
            "Risk/Ödül": target_mult / stop_mult}


def backtest_directional(df, hold_bars=4, fee_pct=.1, slip_pct=.05):
    """Non-overlapping directional signal study. Entry next candle open,
    exit after hold_bars at close. Costs charged at both ends.
    Historical spot OHLC is a proxy, NOT futures execution data."""
    if len(df) < 215:
        return pd.DataFrame()
    c, h, l, vol = df.close, df.high, df.low, df.quote_volume
    e20 = c.ewm(span=20, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    up, down = h.diff(), -l.diff()
    pdm = up.where((up > down) & (up > 0), 0.0)
    mdm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean().replace(0, np.nan)
    pdi = 100 * pdm.ewm(alpha=1/14, adjust=False).mean() / atr
    mdi = 100 * mdm.ewm(alpha=1/14, adjust=False).mean() / atr
    dx = 100 * (pdi-mdi).abs() / (pdi+mdi).replace(0, np.nan)
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    vr = vol / vol.shift(1).rolling(20).mean().replace(0, np.nan)
    long_cond = ((c > e20) & (e20 > e50) & (macd > sig)
                 & (pdi > mdi) & (adx >= 20) & rsi.between(45, 68)
                 & (vr >= 1.2))
    short_cond = ((c < e20) & (e20 < e50) & (macd < sig)
                  & (mdi > pdi) & (adx >= 20) & rsi.between(32, 55)
                  & (vr >= 1.2))
    state = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    trades = []
    i = 205
    while i + hold_bars < len(df):
        direction = int(state[i])
        # One trade per new directional event; no repeated entries on same signal.
        if direction == 0 or state[i-1] == direction:
            i += 1
            continue
        entry_idx = i + 1
        exit_idx = i + hold_bars
        entry = float(df.open.iloc[entry_idx])
        exit_price = float(c.iloc[exit_idx])
        if not (np.isfinite(entry) and np.isfinite(exit_price) and entry > 0):
            i += 1
            continue
        gross = direction * (exit_price / entry - 1) * 100
        net = gross - 2 * (fee_pct + slip_pct)
        trades.append({"Sinyal UTC": df.date.iloc[i], "Yön": "LONG" if direction == 1 else "SHORT",
                       "Giriş UTC": df.date.iloc[entry_idx], "Çıkış UTC": df.date.iloc[exit_idx],
                       "Giriş ($)": entry, "Çıkış ($)": exit_price,
                       "Brüt %": round(gross, 3), "Net %": round(net, 3)})
        i = exit_idx + 1
    return pd.DataFrame(trades)




def nkral_signals(df, sensitivity=1.0, atr_period=10):
    """NKRAL1 original close/ATR trailing stop crossover, evaluated on supplied bars."""
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    prev = c.shift(1)
    tr = pd.concat([h-l, (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    # Pine ta.atr: Wilder RMA seeded with first ATR-length SMA.
    atr = pd.Series(np.nan, index=df.index, dtype=float)
    if len(df) >= atr_period:
        atr.iloc[atr_period-1] = tr.iloc[:atr_period].mean()
        for i in range(atr_period, len(df)):
            atr.iloc[i] = (atr.iloc[i-1] * (atr_period-1) + tr.iloc[i]) / atr_period
    stops = np.full(len(df), np.nan)
    buy = np.zeros(len(df), dtype=bool)
    sell = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        if not np.isfinite(atr.iloc[i]):
            continue
        old = stops[i-1] if i > 0 and np.isfinite(stops[i-1]) else 0.
        price = float(c.iloc[i])
        previous_price = float(c.iloc[i-1]) if i else np.nan
        loss = sensitivity * float(atr.iloc[i])
        if price > old and previous_price > old:
            stop = max(old, price-loss)
        elif price < old and previous_price < old:
            stop = min(old, price+loss)
        else:
            stop = price-loss if price > old else price+loss
        stops[i] = stop
        if i > 0 and np.isfinite(stops[i-1]):
            buy[i] = price > stop and previous_price <= old and price > stop
            sell[i] = price < stop and previous_price >= old and price < stop
    return pd.DataFrame({"NK AL": buy, "NK SAT": sell,
                         "NK stop ($)": stops}, index=df.index)


def strategy_states(df, cfg, sensitivity=1., atr_period=10):
    """Closed-bar historical comparison. Same-timeframe technical filters;
    Fibonacci cross-timeframe confirmation is excluded and explicitly disclosed."""
    c, h, l, vol = df.close, df.high, df.low, df.quote_volume
    e20 = c.ewm(span=20, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    up, down = h.diff(), -l.diff()
    pdm = up.where((up > down) & (up > 0), 0.)
    mdm = down.where((down > up) & (down > 0), 0.)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean().replace(0, np.nan)
    pdi = 100 * pdm.ewm(alpha=1/14, adjust=False).mean() / atr
    mdi = 100 * mdm.ewm(alpha=1/14, adjust=False).mean() / atr
    dx = 100 * (pdi-mdi).abs() / (pdi+mdi).replace(0, np.nan)
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    vr = vol / vol.shift(1).rolling(20).mean().replace(0, np.nan)
    long_checks = {"EMA": (c > e20) & (e20 > e50),
                   "MACD": macd > sig, "DI": pdi > mdi,
                   "ADX": adx >= cfg["adx_min"],
                   "RSI": rsi.between(*cfg["long_rsi"]),
                   "Hacim": vr >= cfg["volume_min"]}
    short_checks = {"EMA": (c < e20) & (e20 < e50),
                    "MACD": macd < sig, "DI": mdi > pdi,
                    "ADX": adx >= cfg["adx_min"],
                    "RSI": rsi.between(*cfg["short_rsi"]),
                    "Hacim": vr >= cfg["volume_min"]}
    active = [k for k, enabled in cfg["enabled"].items() if enabled]
    long_ok = pd.Series(False, index=df.index)
    short_ok = pd.Series(False, index=df.index)
    if active:
        long_ok = pd.concat([long_checks[k] for k in active], axis=1).all(axis=1)
        short_ok = pd.concat([short_checks[k] for k in active], axis=1).all(axis=1)
    nk = nkral_signals(df, sensitivity, atr_period)
    radar = np.where(long_ok & ~short_ok, 1, np.where(short_ok & ~long_ok, -1, 0))
    nk_state = np.where(nk["NK AL"], 1, np.where(nk["NK SAT"], -1, 0))
    # Hybrid requires NKRAL crossing AND radar technical alignment on the same closed bar.
    hybrid = np.where((nk_state == 1) & (radar == 1), 1,
                      np.where((nk_state == -1) & (radar == -1), -1, 0))
    return {"Mevcut Radar (teknik)": radar, "NKRAL1": nk_state,
            "Hibrit": hybrid}, nk


def strategy_trade_test(df, state, hold_bars, fee_pct, slip_pct):
    """Next-bar-open entry; fixed holding-period close exit; no overlapping positions."""
    trades = []
    i = 205
    while i + hold_bars < len(df):
        direction = int(state[i])
        if direction == 0 or (i > 0 and int(state[i-1]) == direction):
            i += 1
            continue
        entry_i, exit_i = i + 1, i + hold_bars
        entry, exit_price = float(df.open.iloc[entry_i]), float(df.close.iloc[exit_i])
        if entry <= 0 or not np.isfinite(entry) or not np.isfinite(exit_price):
            i += 1
            continue
        gross = direction * (exit_price / entry - 1) * 100
        net = gross - 2 * (fee_pct + slip_pct)
        trades.append({"Sinyal UTC": df.date.iloc[i], "Yön": "LONG" if direction == 1 else "SHORT",
                       "Giriş UTC": df.date.iloc[entry_i], "Çıkış UTC": df.date.iloc[exit_i],
                       "Giriş ($)": entry, "Çıkış ($)": exit_price,
                       "Brüt %": gross, "Net %": net})
        i = exit_i + 1
    return pd.DataFrame(trades)


def compare_strategies_ui(cfg):
    st.subheader("🧪 Mevcut Radar / NKRAL1 / Hibrit — Geçmiş Veri Karşılaştırması")
    st.caption("1H ve 4H ayrı test edilir. Yalnızca OKX'ten alınan son 300 mumun kapanmış olanları kullanılır; "
               "205 mum ısınma sonrasında kalan kısa örnek test edilir. Fibonacci 1H/4H teyitleri bu "
               "karşılaştırmaya DAHİL DEĞİLDİR; 'Mevcut Radar' burada yalnızca seçili teknik koşullardır.")
    t1, t2, t3 = st.columns(3)
    with t1:
        coin = st.text_input("Test coin", "BTC", key="comparison_coin").strip().upper()
    with t2:
        hold = st.slider("Pozisyon süresi (mum)", 1, 12, 4, key="comparison_hold")
    with t3:
        fee = st.number_input("Tek yön komisyon (%)", 0., 1., .05, .01, key="comparison_fee")
    t4, t5 = st.columns(2)
    with t4:
        slip = st.number_input("Tek yön kayma (%)", 0., 1., .05, .01, key="comparison_slip")
    with t5:
        nk_sensitivity = st.number_input("NKRAL hassasiyet", .1, 10., 1., .1, key="comparison_nk_sens")
        nk_period = st.number_input("NKRAL ATR periyodu", 1, 100, 10, key="comparison_nk_atr")
    if not st.button("▶️ 1H ve 4H karşılaştırmasını çalıştır", key="comparison_run"):
        return
    base = coin.removesuffix("-USDT-SWAP").removesuffix("-USDT")
    if not base or not base.replace("-", "").isalnum():
        st.error("Geçerli coin sembolü gir.")
        return
    rows, details = [], []
    for tf in ("1h", "4h"):
        try:
            frame = okx_candles(base + "-USDT-SWAP", tf)
            frame = frame[frame["confirm"] == "1"].reset_index(drop=True)
            if len(frame) < 215:
                st.warning(f"{tf}: Yeterli kapanmış mum yok ({len(frame)}).")
                continue
            states, nk = strategy_states(frame, cfg, nk_sensitivity, int(nk_period))
            for name, state in states.items():
                trades = strategy_trade_test(frame, state, hold, fee, slip)
                n = len(trades)
                wins = int((trades["Net %"] > 0).sum()) if n else 0
                profits = trades["Net %"].clip(lower=0).sum() if n else 0.
                losses = -trades["Net %"].clip(upper=0).sum() if n else 0.
                curve = (1 + trades["Net %"] / 100).cumprod() if n else pd.Series(dtype=float)
                dd = ((curve / curve.cummax()) - 1).min() * 100 if n else np.nan
                rows.append({"Zaman": tf, "Strateji": name, "İşlem": n,
                             "Kazanma %": round(100*wins/n, 2) if n else np.nan,
                             "Ort. net %": round(trades["Net %"].mean(), 3) if n else np.nan,
                             "Profit factor": round(profits/losses, 2) if losses > 0 else np.nan,
                             "Bileşik net %": round((curve.iloc[-1]-1)*100, 2) if n else np.nan,
                             "Maks. düşüş %": round(dd, 2) if n else np.nan,
                             "İlk mum UTC": frame.date.iloc[205],
                             "Son mum UTC": frame.date.iloc[-1]})
                if n:
                    trades.insert(0, "Zaman", tf)
                    trades.insert(1, "Strateji", name)
                    details.append(trades)
        except Exception as exc:
            st.warning(f"{tf} test edilemedi: {type(exc).__name__}: {exc}")
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.warning("Bu bir sınırlı örneklem araştırmasıdır; işlem sayısı düşükse oranlar güvenilir değildir. "
                   "Fonlama, likidasyon, spread değişimi, stop/hedef ve gerçek emir gerçekleşmesi modellenmez. "
                   "NKRAL1 varsayılan normal mum kapanışı kullanılır; Heikin Ashi seçeneği uygulanmaz. "
                   "Aynı anda tek pozisyon, sonraki mum açılışında giriş ve seçilen mum sonunda çıkış varsayılır.")
        if details:
            st.download_button("📥 Karşılaştırma işlemlerini CSV indir",
                               pd.concat(details, ignore_index=True).to_csv(index=False).encode("utf-8-sig"),
                               "burak_strateji_karsilastirma.csv", "text/csv",
                               key="comparison_export")

@st.cache_data(ttl=15, show_spinner=False)
def okx_public(path, params=None):
    payload = get_json("https://www.okx.com" + path, params)
    if payload.get("code") != "0":
        raise RuntimeError(f"OKX {payload.get('code')}: {payload.get('msg')}")
    return payload.get("data", [])


@st.cache_data(ttl=15, show_spinner=False)
def okx_perpetual_universe():
    instruments = okx_public("/api/v5/public/instruments", {"instType": "SWAP"})
    tickers = okx_public("/api/v5/market/tickers", {"instType": "SWAP"})
    by_id = {x["instId"]: x for x in tickers}
    records = []
    for inst in instruments:
        inst_id = inst.get("instId", "")
        if inst.get("state") != "live" or not inst_id.endswith("-USDT-SWAP"):
            continue
        ticker = by_id.get(inst_id)
        if not ticker:
            continue
        try:
            last = float(ticker["last"])
            base_volume = float(ticker.get("volCcy24h") or 0)
            volume_usd = last * base_volume
        except (TypeError, ValueError, KeyError):
            continue
        if last <= 0 or volume_usd <= 0:
            continue
        records.append({"Parite": inst_id, "Sembol": inst_id.removesuffix("-USDT-SWAP"),
                        "Fiyat ($)": last, "24s hacim yaklaşık ($)": volume_usd,
                        "Ticker UTC": pd.to_datetime(int(ticker["ts"]), unit="ms", utc=True)})
    return pd.DataFrame(records).sort_values("24s hacim yaklaşık ($)", ascending=False)


@st.cache_data(ttl=15, show_spinner=False)
def okx_candles(inst_id, interval):
    bar = {"1h": "1H", "4h": "4H", "1d": "1Dutc"}[interval]
    data = okx_public("/api/v5/market/candles",
                      {"instId": inst_id, "bar": bar, "limit": "300"})
    if not data:
        raise ValueError("OKX mum verisi boş")
    # OKX: ts,o,h,l,c,vol,volCcy,volCcyQuote,confirm; newest first.
    rows = [r for r in data if len(r) >= 9]
    if not rows:
        raise ValueError("Tamamlanmış OKX mumu yok")
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close",
                                      "volume", "volCcy", "quote_volume", "confirm"])
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(pd.to_numeric(df.ts), unit="ms", utc=True)
    return df.sort_values("date").dropna(subset=["open", "high", "low", "close", "quote_volume"]).reset_index(drop=True)


@st.cache_data(ttl=60, show_spinner=False)
def okx_derivatives(inst_id):
    funding = okx_public("/api/v5/public/funding-rate", {"instId": inst_id})
    oi = okx_public("/api/v5/public/open-interest",
                    {"instType": "SWAP", "instId": inst_id})
    fr = float(funding[0]["fundingRate"]) * 100 if funding and funding[0].get("fundingRate") else np.nan
    oi_usd = float(oi[0]["oiUsd"]) if oi and oi[0].get("oiUsd") else np.nan
    return {"Funding %": fr, "OI ($)": oi_usd}



st.title("📡 BURAK CRYPTO RADAR V6.8 — OKX + BIST")
st.caption("Yalnızca OKX USDT perpetual verileri • LONG / SHORT araştırma sinyalleri • Otomatik emir göndermez")
with st.sidebar:
    st.header("🎛️ Radar koşulları")
    strategy_mode = st.selectbox("🧭 Strateji seçimi",
                                 ["Mevcut Radar", "NKRAL1", "Hibrit"],
                                 key="strategy_mode")
    nk_sens = st.number_input("NKRAL ATR hassasiyeti — stop mesafesi", .1, 10., 1., .1, key="nk_sens", help="ATR çarpanı: düşük değer fiyatı daha yakından izler ve daha sık kesişim üretebilir; yüksek değer daha geniş stop verir.")
    nk_atr = st.number_input("NKRAL ATR periyodu — oynaklık süresi", 1, 100, 10, key="nk_atr", help="ATR hesaplamasında kullanılan mum sayısı; varsayılan 10.")
    st.caption("NKRAL1: ATR trailing stop kesişimi. Hibrit: NKRAL1 kesişimi ve mevcut radar aynı yönde.")
    st.caption("Bu ayarlar OKX LONG/SHORT radarı ve manuel coin analizine uygulanır. Diğer sekmelerin hesaplamaları bağımsızdır.")
    with st.expander("🟢🔴 Teknik teyitler", expanded=True):
        technical_info = {
            "EMA": ("Trend yönü", "Üstel hareketli ortalama. LONG: fiyat EMA20 > EMA50; SHORT: fiyat EMA20 < EMA50."),
            "MACD": ("Momentum", "MACD çizgisi sinyal çizgisinin üstündeyse LONG, altındaysa SHORT yönünü destekler."),
            "DI": ("Alıcı / satıcı baskısı", "+DI > -DI alış, -DI > +DI satış yönünü destekler."),
            "ADX": ("Trend gücü", "Trendin yönünü değil gücünü ölçer. Seçtiğin minimum ADX değerinin üstü aranır."),
            "RSI": ("Göreli güç", "Fiyat momentumunu 0–100 aralığında ölçer. LONG/SHORT için ayrı RSI aralıkları kullanılır."),
            "Hacim": ("İşlem yoğunluğu", "Güncel mumun yaklaşık hacmini önceki 20 mumun ortalamasıyla karşılaştırır.")
        }
        enabled = {}
        for name, (short_desc, full_desc) in technical_info.items():
            enabled[name] = st.checkbox(f"{name} — {short_desc}", value=True,
                                        key="condition_" + name, help=full_desc)
        st.caption("Açıklamanın ayrıntısı için ⓘ simgesine dokun. İşaretini kaldırdığın koşul sinyal hesabından çıkarılır; tabloda görünmeye devam eder.")
    with st.expander("📐 Fibonacci teyitleri", expanded=True):
        fib_1h = st.checkbox("1 saatlik Fibonacci — fiyat seviyeleri", value=True, key="condition_fib_1h", help="1 saatlik kapanmış mumların fiyat aralığındaki Fibonacci seviyesine yakınlık teyidi.")
        fib_4h = st.checkbox("4 saatlik Fibonacci — fiyat seviyeleri", value=True, key="condition_fib_4h", help="4 saatlik kapanmış mumların fiyat aralığındaki Fibonacci seviyesine yakınlık teyidi.")
        fib_atr = st.slider("Fib yakınlığı (ATR) — seviye toleransı", 0.25, 3.0, 1.0, 0.25, key="condition_fib_atr", help="Fiyatın en yakın uygun Fibonacci seviyesine uzaklığı kaç ATR olabilecek? Düşük değer daha sıkı teyittir.")
    with st.expander("📊 Sinyal eşikleri", expanded=False):
        adx_min = st.slider("Minimum ADX — trend gücü eşiği", 10, 45, 20, key="condition_adx_min", help="ADX bu değere eşit veya daha yüksekse trend gücü koşulu geçer.")
        long_rsi = st.slider("LONG RSI aralığı — alış momentumu", 0, 100, (45, 68), key="condition_long_rsi", help="RSI seçilen iki sınır arasında olmalı; varsayılan 45–68.")
        short_rsi = st.slider("SHORT RSI aralığı — satış momentumu", 0, 100, (32, 55), key="condition_short_rsi", help="RSI seçilen iki sınır arasında olmalı; varsayılan 32–55.")
        volume_min = st.slider("Minimum hacim katı — ortalamaya göre", 0.5, 5.0, 1.2, 0.1, key="condition_volume_min", help="1,2 = güncel mumun tahmini hacmi önceki 20 mum ortalamasının en az %120’si.")
    condition_cfg = {"strategy": strategy_mode, "nk_sens": nk_sens, "nk_atr": nk_atr, "enabled": enabled, "fib": {"1h": fib_1h, "4h": fib_4h},
                     "fib_atr": fib_atr, "adx_min": adx_min,
                     "long_rsi": long_rsi, "short_rsi": short_rsi,
                     "volume_min": volume_min}
    if st.button("↩️ Varsayılan koşullara dön", use_container_width=True):
        for key in list(st.session_state):
            if key.startswith("condition_"):
                del st.session_state[key]
        st.rerun()
    st.divider()
    st.header("OKX veri ayarları")
    st.caption("Radarlar kendi seçtiğin aralıklarla yenilenir. Ticker ve mum önbelleği 15 sn, fonlama/OI 60 sn.")
    if st.button("🔄 OKX verilerini yenile"):
        st.cache_data.clear()
        st.rerun()

mobile_tab, account_tab, paper_tab, radar_tab, whale_tab, market_tab, bist_tab, futures, methodology = st.tabs(["📱 iPhone", "🔐 OKX Hesabım", "🧪 Paper Trading", "🟢🔴 OKX Perpetual Radar", "🐋 OKX Balina / Akıllı Para", "🌍 OKX Piyasa Yönü", "🇹🇷 BIST Radar", "⚠️ Vadeli risk ekranı", "ℹ️ Metodoloji"])

def analyze_okx_coin(item, okx_interval, stop_mult, target_mult, cfg):
    inst = item["Parite"]
    frame = okx_candles(inst, okx_interval)
    if frame.iloc[-1]["confirm"] != "0":
        raise ValueError("Açık mum alınamadı")
    frame = frame.copy()
    live_price = float(item["Fiyat ($)"])
    frame.loc[frame.index[-1], "close"] = live_price
    frame.loc[frame.index[-1], "high"] = max(float(frame.iloc[-1]["high"]), live_price)
    frame.loc[frame.index[-1], "low"] = min(float(frame.iloc[-1]["low"]), live_price)
    seconds = {"1h": 3600, "4h": 14400, "1d": 86400}[okx_interval]
    elapsed = (item["Ticker UTC"] - frame.iloc[-1]["date"]).total_seconds()
    fraction = min(1., max(.1, elapsed / seconds))
    frame.loc[frame.index[-1], "quote_volume"] /= fraction
    indicators = technical(frame, cfg)
    if indicators is None:
        raise ValueError(f"{inst} için teknik analiz hesaplanamadı: {len(frame)} mum var; en az 205 geçerli mum ve hesaplanabilir RSI/ADX/hacim gerekli. Yeni listelenen coinlerde sinyal üretilemez.")
    nk_frame = frame[frame["confirm"] == "1"].reset_index(drop=True)
    nk = nkral_signals(nk_frame, cfg["nk_sens"], cfg["nk_atr"])
    nk_long = bool(nk["NK AL"].iloc[-1]) if len(nk) else False
    nk_short = bool(nk["NK SAT"].iloc[-1]) if len(nk) else False
    indicators["NKRAL AL (son kapanış)"] = nk_long
    indicators["NKRAL SAT (son kapanış)"] = nk_short
    indicators["NKRAL stop ($)"] = float(nk["NK stop ($)"].iloc[-1]) if len(nk) else np.nan
    long_tests = indicators.pop("_long_tests")
    short_tests = indicators.pop("_short_tests")
    active = [k for k, yes in cfg["enabled"].items() if yes]
    fib = {}
    for tf in ("1h", "4h"):
        if cfg["fib"][tf]:
            fib_frame = frame if tf == okx_interval else okx_candles(inst, tf)
            if fib_frame.iloc[-1]["confirm"] != "0":
                raise ValueError("Fibonacci için açık mum yok")
            for side in ("LONG", "SHORT"):
                fib[(tf, side)] = fibonacci_check(fib_frame, live_price, side, cfg["fib_atr"])
        else:
            for side in ("LONG", "SHORT"):
                fib[(tf, side)] = (False, np.nan, np.nan)
    fib_active = [tf for tf in ("1h", "4h") if cfg["fib"][tf]]
    total = len(active) + len(fib_active)
    long_n = sum(long_tests[k] for k in active) + sum(fib[(tf, "LONG")][0] for tf in fib_active)
    short_n = sum(short_tests[k] for k in active) + sum(fib[(tf, "SHORT")][0] for tf in fib_active)
    indicators["LONG koşul"] = f"{long_n}/{total}"
    indicators["SHORT koşul"] = f"{short_n}/{total}"
    for tf in ("1h", "4h"):
        for side in ("LONG", "SHORT"):
            passed, level, distance = fib[(tf, side)]
            indicators[f"Fib {tf} {side}"] = passed if cfg["fib"][tf] else None
            indicators[f"Fib {tf} {side} seviye ($)"] = level
            indicators[f"Fib {tf} {side} uzaklık ATR"] = round(distance, 2) if np.isfinite(distance) else np.nan
    long_missing = [k for k in active if not long_tests[k]] + [
        f"Fib {tf}" for tf in fib_active if not fib[(tf, "LONG")][0]]
    short_missing = [k for k in active if not short_tests[k]] + [
        f"Fib {tf}" for tf in fib_active if not fib[(tf, "SHORT")][0]]
    if total == 0:
        stage, missing = "⚪ BEKLE", "Sol menüden en az bir koşul seç"
    elif long_n == total and short_n == total:
        stage, missing = "⚪ Çelişkili aday", "LONG ve SHORT aynı anda tüm seçili koşulları sağlıyor"
    elif long_n == total:
        stage, missing = "🟢 LONG", "—"
    elif short_n == total:
        stage, missing = "🔴 SHORT", "—"
    elif total >= 2 and long_n == total - 1 and short_n < total - 1:
        stage, missing = "🟡 LONG adayı", ", ".join(long_missing)
    elif total >= 2 and short_n == total - 1 and long_n < total - 1:
        stage, missing = "🟠 SHORT adayı", ", ".join(short_missing)
    else:
        stage = "⚪ BEKLE"
        missing = "LONG: " + (", ".join(long_missing) or "OK") + " | SHORT: " + (", ".join(short_missing) or "OK")
    if cfg["strategy"] == "NKRAL1":
        stage = "🟢 LONG" if nk_long and not nk_short else ("🔴 SHORT" if nk_short and not nk_long else "⚪ BEKLE")
        missing = "NKRAL1 son kapanmış mumda AL/SAT kesişimi yok" if stage == "⚪ BEKLE" else "—"
        indicators["LONG koşul"] = f"{int(nk_long)}/1"
        indicators["SHORT koşul"] = f"{int(nk_short)}/1"
    elif cfg["strategy"] == "Hibrit":
        radar_long, radar_short = stage == "🟢 LONG", stage == "🔴 SHORT"
        if radar_long and nk_long:
            stage, missing = "🟢 LONG", "—"
        elif radar_short and nk_short:
            stage, missing = "🔴 SHORT", "—"
        else:
            stage, missing = "⚪ BEKLE", "Radar teyidi ve NKRAL1 son kapanış kesişimi birlikte gerekli"
        indicators["LONG koşul"] += f" + NK {int(nk_long)}/1"
        indicators["SHORT koşul"] += f" + NK {int(nk_short)}/1"
    indicators["Strateji"] = cfg["strategy"]
    indicators["Fırsat durumu"] = stage
    indicators["Eksik koşul"] = missing
    indicators["Sinyal"] = stage if stage in ("🟢 LONG", "🔴 SHORT") else "⚪ BEKLE"
    try:
        derivative = okx_derivatives(inst)
    except Exception:
        derivative = {"Funding %": np.nan, "OI ($)": np.nan}
    levels = levels_and_risks(frame, {})
    scenario = atr_scenario(indicators, levels, stop_mult, target_mult)
    indicators["Sinyal mumu"] += " (açık / geçici)"
    return {**item.to_dict(), **indicators, **levels, **scenario, **derivative}


def live_radar():
        st.subheader(f"OKX USDT Perpetual — LONG / SHORT Radar · {strategy_mode}")
        st.caption(f"OKX canlı ticker ve açık perpetual mumundan geçici LONG/SHORT adayları. Sol menüdeki seçili koşullar uygulanır. RSI14: LONG {condition_cfg['long_rsi'][0]}–{condition_cfg['long_rsi'][1]}, SHORT {condition_cfg['short_rsi'][0]}–{condition_cfg['short_rsi'][1]}. Hesap bağlanmaz, emir gönderilmez.")
        p1, p2, p3 = st.columns(3)
        with p1:
            okx_interval = st.selectbox("Perpetual zaman dilimi", ["1h", "4h", "1d"], key="okx_interval")
        with p2:
            okx_limit = st.slider("Hacme göre analiz edilecek parite", 5, 60, 30, 5, key="okx_limit")
        with p3:
            okx_min_vol = st.number_input("En düşük yaklaşık 24s hacim ($ milyon)", min_value=0., value=1., step=1., key="okx_min_vol")
        st.info("Bu ekran CoinGecko market cap filtresinden bağımsızdır: OKX'teki USDT perpetual pariteleri yaklaşık 24 saatlik işlem hacmine göre tarar.")
        a1, a2 = st.columns(2)
        with a1:
            stop_mult = st.slider("ATR stop katsayısı", 0.5, 5.0, 1.5, 0.25)
        with a2:
            target_mult = st.slider("ATR hedef katsayısı", 0.5, 8.0, 3.0, 0.25)
        st.caption("ATR seviyeleri son kapanmış mum kapanışını referans alır; canlı giriş fiyatı veya emir değildir.")
        try:
            universe = okx_perpetual_universe()
            universe = universe[(universe["24s hacim yaklaşık ($)"] >= okx_min_vol * 1e6)
                                & (~universe["Sembol"].str.lower().isin(STABLE))]
            st.caption(f"OKX filtreyi geçen {len(universe)} USDT perpetual paritesi; ilk {min(okx_limit, len(universe))} analiz ediliyor.")
            okx_rows = []
            okx_fail = 0
            with st.spinner("OKX perpetual mumları ve vadeli göstergeleri alınıyor..."):
                for _, item in universe.head(okx_limit).iterrows():
                    try:
                        okx_rows.append(analyze_okx_coin(item, okx_interval, stop_mult, target_mult, condition_cfg))
                    except Exception:
                        okx_fail += 1
            if okx_fail:
                st.warning(f"{okx_fail} paritede yeterli mum veya veri alınamadı.")
            if not okx_rows:
                st.warning("Analiz edilebilir perpetual parite bulunamadı. Daha sonra tekrar dene.")
            else:
                radar = pd.DataFrame(okx_rows)
                order = {"🟢 LONG": 0, "🔴 SHORT": 1, "🟡 LONG adayı": 2, "🟠 SHORT adayı": 3, "⚪ Çelişkili aday": 4, "⚪ BEKLE": 5}
                radar["_order"] = radar["Fırsat durumu"].map(order).fillna(6)
                radar = radar.sort_values(["_order", "24s hacim yaklaşık ($)"],
                                          ascending=[True, False]).drop(columns="_order")
                counts = radar["Sinyal"].value_counts()
                r1, r2, r3 = st.columns(3)
                r1.metric("🟢 LONG", int(counts.get("🟢 LONG", 0)))
                r2.metric("🔴 SHORT", int(counts.get("🔴 SHORT", 0)))
                r3.metric("🟡🟠 Yaklaşan aday", int(radar["Fırsat durumu"].isin(["🟡 LONG adayı", "🟠 SHORT adayı"]).sum()))
                show = ["Parite", "Strateji", "NKRAL AL (son kapanış)", "NKRAL SAT (son kapanış)", "NKRAL stop ($)", "Fırsat durumu", "LONG koşul", "SHORT koşul", "Eksik koşul", "Fib 1h LONG", "Fib 4h LONG", "Fib 1h SHORT", "Fib 4h SHORT", "Sinyal", "Ticker UTC", "Sinyal mumu", "Fiyat ($)",
                        "Referans giriş ($)", "Stop ($)", "Hedef ($)",
                        "Stop uzaklık %", "Hedef uzaklık %", "Risk/Ödül",
                        "24s hacim yaklaşık ($)", "RSI", "ADX", "Hacim katı",
                        "Funding %", "OI ($)", "20 mum destek ($)",
                        "20 mum direnç ($)", "ATR14 %", "Risk notları"]
                def okx_signal_color(row):
                    signal = row["Fırsat durumu"]
                    bg = ("background-color: #143d2b; color: #e6fff0" if signal == "🟢 LONG"
                          else "background-color: #52232b; color: #fff0f0" if signal == "🔴 SHORT"
                          else "background-color: #54431b; color: #fff4d0" if signal == "🟡 LONG adayı"
                          else "background-color: #5a351b; color: #fff0d7" if signal == "🟠 SHORT adayı"
                          else "")
                    return [bg] * len(row)
                st.dataframe(radar[show].style.apply(okx_signal_color, axis=1),
                             hide_index=True, use_container_width=True)
                st.caption("Koşullar sol menüden seçilir; devre dışı bırakılan Fibonacci hücreleri boş görünür. Funding % mevcut fonlama oranıdır; tek başına LONG/SHORT koşuluna katılmaz. OI ($) mevcut açık pozisyon anlık görüntüsüdür; OI değişimi değildir. OKX hacmi yaklaşık USD cinsindedir.")
                st.download_button("📥 OKX perpetual radar CSV",
                                   radar.to_csv(index=False).encode("utf-8-sig"),
                                   "burak_okx_perpetual_radar.csv", "text/csv")
                st.warning("Canlı aday sinyal: Açık mum kapanmadan LONG/SHORT değişebilir. Ticker UTC zamanını kontrol et; eski fiyatı işlem referansı alma.")
        except Exception as exc:
            st.error(f"OKX radar yüklenemedi: {type(exc).__name__}: {exc}")

        st.divider()
        st.subheader("🔎 Kendi coinini analiz et")
        st.caption("Ana tarama listesinden bağımsızdır. OKX USDT perpetual paritesini gir; sol menüde seçtiğin teknik ve Fibonacci koşullarıyla değerlendirilir. Ana taramanın hacim ve ilk-N sınırına tabi değildir.")
        custom_text = st.text_input("Coin sembolü", placeholder="Örn. ETH, SOL veya ETH-USDT-SWAP", key="custom_coin")
        if custom_text.strip():
            raw = custom_text.strip().upper().replace("/", "-").replace(" ", "")
            symbol = raw.removesuffix("-USDT-SWAP").removesuffix("-USDT")
            inst_id = symbol + "-USDT-SWAP"
            try:
                full_universe = okx_perpetual_universe()
                matched = full_universe[full_universe["Parite"] == inst_id]
                if matched.empty:
                    st.warning(f"{inst_id} OKX USDT perpetual listesinde bulunamadı. Coin sembolünü kontrol et.")
                else:
                    with st.spinner(f"{inst_id} canlı verileri ve Fibonacci teyitleri hesaplanıyor..."):
                        custom = analyze_okx_coin(matched.iloc[0], okx_interval, stop_mult, target_mult, condition_cfg)
                    st.metric("Fırsat durumu", custom["Fırsat durumu"])
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Canlı fiyat ($)", f'{custom["Fiyat ($)"]:,.6g}')
                    c2.metric("LONG koşul", custom["LONG koşul"])
                    c3.metric("SHORT koşul", custom["SHORT koşul"])
                    st.write("**Eksik koşul:**", custom["Eksik koşul"])
                    fields = ["Parite", "Strateji", "NKRAL AL (son kapanış)", "NKRAL SAT (son kapanış)", "NKRAL stop ($)", "Ticker UTC", "Sinyal mumu", "Fırsat durumu", "Eksik koşul",
                              "Fib 1h LONG", "Fib 4h LONG", "Fib 1h SHORT", "Fib 4h SHORT",
                              "Fib 1h LONG seviye ($)", "Fib 4h LONG seviye ($)",
                              "Fib 1h SHORT seviye ($)", "Fib 4h SHORT seviye ($)",
                              "RSI", "ADX", "Hacim katı", "ATR14 %",
                              "Referans giriş ($)", "Stop ($)", "Hedef ($)", "Risk/Ödül",
                              "Funding %", "OI ($)"]
                    st.dataframe(pd.DataFrame([custom])[fields], hide_index=True, use_container_width=True)
                    st.caption("Canlı açık mumdaki aday sinyaller değişebilir. ATR stop/hedef örnektir; emir gönderilmez.")
            except Exception as exc:
                st.error(f"{inst_id} analiz edilemedi: {type(exc).__name__}: {exc}")


with radar_tab:
    refresh_minutes = st.selectbox(
        "⏱️ Otomatik yenileme aralığı",
        options=[1, 2, 3, 5, 10, 15, 30],
        index=3,
        format_func=lambda n: f"{n} dakika",
        key="radar_refresh_minutes",
        help="Ana radar ve kendi coin analiz alanı bu aralıkla yeniden hesaplanır. Değişiklik seçildiğinde uygulanır."
    )
    st.caption(f"Otomatik yenileme: {refresh_minutes} dakikada bir. Sayfa açık kaldığı sürece çalışır.")
    st.fragment(run_every=f"{refresh_minutes * 60}s")(live_radar)()
    st.divider()
    compare_strategies_ui(condition_cfg)



with mobile_tab:
    st.subheader("📱 BURAK RADAR | iPhone")
    st.caption("iPhone 14 Pro için sade görünüm · OKX USDT perpetual · Emir göndermez")
    st.info("Strateji ve NKRAL ayarları sol üstteki ☰ menüsündedir. Bu ekran mevcut analiz motorunu kullanır.")
    m_tf = st.segmented_control("Zaman dilimi", ["1h", "4h"], default="1h", key="mobile_tf")
    m_coins = st.multiselect("Takip listem", ["BTC", "ETH", "SOL", "HYPE", "XRP", "DOGE", "BNB", "SUI", "WLD", "AVAX", "LINK", "ADA"],
                             default=["BTC", "ETH", "SOL"], max_selections=6, key="mobile_watch")
    m_extra = st.text_input("Başka coin ekle (sembol)", placeholder="Örn. PEPE", key="mobile_extra").strip().upper()
    m_symbols = list(dict.fromkeys(m_coins + ([m_extra.removesuffix("-USDT-SWAP").removesuffix("-USDT")] if m_extra else [])))
    st.caption(f"Seçili strateji: **{strategy_mode}** · NKRAL sinyalleri son kapanmış mumdan, mevcut radar açık mumdan hesaplanır.")
    m_refresh = st.selectbox("Otomatik yenileme", [1, 2, 3, 5, 10, 15], index=3,
                             format_func=lambda x: f"{x} dakika", key="mobile_refresh")
    if st.button("🔄 Şimdi yenile", use_container_width=True, key="mobile_reload"):
        st.cache_data.clear()
        st.rerun()

    @st.fragment(run_every=f"{m_refresh * 60}s")
    def mobile_watchlist():
        if not m_symbols:
            st.warning("Takip listene en az bir coin ekle.")
            return
        try:
            universe = okx_perpetual_universe()
        except Exception as exc:
            st.error(f"OKX listesi alınamadı: {exc}")
            return
        for symbol in m_symbols:
            inst = symbol + "-USDT-SWAP"
            match = universe[universe["Parite"] == inst]
            if match.empty:
                st.warning(f"{symbol}: OKX USDT perpetual bulunamadı.")
                continue
            try:
                result = analyze_okx_coin(match.iloc[0], m_tf, 1.5, 3.0, condition_cfg)
            except Exception as exc:
                st.warning(f"{symbol}: veri/analiz hatası ({type(exc).__name__}: {exc})")
                continue
            with st.container(border=True):
                st.markdown(f"### {symbol} · {result['Fırsat durumu']}")
                st.caption(f"{result['Strateji']} · {m_tf} · {result['Sinyal mumu']}")
                x, y = st.columns(2)
                x.metric("Fiyat ($)", f"{result['Fiyat ($)']:,.6g}")
                y.metric("Risk/Ödül", f"{result['Risk/Ödül']:.2f}" if np.isfinite(result['Risk/Ödül']) else "—")
                x, y = st.columns(2)
                x.metric("LONG koşul", result["LONG koşul"])
                y.metric("SHORT koşul", result["SHORT koşul"])
                with st.expander("📊 Teknik detaylar ve seviyeler"):
                    st.write("**Eksik koşullar:**", result["Eksik koşul"])
                    st.write("**NKRAL AL / SAT:**", result["NKRAL AL (son kapanış)"], "/", result["NKRAL SAT (son kapanış)"])
                    st.write("**Fib 1h LONG / SHORT:**", result["Fib 1h LONG"], "/", result["Fib 1h SHORT"])
                    st.write("**Fib 4h LONG / SHORT:**", result["Fib 4h LONG"], "/", result["Fib 4h SHORT"])
                    for label in ("Referans giriş ($)", "Stop ($)", "Hedef ($)", "NKRAL stop ($)", "RSI", "ADX", "Funding %", "OI ($)"):
                        value = result.get(label)
                        st.write(f"**{label}:**", f"{value:,.6g}" if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value) else "—")
        st.caption("⚠️ Açık mumdaki radar sinyali değişebilir; NKRAL kapanmış mum kesişimi yeni mum gelene kadar korunur.")

    mobile_watchlist()
    st.caption("iPhone Safari: Paylaş → Ana Ekrana Ekle. Bu bir web uygulaması kısayoludur; App Store uygulaması veya çevrimdışı PWA değildir.")


with account_tab:
    st.subheader("🔐 OKX Hesabım — salt okunur V6.1")
    st.caption("Bakiye, açık pozisyon ve bekleyen emir görüntüleme. Bu uygulamada emir açma, kapatma veya para çekme kodu yoktur.")
    credentials = okx_account_secrets()
    configured = all(credentials.get(k) for k in ("api_key", "secret_key", "passphrase", "dashboard_password"))
    if not configured:
        st.warning("Hesap bağlantısı henüz yapılandırılmadı. API bilgilerini buraya veya GitHub'a yazma.")
        st.markdown("**Kurulum:** Streamlit Community Cloud → uygulaman → Settings → Secrets alanına aşağıdaki şablonu kendi bilgilerinle gir:")
        st.code('[okx]\\napi_key = "OKX_API_KEY"\\nsecret_key = "OKX_SECRET_KEY"\\npassphrase = "OKX_PASSPHRASE"\\ndashboard_password = "UZUN_BENZERSIZ_PANEL_SIFRESI"\\ndemo = false', language="toml")
        st.info("OKX API anahtarını yalnızca Read (Okuma) yetkisiyle oluştur. Trade ve Withdraw izinlerini açma. Panel şifresini API passphrase'inden farklı belirle.")
        st.caption("Bu uygulamanın genel radar sekmeleri herkese açık kalır. Hesap sekmesi ayrı panel şifresi ile korunur. Daha güçlü erişim kontrolü için tüm uygulamayı özel erişime al.")
    else:
        if not st.session_state.get("okx_account_unlocked", False):
            with st.form("okx_account_login"):
                supplied_password = st.text_input("Hesap paneli şifresi", type="password")
                login = st.form_submit_button("🔓 Hesabımı göster", use_container_width=True)
            if login:
                if hmac.compare_digest(supplied_password.encode("utf-8"), credentials["dashboard_password"].encode("utf-8")):
                    st.session_state["okx_account_unlocked"] = True
                    st.rerun()
                else:
                    st.error("Panel şifresi hatalı.")
        else:
            left, right = st.columns(2)
            if left.button("🔄 Hesabı yenile", use_container_width=True, key="okx_account_refresh"):
                st.rerun()
            if right.button("🔒 Paneli kilitle", use_container_width=True, key="okx_account_lock"):
                st.session_state["okx_account_unlocked"] = False
                st.rerun()
            st.caption("Ortam: " + ("OKX Demo" if credentials["demo"] else "OKX Canlı") + " · Her yenilemede yalnızca GET istekleri gönderilir.")
            try:
                balances = okx_private_get("/api/v5/account/balance", credentials)
                positions = okx_private_get("/api/v5/account/positions?instType=SWAP", credentials)
                pending = okx_private_get("/api/v5/trade/orders-pending?instType=SWAP", credentials)
                account = balances[0] if balances else {}
                details = account.get("details", [])
                usdt = next((item for item in details if item.get("ccy") == "USDT"), {})
                def fmt_amount(value, decimals=2):
                    try:
                        return f"{float(value):,.{decimals}f}"
                    except (TypeError, ValueError):
                        return "—"
                a, b = st.columns(2)
                a.metric("Toplam hesap özkaynağı (USDT karşılığı)", fmt_amount(account.get("totalEq")))
                b.metric("Kullanılabilir USDT", fmt_amount(usdt.get("availBal") or usdt.get("availEq")))
                a, b = st.columns(2)
                a.metric("USDT özkaynak", fmt_amount(usdt.get("eq")))
                b.metric("USDT gerçekleşmemiş P&L", fmt_amount(usdt.get("upl")))
                st.caption("500 USDT bot bütçesi henüz emir yetkisine bağlı değil; burada görünen toplam hesap özkaynağı farklı para birimlerini de içerebilir.")
                st.markdown("#### Açık USDT perpetual pozisyonları")
                active = [p for p in positions if float(p.get("pos") or 0) != 0]
                if active:
                    fields = {
                        "instId": "Parite", "posSide": "Yön", "pos": "Kontrat",
                        "avgPx": "Giriş ($)", "markPx": "Mark fiyatı ($)",
                        "lever": "Kaldıraç", "mgnMode": "Marjin", "upl": "P&L (USDT)",
                        "liqPx": "Likidasyon ($)"
                    }
                    st.dataframe(pd.DataFrame(active).reindex(columns=fields).rename(columns=fields),
                                 use_container_width=True, hide_index=True)
                else:
                    st.success("Açık perpetual pozisyon bulunamadı.")
                st.markdown("#### Bekleyen USDT perpetual emirleri")
                if pending:
                    fields = {
                        "instId": "Parite", "side": "Emir", "posSide": "Pozisyon yönü",
                        "ordType": "Tip", "px": "Fiyat ($)", "sz": "Kontrat",
                        "state": "Durum", "clOrdId": "İstemci emir ID"
                    }
                    st.dataframe(pd.DataFrame(pending).reindex(columns=fields).rename(columns=fields),
                                 use_container_width=True, hide_index=True)
                else:
                    st.info("Bekleyen standart perpetual emir bulunamadı.")
                st.caption("Koşullu stop/TP emirleri bu ilk sürümün bekleyen emir tablosuna dahil değildir. OKX API verileri anlık değişebilir.")
            except RuntimeError as exc:
                st.error(str(exc))


# V6.2: paper ledger is scoped to the current authenticated Streamlit session.
# It deliberately has NO API trade permission, durable storage or background worker.
def paper_default_state():
    return {"cash": 500., "positions": [], "trades": [], "seen": [], "strategy": None,
            "running": False, "day": "", "day_start": 500., "last_scan": ""}


def paper_db_request(method, payload=None):
    """Supabase REST calls only on the Streamlit server; never expose secret keys."""
    cfg = st.secrets.get("supabase", {})
    url = str(cfg.get("url", "")).strip().rstrip("/")
    key = str(cfg.get("secret_key", "")).strip()
    if not url.startswith("https://") or not key:
        raise RuntimeError("Streamlit Secrets [supabase] url / secret_key eksik.")
    headers = {"apikey": key, "Authorization": "Bearer " + key,
               "Content-Type": "application/json"}
    if method == "POST":
        headers["Prefer"] = "resolution=merge-duplicates,return=minimal"
    response = requests.request(
        method, url + "/rest/v1/paper_trading_state", headers=headers,
        params={"account_id": "eq.burak_paper_main"} if method == "GET" else
               {"on_conflict": "account_id"}, json=payload, timeout=15
    )
    response.raise_for_status()
    return response.json() if method == "GET" else None


def paper_db_load():
    rows = paper_db_request("GET")
    if not rows:
        return None
    state = rows[0].get("state")
    if not isinstance(state, dict) or not all(k in state for k in ("cash", "positions", "trades")):
        raise RuntimeError("Kalıcı sanal hesap kaydı beklenen biçimde değil.")
    state["running"] = False  # Never silently resume automation on a new session.
    state.setdefault("strategy", None)
    state.setdefault("seen", [])
    state.setdefault("day", "")
    state.setdefault("day_start", 500.)
    state.setdefault("last_scan", "")
    for p in state["positions"]:
        p.setdefault("strategy", "Hibrit (V6.2)")
    return state


def paper_db_save(state):
    import json
    safe = json.loads(json.dumps(state, allow_nan=False))
    paper_db_request("POST", [{"account_id": "burak_paper_main", "state": safe}])


def paper_state():
    if "paper_v62" not in st.session_state:
        st.session_state["paper_v62"] = paper_default_state()
    state = st.session_state["paper_v62"]
    state.setdefault("strategy", None)
    for position in state.get("positions", []):
        position.setdefault("strategy", "Hibrit (V6.2)")
    return state


def paper_equity(state, quotes):
    return state["cash"] + sum(
        p["margin"] + p["direction"] * p["notional"] *
        (quotes.get(p["inst"], p["entry"]) / p["entry"] - 1)
        for p in state["positions"]
    )


def paper_close_position(state, p, price, reason):
    """Settle a single virtual position at an observed OKX ticker price."""
    now = datetime.now(timezone.utc)
    exit_price = price
    exit_fee = p["notional"] * (price / p["entry"]) * .0005
    pnl = p["direction"] * p["notional"] * (price / p["entry"] - 1) - p["entry_fee"] - exit_fee
    state["cash"] += p["margin"] + p["direction"] * p["notional"] * (price / p["entry"] - 1) - exit_fee
    state["trades"].append({
        "Parite": p["inst"], "Yön": "LONG" if p["direction"] == 1 else "SHORT",
        "Strateji": p["strategy"], "Risk/Ödül": p.get("reward_ratio", 2), "Giriş UTC": p["time"], "Çıkış UTC": now.isoformat(timespec="seconds"),
        "Giriş": p["entry"], "Çıkış": exit_price, "Net P&L (USDT)": round(pnl, 4),
        "Çıkış nedeni": reason
    })
    state["positions"].remove(p)
    return pnl


def paper_check_exits(state, quotes):
    """Check observed prices against virtual stops/targets; never send orders."""
    now = datetime.now(timezone.utc)
    closed_count = 0
    for p in list(state["positions"]):
        price = quotes.get(p["inst"])
        if price is None or price <= 0:
            continue
        stop_hit = price <= p["stop"] if p["direction"] == 1 else price >= p["stop"]
        target_hit = price >= p["target"] if p["direction"] == 1 else price <= p["target"]
        if not (stop_hit or target_hit):
            continue
        paper_close_position(state, p, price, "STOP" if stop_hit else "HEDEF")
        closed_count += 1
    return closed_count


def paper_scan(state, cfg, universe, max_coins, reward_ratio, allow_manual=False):
    """One on-demand simulation tick. Closed-bar same-timeframe technical hybrid;
    with cross-timeframe Fibonacci. No exchange orders."""
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    if state["day"] != today:
        state["day"], state["day_start"] = today, state["cash"] + sum(p["margin"] for p in state["positions"])
    quotes = {row["Parite"]: float(row["Fiyat ($)"]) for _, row in universe.iterrows()}
    paper_check_exits(state, quotes)
    equity = paper_equity(state, quotes)
    if state["day_start"] - equity >= 15:
        state["running"] = False
        return "Günlük 15 USDT zarar eşiği görüldü; yeni sanal işlemler durduruldu."
    if not state["running"] and not allow_manual:
        return "Otomatik simülasyon duraklatılmış."
    if state["strategy"] != cfg["strategy"]:
        return "Strateji değişti. Yeni strateji için sanal oturumu sıfırlayıp yeniden başlat."
    scanned, errors, opened = 0, 0, 0
    for _, item in universe.head(max_coins).iterrows():
        if len(state["positions"]) >= 5 or state["cash"] < 10:
            break
        inst = item["Parite"]
        if any(p["inst"] == inst for p in state["positions"]):
            continue
        try:
            frame = okx_candles(inst, "1h")
            closed = frame[frame["confirm"] == "1"].reset_index(drop=True)
            if len(closed) < 210:
                continue
            bar_id = str(closed.iloc[-1]["date"])
            signal_key = inst + "|" + bar_id
            if signal_key in state["seen"]:
                continue
            mode = cfg["strategy"]
            nk = nkral_signals(closed, cfg["nk_sens"], cfg["nk_atr"])
            nk_direction = (1 if bool(nk["NK AL"].iloc[-1]) else
                            -1 if bool(nk["NK SAT"].iloc[-1]) else 0)
            radar_direction = 0
            if mode != "NKRAL1":
                indicators = technical(closed, cfg)
                if indicators is None:
                    continue
                active = [name for name, enabled in cfg["enabled"].items() if enabled]
                fib_active = [tf for tf in ("1h", "4h") if cfg["fib"][tf]]
                count = len(active) + len(fib_active)
                if count:
                    long_tests = indicators["_long_tests"]
                    short_tests = indicators["_short_tests"]
                    long_ok = all(long_tests[name] for name in active)
                    short_ok = all(short_tests[name] for name in active)
                    signal_price = float(closed["close"].iloc[-1])
                    for tf in fib_active:
                        fib_frame = frame if tf == "1h" else okx_candles(inst, tf)
                        long_ok = long_ok and fibonacci_check(fib_frame, signal_price, "LONG", cfg["fib_atr"])[0]
                        short_ok = short_ok and fibonacci_check(fib_frame, signal_price, "SHORT", cfg["fib_atr"])[0]
                    radar_direction = 1 if long_ok and not short_ok else (-1 if short_ok and not long_ok else 0)
            direction = (nk_direction if mode == "NKRAL1" else
                         radar_direction if mode == "Mevcut Radar" else
                         radar_direction if radar_direction == nk_direction else 0)
            state["seen"].append(signal_key)
            if direction == 0:
                continue
            # ATR from last fully closed 1h candle; paper entry at observed ticker.
            c = closed["close"].astype(float)
            tr = pd.concat([
                closed["high"] - closed["low"],
                (closed["high"] - c.shift()).abs(),
                (closed["low"] - c.shift()).abs()
            ], axis=1).max(axis=1)
            atr = float(tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1])
            price = float(item["Fiyat ($)"])
            if not np.isfinite(atr) or atr <= 0 or price <= 0:
                continue
            stop = price - direction * 1.5 * atr
            target = price + direction * (1.5 * reward_ratio) * atr
            if stop <= 0 or target <= 0:
                continue
            stop_fraction = 1.5 * atr / price
            # Gross stop risk <=5 USDT, per-position margin <=16% equity,
            # total reserved margin <=80% equity.
            current_equity = paper_equity(state, quotes)
            reserved_margin = sum(p["margin"] for p in state["positions"])
            available_margin = max(0., current_equity * .80 - reserved_margin)
            notional = min(5. / stop_fraction, current_equity * 5 * .16,
                           available_margin * 5, state["cash"] * 5 * .95)
            margin = notional / 5
            entry_fee = notional * .0005
            if margin + entry_fee > state["cash"] or notional < 10:
                continue
            state["cash"] -= margin + entry_fee
            state["positions"].append({
                "inst": inst, "strategy": cfg["strategy"], "direction": direction, "entry": price,
                "stop": stop, "target": target, "reward_ratio": reward_ratio, "notional": notional,
                "margin": margin, "entry_fee": entry_fee,
                "time": now.isoformat(timespec="seconds"), "bar": bar_id
            })
            opened += 1
            scanned += 1
        except (ValueError, KeyError, TypeError, IndexError, requests.RequestException):
            errors += 1
            continue
    state["seen"] = state["seen"][-1500:]
    state["last_scan"] = now.isoformat(timespec="seconds")
    return f"Tarama tamamlandı: {max_coins} hacimli pariteye kadar kontrol; {opened} yeni sanal pozisyon; {errors} veri/analiz hatası."


with paper_tab:
    st.subheader("🧪 Paper Trading V6.9 — 500 USDT / 5x / seçili strateji")
    st.warning("Bu bir OTURUM İÇİ simülasyondur: tarayıcı/oturum kapalıyken veya uygulama uyuduğunda otomatik tarama/stop çalışmaz; uygulama yeniden başlarsa kayıtlar silinebilir. 7/24 bot veya güvenilir geçmiş performans testi değildir.")
    st.caption("Gerçek OKX hesabına emir gönderilmez. İşlemler sanal 500 USDT ile, maksimum 5 isolated pozisyon ve işlem başına en fazla 5 USDT brüt planlanan stop riskiyle modellenir. Pozisyon başına teminat en fazla özkaynağın %16’sı, toplam ayrılan teminat en fazla %80’idir.")
    paper_creds = okx_account_secrets()
    if not paper_creds.get("dashboard_password") or not st.session_state.get("okx_account_unlocked", False):
        st.info("Bu sekme için önce 🔐 OKX Hesabım bölümünde panel şifrenle giriş yap.")
    else:
        if not st.session_state.get("paper_db_loaded", False):
            try:
                saved = paper_db_load()
                if saved is None:
                    paper_db_save(paper_state())
                else:
                    st.session_state["paper_v62"] = saved
                st.session_state["paper_db_loaded"] = True
            except (RuntimeError, ValueError, requests.RequestException) as exc:
                st.error("Supabase bağlantısı kurulamadı; veri kaybını önlemek için sanal işlem durduruldu: " + str(exc))
                st.stop()
        ps = paper_state()
        st.caption("☁️ Supabase kalıcı hafıza etkin · Yeniden açılan oturumda otomatik tarama duraklatılır.")
        st.write("Sol menüde seçilen strateji:", strategy_mode)
        if ps["strategy"] is not None and ps["strategy"] != strategy_mode:
            st.warning("Sanal oturum " + ps["strategy"] + " ile açıldı. Yeni strateji için oturumu sıfırla.")
        paper_mode = st.radio(
            "🔎 Sanal tarama modu",
            ["Manuel tarama", "Otomatik tarama (5 dakikada bir)"],
            horizontal=True, key="paper_scan_mode",
            help="Manuel modda yeni işlem yalnızca tarama butonuyla aranır. Otomatik modda Başlat butonu gerekir."
        )
        if paper_mode == "Manuel tarama":
            ps["running"] = False
            st.info("Manuel mod: yalnızca 🔎 Sanal tarama butonuna bastığında sinyal, stop ve hedef kontrolü yapılır.")
        else:
            if st.button("⏸️ Otomatik taramayı duraklat" if ps["running"]
                         else "▶️ Otomatik taramayı başlat (5 dk)",
                         use_container_width=True, key="paper_toggle"):
                if ps["strategy"] is None:
                    ps["strategy"] = strategy_mode
                if ps["strategy"] != strategy_mode:
                    st.error("Önce sanal oturumu sıfırla veya önceki stratejiyi seç.")
                else:
                    ps["running"] = not ps["running"]
                    if ps["running"]:
                        ps["last_auto_scan_ts"] = 0.
                    st.rerun()
            st.write("Otomatik durum:", "🟢 Aktif · 5 dakikada bir" if ps["running"]
                     else "⏸️ Duraklatıldı")
        with st.expander("🗑️ Sanal oturumu sıfırla / strateji değiştir"):
            st.caption("Sanal bakiye, açık pozisyonlar ve işlem geçmişi silinir. Önce CSV indirebilirsin.")
            if st.button("Sanal oturumu sıfırla", key="paper_reset"):
                try:
                    fresh_state = paper_default_state()
                    paper_db_save(fresh_state)
                    st.session_state["paper_v62"] = fresh_state
                    st.rerun()
                except (RuntimeError, ValueError, requests.RequestException) as exc:
                    st.error("Sıfırlama kaydedilemedi; eski kayıt korundu: " + str(exc))
        reward_ratio = st.selectbox("Risk / Ödül oranı", [2, 3, 4, 5],
                                    format_func=lambda x: f"1:{x}", index=0,
                                    key="paper_reward_ratio",
                                    help="Stop 1,5 ATR sabit; hedef oran × 1,5 ATR. Açık işlemlerin hedefi değişmez.")
        st.caption(f"Yeni sanal işlemler: stop 1,5 ATR · hedef {1.5 * reward_ratio:g} ATR · en fazla 5 açık pozisyon.")
        scan_count = st.select_slider("Her turda hacme göre taranacak coin", [10, 20, 30, 40, 50, 60], value=30,
                                      help="V6.2 prototipi bütün OKX coinlerini taramaz; ilk 60'a kadar seçilebilir.")
        if st.button("🔎 Sanal tarama + stop/hedef kontrolü", type="primary", use_container_width=True):
            try:
                with st.spinner("OKX kapanmış mumları kontrol ediliyor..."):
                    pu = okx_perpetual_universe()
                    if ps["strategy"] is None:
                        ps["strategy"] = strategy_mode
                    message = paper_scan(ps, condition_cfg, pu, scan_count, reward_ratio,
                                         allow_manual=(paper_mode == "Manuel tarama"))
                paper_db_save(ps)
                st.info(message)
            except (ValueError, requests.RequestException, KeyError, RuntimeError) as exc:
                st.error(f"Tarama başarısız: {type(exc).__name__}")
        st.caption("Mevcut Radar: seçili teknik + 1H/4H Fibonacci teyitleri. NKRAL1: kapanmış mum kesişimi. Hibrit: aynı kapanmış 1H mumda NKRAL1 + teknik/Fibonacci teyidi. Canlı radar açık mum kullandığından anlık sinyaller farklı olabilir.")
        @st.fragment(run_every="10s")
        def paper_live_monitor():
            """UI-bound refresh only; stops when the client/session stops."""
            try:
                live_universe = okx_perpetual_universe()
                current_quotes = {r["Parite"]: float(r["Fiyat ($)"])
                                  for _, r in live_universe.iterrows()}
            except (ValueError, requests.RequestException, KeyError, RuntimeError) as exc:
                current_quotes = {}
                st.warning("OKX fiyatları güncellenemedi; son bilinen fiyatlar kullanılıyor.")
            if paper_mode == "Otomatik tarama (5 dakikada bir)" and ps["running"] and current_quotes:
                closed_count = paper_check_exits(ps, current_quotes)
                if closed_count:
                    st.success(f"{closed_count} sanal pozisyon stop/hedef nedeniyle kapatıldı.")
                # Scan on activation, then at 300-second intervals while the UI session is alive.
                now_ts = datetime.now(timezone.utc).timestamp()
                elapsed = now_ts - float(ps.get("last_auto_scan_ts", 0))
                if elapsed >= 300:
                    try:
                        result = paper_scan(ps, condition_cfg, live_universe,
                                            scan_count, reward_ratio)
                        ps["last_auto_scan_ts"] = now_ts
                        st.info("Otomatik 5 dk tarama: " + result)
                    except (ValueError, requests.RequestException, KeyError, RuntimeError) as exc:
                        st.warning(f"Otomatik tarama tamamlanamadı: {type(exc).__name__}. Sonraki yenilemede tekrar denenecek.")
                if ps["running"] and ps.get("last_auto_scan_ts", 0):
                    remaining = max(0, 300 - (datetime.now(timezone.utc).timestamp() -
                                               ps["last_auto_scan_ts"]))
                    st.caption(f"Sonraki otomatik taramaya yaklaşık {remaining:.0f} saniye.")
            import json
            try:
                fingerprint = json.dumps(ps, sort_keys=True, allow_nan=False)
                if st.session_state.get("paper_db_last_saved") != fingerprint:
                    paper_db_save(ps)
                    st.session_state["paper_db_last_saved"] = fingerprint
            except (RuntimeError, ValueError, requests.RequestException) as exc:
                ps["running"] = False
                st.error("☁️ Kayıt başarısız; otomatik tarama durduruldu: " + str(exc))
            st.caption("Son ekran kontrolü: " + datetime.now(timezone.utc).strftime("%H:%M:%S UTC") +
                       " · OKX ticker önbelleği 15 sn · ekran kontrolü yaklaşık 10 sn.")
            eq = paper_equity(ps, current_quotes)
            a, b, c = st.columns(3)
            a.metric("Sanal özkaynak", f"{eq:,.2f} USDT")
            b.metric("Sanal nakit", f"{ps['cash']:,.2f} USDT")
            c.metric("Açık pozisyon", f"{len(ps['positions'])}/5")
            if ps["positions"]:
                st.markdown("**📂 Açık sanal pozisyonlar**")
                st.caption("Her kart yalnızca kendi pozisyonunu kapatır. Kapatma tam miktar ve güncel OKX ticker fiyatı üzerinden sanaldır.")
                for p in list(ps["positions"]):
                    mark = current_quotes.get(p["inst"], p["entry"])
                    pnl = p["direction"] * p["notional"] * (mark / p["entry"] - 1) - p["entry_fee"]
                    side = "🟢 LONG" if p["direction"] == 1 else "🔴 SHORT"
                    with st.container(border=True):
                        st.markdown(f"**{p['inst']} · {side}**")
                        x1, x2, x3 = st.columns(3)
                        x1.metric("Giriş", f"{p['entry']:,.6g} USDT")
                        x2.metric("Gözlenen fiyat", f"{mark:,.6g} USDT")
                        x3.metric("Açık net P&L (tahmini)", f"{pnl:+,.2f} USDT")
                        y1, y2, y3 = st.columns(3)
                        y1.caption(f"Stop: {p['stop']:,.6g}")
                        y2.caption(f"Hedef: {p['target']:,.6g}")
                        y3.caption(f"Teminat: {p['margin']:,.2f} USDT · R/Ö 1:{p.get('reward_ratio', 2)}")
                        if st.button(f"✋ Yalnızca {p['inst']} pozisyonunu kapat",
                                     key=f"paper_close_{p['inst']}_{p['time']}",
                                     use_container_width=True):
                            try:
                                okx_perpetual_universe.clear()
                                fresh = okx_perpetual_universe()
                                match = fresh.loc[fresh["Parite"] == p["inst"], "Fiyat ($)"]
                                if match.empty or float(match.iloc[0]) <= 0:
                                    st.error("Güncel OKX fiyatı alınamadı; pozisyon açık bırakıldı.")
                                elif p not in ps["positions"]:
                                    st.warning("Bu pozisyon zaten kapanmış.")
                                else:
                                    exit_price = float(match.iloc[0])
                                    net = paper_close_position(ps, p, exit_price, "MANUEL")
                                    st.toast(f"{p['inst']} sanal kapatıldı · net P&L: {net:+.2f} USDT")
                                    paper_db_save(ps)
                                    st.rerun()
                            except (ValueError, KeyError, TypeError, requests.RequestException, RuntimeError):
                                st.error("OKX fiyatı alınamadı; pozisyon açık bırakıldı.")
            else:
                st.info("Açık sanal pozisyon yok.")
            if ps["trades"]:
                history = pd.DataFrame(ps["trades"])
                st.dataframe(history.iloc[::-1], use_container_width=True, hide_index=True)
                st.download_button("📥 İşlem geçmişini CSV indir", history.to_csv(index=False).encode("utf-8-sig"),
                                   "burak_paper_trades.csv", "text/csv")
        paper_live_monitor()
        st.caption("Her giriş ve çıkışta varsayımsal %0,05 komisyon kullanılır; fonlama, spread, kayma ve likidasyon modellenmez. Otomatik mod aktifken stop/hedef açık oturumda yaklaşık 10 saniyede bir kontrol edilir (OKX ticker önbelleği 15 saniye), yeni sinyal 5 dakikada bir taranır. Manuel modda kontrol yalnızca tarama butonuyla yapılır. Kontroller arasında stop geçişleri kaçabilir.")


# The whale tab uses only public OKX market aggregates, never private wallets.
@st.cache_data(ttl=60, show_spinner=False)
def okx_recent_trades(inst_id):
    return okx_public("/api/v5/market/trades", {"instId": inst_id, "limit": "500"})


def whale_market_reading(item, instrument, min_trade_usd):
    inst_id = item["Parite"]
    price = float(item["Fiyat ($)"])
    trades = okx_recent_trades(inst_id)
    contract_value = float(instrument.get("ctVal") or 0)
    contract_mult = float(instrument.get("ctMult") or 1)
    contract_ccy = instrument.get("ctValCcy", "")
    base = item["Sembol"]
    if contract_value <= 0 or contract_mult <= 0 or contract_ccy != base:
        raise ValueError("Sözleşme USD nominali güvenle hesaplanamadı (ctValCcy uyuşmuyor)")
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    parsed = []
    for trade in trades:
        try:
            px = float(trade["px"])
            sz = float(trade["sz"])
            ts = int(trade["ts"])
            side = trade["side"]
            if px > 0 and sz > 0 and side in ("buy", "sell") and 0 <= now_ms - ts <= 600000:
                parsed.append((side, px * sz * contract_value * contract_mult, ts))
        except (KeyError, TypeError, ValueError):
            continue
    if len(parsed) < 30:
        raise ValueError("Son 10 dakikada en az 30 geçerli işlem yok; 500 işlem sınırı nedeniyle örnek yetersiz")
    buy = sum(v for side, v, _ in parsed if side == "buy")
    sell = sum(v for side, v, _ in parsed if side == "sell")
    total = buy + sell
    if total <= 0:
        raise ValueError("İşlem nominali hesaplanamadı")
    buy_share = buy / total
    big = [t for t in parsed if t[1] >= min_trade_usd]
    big_buy = sum(t[1] for t in big if t[0] == "buy")
    big_sell = sum(t[1] for t in big if t[0] == "sell")
    big_total = big_buy + big_sell
    big_share = big_buy / big_total if big_total > 0 else np.nan
    derivatives = okx_derivatives(inst_id)
    oi = float(derivatives["OI ($)"])
    history = st.session_state.setdefault("whale_oi_history", {})
    previous = history.get(inst_id)
    now = pd.Timestamp.now(tz="UTC")
    oi_change = np.nan
    if np.isfinite(oi) and oi > 0:
        if previous is not None:
            prev_oi, prev_ts = previous
            age = (now - prev_ts).total_seconds()
            if 60 <= age <= 86400 and prev_oi > 0:
                oi_change = (oi / prev_oi - 1) * 100
        if previous is None or (now - previous[1]).total_seconds() >= 60:
            history[inst_id] = (oi, now)
    frame = okx_candles(inst_id, "1h")
    if frame.empty or frame.iloc[-1]["confirm"] != "0":
        raise ValueError("Güncel 1 saatlik mum alınamadı")
    hour_open = float(frame.iloc[-1]["open"])
    price_change = (price / hour_open - 1) * 100 if hour_open > 0 else np.nan
    # No signal before an actual OI comparison is available.
    long_tests = {"Agresif alış ≥%58": buy_share >= .58,
                  "1s fiyat artışı ≥%0,2": price_change >= .2,
                  "OI artışı ≥%1": np.isfinite(oi_change) and oi_change >= 1}
    short_tests = {"Agresif satış ≥%58": buy_share <= .42,
                   "1s fiyat düşüşü ≥%0,2": price_change <= -.2,
                   "OI artışı ≥%1": np.isfinite(oi_change) and oi_change >= 1}
    if not np.isfinite(oi_change):
        status = "⏳ OI karşılaştırması bekleniyor"
    elif all(long_tests.values()):
        status = "🟢 LONG yönlü akış"
    elif all(short_tests.values()):
        status = "🔴 SHORT yönlü akış"
    elif sum(long_tests.values()) >= 2:
        status = "🟡 LONG akış adayı"
    elif sum(short_tests.values()) >= 2:
        status = "🟠 SHORT akış adayı"
    else:
        status = "⚪ Yön teyidi yok"
    return {"Parite": inst_id, "Akış durumu": status, "Fiyat ($)": price,
            "1s fiyat değişimi %": round(price_change, 2),
            "Agresif alış payı %": round(buy_share * 100, 1),
            "Büyük işlem alış payı %": round(big_share * 100, 1) if np.isfinite(big_share) else np.nan,
            "Büyük işlem sayısı": len(big), "İncelenen işlem": len(parsed),
            "Örneklem süresi sn": round((max(x[2] for x in parsed) - min(x[2] for x in parsed)) / 1000),
            "OI ($)": oi, "OI değişimi %": round(oi_change, 2) if np.isfinite(oi_change) else np.nan,
            "Funding %": derivatives["Funding %"],
            "LONG akış koşulu": f"{sum(long_tests.values())}/3",
            "SHORT akış koşulu": f"{sum(short_tests.values())}/3",
            "Eksik LONG": ", ".join(k for k, v in long_tests.items() if not v),
            "Eksik SHORT": ", ".join(k for k, v in short_tests.items() if not v),
            "Ticker UTC": item["Ticker UTC"]}


def whale_radar():
    st.subheader("🐋 OKX Balina / Akıllı Para — Bağımsız araştırma radarı")
    st.info("Yalnızca OKX halka açık USDT perpetual işlemleri kullanılır. Cüzdan, yatırımcı kimliği veya %80 başarı oranı doğrulanamaz. Büyük işlem = nominali seçilen eşiği aşan gerçekleşmiş işlem; aynı kişi anlamına gelmez.")
    w1, w2, w3 = st.columns(3)
    with w1:
        top_n = st.slider("Hacme göre taranacak parite", 3, 30, 10, key="whale_limit")
    with w2:
        minimum = st.number_input("Minimum 24s hacim ($ milyon)", min_value=0., value=10., step=5., key="whale_minimum")
    with w3:
        large_trade = st.number_input("Büyük işlem eşiği ($)", min_value=1000, value=50000, step=10000, key="whale_large_trade")
    st.caption("Akış: son en fazla 500 gerçekleşmiş işlem, son 10 dakika; 1 saatlik fiyat değişimi ve iki ayrı zamanda gözlemlenmiş OI. OI karşılaştırması için sekmeyi açık tutup en az 60 saniye sonra yenile.")
    try:
        universe = okx_perpetual_universe()
        universe = universe[(universe["24s hacim yaklaşık ($)"] >= minimum * 1e6)
                            & (~universe["Sembol"].str.lower().isin(STABLE))]
        instruments = okx_public("/api/v5/public/instruments", {"instType": "SWAP"})
        by_id = {x["instId"]: x for x in instruments}
        rows, errors = [], []
        for _, item in universe.head(top_n).iterrows():
            try:
                rows.append(whale_market_reading(item, by_id[item["Parite"]], large_trade))
            except Exception as exc:
                errors.append(f'{item["Parite"]}: {exc}')
        if rows:
            result = pd.DataFrame(rows)
            priority = {"🟢 LONG yönlü akış": 0, "🔴 SHORT yönlü akış": 1,
                        "🟡 LONG akış adayı": 2, "🟠 SHORT akış adayı": 3,
                        "⏳ OI karşılaştırması bekleniyor": 4, "⚪ Yön teyidi yok": 5}
            result["_rank"] = result["Akış durumu"].map(priority).fillna(6)
            result = result.sort_values("_rank").drop(columns="_rank")
            st.dataframe(result, use_container_width=True, hide_index=True)
            st.download_button("📥 OKX balina akışı CSV",
                               result.to_csv(index=False).encode("utf-8-sig"),
                               "burak_okx_balina_akisi.csv", "text/csv", key="whale_csv")
        else:
            st.warning("Şu anda güvenilir akış analizi için yeterli OKX verisi bulunamadı.")
        if errors:
            with st.expander(f"Veri alınamayan pariteler ({len(errors)})"):
                st.write("\n".join(errors))
    except Exception as exc:
        st.error(f"OKX balina radarı yüklenemedi: {type(exc).__name__}: {exc}")
    st.caption("Yorum: LONG akış = alış payı ≥%58 + 1s fiyat ≥%0,2 + OI ≥%1; SHORT akış = satış payı ≥%58 + 1s fiyat ≤-%0,2 + OI ≥%1. Bu eşikler deneysel başlangıç parametreleridir, test edilmiş kazanma oranı değildir. OI artışı pozisyon yönünü tek başına göstermez. Fonlama ve büyük işlem payı bilgi amaçlıdır. Otomatik emir yok.")


with whale_tab:
    whale_refresh = st.selectbox("🐋 Balina radarı yenileme", [1, 2, 3, 5, 10, 15], index=1,
                                 format_func=lambda x: f"{x} dakika", key="whale_refresh")
    st.fragment(run_every=f"{whale_refresh * 60}s")(whale_radar)()




def market_trend_reading(df, symbol, timeframe):
    """Descriptive EMA / MACD / ADX / RSI snapshot from completed OKX candles."""
    closed = df[df["confirm"] == "1"].copy()
    if len(closed) < 205:
        raise ValueError(f"{symbol}: en az 205 kapanmış mum gerekli")
    c, h, l = closed.close, closed.high, closed.low
    e20 = c.ewm(span=20, adjust=False).mean()
    e50 = c.ewm(span=50, adjust=False).mean()
    e200 = c.ewm(span=200, adjust=False).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    up, down = h.diff(), -l.diff()
    pdm = up.where((up > down) & (up > 0), 0.)
    mdm = down.where((down > up) & (down > 0), 0.)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean().replace(0, np.nan)
    pdi = 100 * pdm.ewm(alpha=1/14, adjust=False).mean() / atr
    mdi = 100 * mdm.ewm(alpha=1/14, adjust=False).mean() / atr
    dx = 100 * (pdi-mdi).abs() / (pdi+mdi).replace(0, np.nan)
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    price = float(c.iloc[-1])
    r, a = float(rsi.iloc[-1]), float(adx.iloc[-1])
    if not all(np.isfinite(x) for x in (price, r, a, float(pdi.iloc[-1]), float(mdi.iloc[-1]))):
        raise ValueError(f"{symbol}: RSI/ADX hesaplanamadı")
    bull = {"EMA20/50": price > e20.iloc[-1] > e50.iloc[-1],
            "EMA200": price > e200.iloc[-1],
            "MACD": macd.iloc[-1] > sig.iloc[-1],
            "DI": pdi.iloc[-1] > mdi.iloc[-1],
            "RSI": r >= 50}
    bear = {"EMA20/50": price < e20.iloc[-1] < e50.iloc[-1],
            "EMA200": price < e200.iloc[-1],
            "MACD": macd.iloc[-1] < sig.iloc[-1],
            "DI": mdi.iloc[-1] > pdi.iloc[-1],
            "RSI": r < 50}
    bc, sc = sum(bull.values()), sum(bear.values())
    if a < 20:
        state = "🟡 Yatay / zayıf trend"
    elif bc >= 4 and bc > sc:
        state = "🟢 Yükseliş"
    elif sc >= 4 and sc > bc:
        state = "🔴 Düşüş"
    else:
        state = "🟡 Kararsız"
    return {"Parite": symbol, "Zaman dilimi": timeframe, "Trend": state,
            "Yükseliş koşulu": f"{bc}/5", "Düşüş koşulu": f"{sc}/5",
            "ADX": round(a, 1), "RSI": round(r, 1),
            "EMA20 üstü": bool(price > e20.iloc[-1]),
            "EMA50 üstü": bool(price > e50.iloc[-1]),
            "EMA200 üstü": bool(price > e200.iloc[-1]),
            "MACD yükseliş": bool(macd.iloc[-1] > sig.iloc[-1]),
            "Kapanış ($)": price,
            "Son kapanmış mum UTC": closed.date.iloc[-1]}


def market_direction_summary(rows, timeframe):
    part = pd.DataFrame([x for x in rows if x["Zaman dilimi"] == timeframe])
    if part.empty:
        return None
    valid = len(part)
    up = int((part["Trend"] == "🟢 Yükseliş").sum())
    down = int((part["Trend"] == "🔴 Düşüş").sum())
    btc = part.loc[part["Parite"] == "BTC-USDT-SWAP"]
    eth = part.loc[part["Parite"] == "ETH-USDT-SWAP"]
    btc_trend = btc.iloc[0]["Trend"] if not btc.empty else "Veri yok"
    eth_trend = eth.iloc[0]["Trend"] if not eth.empty else "Veri yok"
    # Breadth is equally weighted among successfully scanned, volume-selected pairs.
    up_share, down_share = up / valid, down / valid
    if up_share >= .60 and btc_trend == "🟢 Yükseliş" and eth_trend == "🟢 Yükseliş":
        status = "🟢 Geniş katılımlı yükseliş"
    elif down_share >= .60 and btc_trend == "🔴 Düşüş" and eth_trend == "🔴 Düşüş":
        status = "🔴 Geniş katılımlı düşüş"
    elif up_share >= .50:
        status = "🟡 Yükseliş eğilimi / ayrışma"
    elif down_share >= .50:
        status = "🟡 Düşüş eğilimi / ayrışma"
    else:
        status = "⚪ Karışık / yatay"
    return {"Zaman dilimi": timeframe, "Piyasa durumu": status, "BTC": btc_trend,
            "ETH": eth_trend, "Yükselişte %": round(up_share * 100, 1),
            "Düşüşte %": round(down_share * 100, 1),
            "Yatay/kararsız %": round((valid-up-down) / valid * 100, 1),
            "Analiz edilen": valid, "Ortanca ADX": round(float(part["ADX"].median()), 1)}


def market_direction_radar():
    st.subheader("🌍 OKX Genel Piyasa Yönü — 4 Saatlik ve Günlük")
    st.caption("Yalnızca OKX USDT perpetual kapanmış mumları. BTC ve ETH her zaman dahil edilir; kalan pariteler yaklaşık 24 saatlik hacme göre seçilir. Ana radar ve balina radarının parametreleri değişmez.")
    c1, c2 = st.columns(2)
    with c1:
        market_n = st.slider("Taranacak toplam parite", 10, 60, 30, 5, key="market_n")
    with c2:
        market_min = st.number_input("Altcoin minimum 24s hacim ($ milyon)", min_value=0., value=5., step=5., key="market_min")
    try:
        universe = okx_perpetual_universe()
        filtered = universe[(universe["24s hacim yaklaşık ($)"] >= market_min * 1e6)
                            & (~universe["Sembol"].str.lower().isin(STABLE))]
        selected = list(dict.fromkeys(["BTC-USDT-SWAP", "ETH-USDT-SWAP"] +
                                      filtered["Parite"].tolist()))[:market_n]
        rows, errors = [], []
        with st.spinner("OKX 4 saatlik ve günlük kapanmış mumlar taranıyor..."):
            for inst in selected:
                for tf in ("4h", "1d"):
                    try:
                        rows.append(market_trend_reading(okx_candles(inst, tf), inst, tf))
                    except Exception as exc:
                        errors.append(f"{inst} {tf}: {exc}")
        summaries = [x for tf in ("4h", "1d")
                     if (x := market_direction_summary(rows, tf)) is not None]
        if summaries:
            cols = st.columns(len(summaries))
            for col, x in zip(cols, summaries):
                with col:
                    st.metric(f'{x["Zaman dilimi"]} piyasa', x["Piyasa durumu"])
                    st.write(f'BTC: {x["BTC"]} · ETH: {x["ETH"]}')
                    st.write(f'Yükselişte **%{x["Yükselişte %"]}** · Düşüşte **%{x["Düşüşte %"]}**')
                    st.caption(f'Analiz edilen: {x["Analiz edilen"]}/{len(selected)} · Ortanca ADX: {x["Ortanca ADX"]}')
            by_tf = {x["Zaman dilimi"]: x for x in summaries}
            if "4h" in by_tf and "1d" in by_tf:
                four, day = by_tf["4h"]["Piyasa durumu"], by_tf["1d"]["Piyasa durumu"]
                if four.startswith("🟢") and day.startswith("🟢"):
                    interpretation = "4 saatlik ve günlük yükseliş yönünde uyumlu."
                elif four.startswith("🔴") and day.startswith("🔴"):
                    interpretation = "4 saatlik ve günlük düşüş yönünde uyumlu."
                elif four.startswith("🟢") and day.startswith("🔴"):
                    interpretation = "Günlük düşüş içinde 4 saatlik toparlanma; kesin trend dönüşü değildir."
                elif four.startswith("🔴") and day.startswith("🟢"):
                    interpretation = "Günlük yükseliş içinde 4 saatlik düzeltme; kesin trend dönüşü değildir."
                else:
                    interpretation = "Zaman dilimleri veya piyasa katılımı ayrışıyor; yön teyidi sınırlı."
                st.info("**Zaman dilimi ilişkisi:** " + interpretation)
            st.subheader("💸 OKX fonlama oranları")
            st.caption("Anlık OKX USDT perpetual funding oranlarıdır; pozitif değer LONG tarafının, negatif değer SHORT tarafının ödeme yaptığı olağan durumu gösterir. Fonlama periyodu sözleşmeye göre değişebilir.")
            funding_rows, funding_errors = [], []
            for inst in selected:
                try:
                    funding_data = okx_public("/api/v5/public/funding-rate", {"instId": inst})
                    if funding_data:
                        entry = funding_data[0]
                        rate = float(entry["fundingRate"]) * 100
                        next_ts = entry.get("nextFundingTime")
                        funding_rows.append({"Parite": inst, "Funding %": round(rate, 5),
                                             "Sonraki fonlama UTC": pd.to_datetime(int(next_ts), unit="ms", utc=True) if next_ts else pd.NaT})
                except Exception as exc:
                    funding_errors.append(f"{inst}: {exc}")
            if funding_rows:
                funding_df = pd.DataFrame(funding_rows)
                fc1, fc2, fc3 = st.columns(3)
                for col, inst, label in ((fc1, "BTC-USDT-SWAP", "BTC funding"),
                                          (fc2, "ETH-USDT-SWAP", "ETH funding")):
                    matched = funding_df.loc[funding_df["Parite"] == inst, "Funding %"]
                    col.metric(label, f"{matched.iloc[0]:+.5f}%" if not matched.empty else "Veri yok")
                fc3.metric("Taranan paritelerde ortanca funding", f'{funding_df["Funding %"].median():+.5f}%')
                st.dataframe(funding_df, hide_index=True, use_container_width=True)
            else:
                st.warning("OKX fonlama verisi alınamadı.")
            if funding_errors:
                st.caption(f"{len(funding_errors)} paritenin fonlama verisi eksik.")

            st.subheader("😨 OKX piyasa korku / iştah göstergesi")
            st.caption("Resmî Crypto Fear & Greed Index DEĞİLDİR. Yalnızca OKX verileriyle hesaplanan, 0–100 arası deneysel piyasa duyarlılığı göstergesidir.")
            day_data = pd.DataFrame([x for x in rows if x["Zaman dilimi"] == "1d"])
            if len(day_data) >= 5:
                bullish_share = float((day_data["Trend"] == "🟢 Yükseliş").mean())
                bearish_share = float((day_data["Trend"] == "🔴 Düşüş").mean())
                neutral_share = max(0., 1. - bullish_share - bearish_share)
                breadth_score = 100 * (bullish_share + neutral_share * .5)
                rsi_score = float(day_data["RSI"].clip(0, 100).median())
                fear_score = round(.6 * breadth_score + .4 * rsi_score)
                if fear_score < 20:
                    mood = "Yoğun korku / satış baskısı"
                elif fear_score < 40:
                    mood = "Korku / zayıf piyasa"
                elif fear_score <= 60:
                    mood = "Dengeli / kararsız"
                elif fear_score <= 80:
                    mood = "Yükseliş iştahı"
                else:
                    mood = "Yoğun yükseliş iştahı"
                st.metric("OKX duyarlılık puanı (0–100)", f"{fear_score}/100", mood)
                st.progress(fear_score / 100)
                st.caption(f"Günlük piyasa genişliği %60 + günlük RSI14 ortancası %40; {len(day_data)} analiz edilebilir parite. Fonlama, bu puana dahil edilmez; ayrı gösterilir. Bu bir anket veya yatırımcı psikolojisinin doğrudan ölçümü değildir.")
            else:
                st.warning("OKX duyarlılık puanı için en az 5 günlük parite analizi gerekli.")
            st.dataframe(pd.DataFrame(summaries), hide_index=True, use_container_width=True)
            detail = pd.DataFrame(rows)
            st.subheader("Parite bazında trend ve piyasa genişliği")
            st.dataframe(detail, hide_index=True, use_container_width=True)
            st.download_button("📥 Piyasa trend CSV",
                               detail.to_csv(index=False).encode("utf-8-sig"),
                               "burak_okx_piyasa_trend.csv", "text/csv", key="market_csv")
        else:
            st.warning("Piyasa yönü için yeterli kapanmış mum alınamadı.")
        if errors:
            st.warning(f"{len(errors)} parite/zaman diliminde veri eksik; oranlar yalnızca başarılı analizler üzerinden hesaplandı.")
            with st.expander("Eksik veri ayrıntıları"):
                st.write("\n".join(errors))
    except Exception as exc:
        st.error(f"Piyasa yönü yüklenemedi: {type(exc).__name__}: {exc}")
    st.caption("Trend: ADX≥20 ve 5 yön koşulundan en az 4'ü. Geniş katılımlı yön: tarananların ≥%60'ı aynı yönde, BTC ve ETH de aynı yönde. Eşikler araştırma amaçlıdır; getiri/başarı garantisi değildir. OI/funding bu sürümde piyasa yönü sınıflandırmasına dahil edilmez.")



@st.cache_data(ttl=20, show_spinner=False)
def okx_book_snapshot(inst_id):
    """Public resting limit orders, not open positions or liquidation levels."""
    books = okx_public("/api/v5/market/books", {"instId": inst_id, "sz": "400"})
    if not books:
        raise ValueError("OKX emir defteri boş")
    instrument = okx_public("/api/v5/public/instruments",
                            {"instType": "SWAP", "instId": inst_id})
    if not instrument:
        raise ValueError("Sözleşme çarpanı alınamadı")
    meta = instrument[0]
    ct_val = float(meta.get("ctVal") or 0)
    ct_mult = float(meta.get("ctMult") or 1)
    base = inst_id.removesuffix("-USDT-SWAP")
    if ct_val <= 0 or ct_mult <= 0 or meta.get("ctValCcy") != base:
        raise ValueError("Sözleşme nominali güvenle USD'ye çevrilemedi")
    rows = []
    for side, key in (("ALIŞ", "bids"), ("SATIŞ", "asks")):
        for entry in books[0].get(key, []):
            try:
                price, contracts = float(entry[0]), float(entry[1])
                count = int(entry[3]) if len(entry) > 3 else 0
                if price > 0 and contracts > 0:
                    rows.append({"Taraf": side, "Fiyat ($)": price,
                                 "Emir (kontrat)": contracts,
                                 "Nominal ($)": price * contracts * ct_val * ct_mult,
                                 "Emir sayısı": count})
            except (ValueError, TypeError, IndexError):
                continue
    if not rows:
        raise ValueError("Geçerli emir defteri seviyesi yok")
    return pd.DataFrame(rows), pd.to_datetime(int(books[0]["ts"]), unit="ms", utc=True)


def orderbook_heatmap():
    st.subheader("🔥 OKX Emir Defteri Isı Haritası")
    st.info("Bu harita gerçekleşmemiş BEKLEYEN limit alış/satış emirlerini gösterir; açılmış LONG/SHORT pozisyonlarını, pozisyonların giriş fiyatlarını veya likidasyon kümelerini göstermez. Emirler anlık iptal edilebilir.")
    x1, x2, x3 = st.columns([2, 1, 1])
    with x1:
        symbol = st.text_input("USDT perpetual coin", "BTC", key="depth_symbol",
                               help="BTC, ETH, SOL veya BTC-USDT-SWAP yazabilirsin.")
    with x2:
        bands = st.selectbox("Fiyat dilimi", [30, 50, 80, 100],
                             index=1, key="depth_bands")
    with x3:
        timeframe = st.selectbox("⏱️ Zaman dilimi (bilgi)", ["1h", "4h", "1d"],
                                 format_func=lambda x: {"1h": "1 saat", "4h": "4 saat", "1d": "Günlük"}[x],
                                 key="depth_timeframe")
    st.caption("Tüm alınabilen emir defteri fiyat kademeleri gösterilir; mum veya yüzde aralığı filtresi yoktur. Zaman dilimi yalnızca bilgi amaçlıdır: emir defteri anlıktır, geçmiş 1s/4s/günlük emir birikimi değildir.")
    raw = symbol.strip().upper().replace("/", "-").replace(" ", "")
    base = raw.removesuffix("-USDT-SWAP").removesuffix("-USDT")
    if not base or not base.replace("-", "").isalnum():
        st.warning("Geçerli coin sembolü gir.")
        return
    inst_id = base + "-USDT-SWAP"
    try:
        with st.spinner(f"{inst_id} emir defteri alınıyor..."):
            df, ts = okx_book_snapshot(inst_id)
        best_bid = df.loc[df["Taraf"] == "ALIŞ", "Fiyat ($)"].max()
        best_ask = df.loc[df["Taraf"] == "SATIŞ", "Fiyat ($)"].min()
        if not np.isfinite(best_bid) or not np.isfinite(best_ask):
            st.warning("Alış ve satış tarafları birlikte alınamadı.")
            return
        mid = (best_bid + best_ask) / 2
        lo, hi = float(df["Fiyat ($)"].min()), float(df["Fiyat ($)"].max())
        if hi <= lo:
            st.warning("Fiyat kademeleri harita oluşturmak için yeterli değil.")
            return
        inside = df.copy()
        if inside.empty:
            st.warning("Seçilen fiyat aralığında emir bulunamadı.")
            return
        edges = np.linspace(lo, hi, bands + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        bid = inside[inside["Taraf"] == "ALIŞ"]
        ask = inside[inside["Taraf"] == "SATIŞ"]
        buy_usd = np.histogram(bid["Fiyat ($)"], bins=edges,
                               weights=bid["Nominal ($)"])[0]
        sell_usd = np.histogram(ask["Fiyat ($)"], bins=edges,
                                weights=ask["Nominal ($)"])[0]
        values = np.column_stack([buy_usd, sell_usd])
        peak = float(values.max())
        if peak <= 0:
            st.warning("Gösterilebilir emir nominali bulunamadı.")
            return
        heat = go.Figure(go.Heatmap(
            x=["🟢 ALIŞ", "🔴 SATIŞ"], y=centers,
            z=np.log1p(values), customdata=values,
            colorscale=[[0, "#111827"], [0.15, "#334155"], [0.4, "#0f766e"],
                        [0.7, "#f59e0b"], [1, "#ef4444"]],
            zmin=0, zmax=np.log1p(peak),
            hovertemplate="Taraf: %{x}<br>Fiyat: $%{y:,.6g}<br>Bekleyen emir: $%{customdata:,.0f}<extra></extra>",
            colorbar=dict(title="log(1 + USD)")))
        heat.add_hline(y=mid, line_dash="dash", line_color="#e5e7eb",
                       annotation_text="Orta fiyat", annotation_position="top right")
        heat.update_layout(height=620, xaxis_title="Emir tarafı",
                           yaxis_title="Fiyat ($)", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(heat, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("En iyi alış ($)", f"{best_bid:,.6g}")
        c2.metric("En iyi satış ($)", f"{best_ask:,.6g}")
        c3.metric("Gösterilen nominal ($)", f"{inside['Nominal ($)'].sum():,.0f}")
        st.caption(f"OKX anlık defter zamanı: {ts.strftime('%Y-%m-%d %H:%M:%S UTC')} · "
                   f"Orta fiyat: ${mid:,.6g} · Defter fiyat aralığı: ${lo:,.6g}–${hi:,.6g} · "
                   "Her taraftan en fazla 400 fiyat kademesi; yalnızca bu kademeler toplanır. "
                   "Renk yoğunluğu logaritmiktir; dolar tutarı üzerine gelince görünür.")
        band_table = pd.DataFrame({"Fiyat ($)": centers,
                                   "Bekleyen alış ($)": buy_usd,
                                   "Bekleyen satış ($)": sell_usd})
        band_table["Toplam ($)"] = band_table["Bekleyen alış ($)"] + band_table["Bekleyen satış ($)"]
        st.markdown("**En yoğun 10 fiyat dilimi**")
        st.dataframe(band_table.nlargest(10, "Toplam ($)"),
                     hide_index=True, use_container_width=True)
        st.download_button("📥 Fiyat seviyelerini CSV indir",
                           band_table.to_csv(index=False).encode("utf-8-sig"),
                           f"okx_{base.lower()}_emir_defteri.csv", "text/csv",
                           key="depth_download")
        st.caption("Tüm fiyat aralıkları yalnızca OKX tarafından döndürülen defter kademelerini kapsar. "
                   "Bu tek zamanlı bir emir defteri fotoğrafıdır; geçmişte biriken likidite haritası değildir. "
                   "Açık pozisyon (OI) toplamı fiyat seviyelerine dağıtılamaz.")
    except Exception as exc:
        st.error(f"Emir defteri alınamadı: {type(exc).__name__}: {exc}")


with market_tab:
    market_refresh = st.selectbox("🌍 Piyasa yönü yenileme", [5, 10, 15, 30, 60],
                                  index=1, format_func=lambda x: f"{x} dakika",
                                  key="market_refresh")
    st.fragment(run_every=f"{market_refresh * 60}s")(market_direction_radar)()
    st.divider()
    depth_refresh = st.selectbox("🔥 Isı haritası yenileme", [1, 2, 5, 10],
                                 index=0, format_func=lambda n: f"{n} dakika",
                                 key="depth_refresh")
    st.fragment(run_every=f"{depth_refresh * 60}s")(orderbook_heatmap)()



def bist_parse_csv(upload):
    """User-supplied licensed/exported daily OHLCV, no scraping or implicit Midas API."""
    df = pd.read_csv(upload, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [str(x).strip().lower() for x in df.columns]
    aliases = {"symbol": "symbol", "ticker": "symbol", "sembol": "symbol",
               "hisse": "symbol", "date": "date", "tarih": "date",
               "open": "open", "açılış": "open", "acilis": "open",
               "high": "high", "yüksek": "high", "yuksek": "high",
               "low": "low", "düşük": "low", "dusuk": "low",
               "close": "close", "kapanış": "close", "kapanis": "close",
               "volume": "volume", "hacim": "volume"}
    df = df.rename(columns={c: aliases.get(c, c) for c in df.columns})
    required = {"symbol", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Eksik CSV sütunları: " + ", ".join(sorted(missing)))
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper().str.replace(".IS", "", regex=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["symbol", "date", "open", "high", "low", "close", "volume"])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1) & (df["volume"] >= 0)]
    df = df.drop_duplicates(["symbol", "date"], keep="last").sort_values(["symbol", "date"])
    if df.empty:
        raise ValueError("Geçerli günlük fiyat satırı bulunamadı")
    return df


def bist_resample_weekly(df):
    weekly = (df.set_index("date").resample("W-FRI")
              .agg({"open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum"}).dropna().reset_index())
    # Exclude current unfinished week to prevent false weekly confirmations.
    today = pd.Timestamp.now(tz="Europe/Istanbul").tz_localize(None).normalize()
    week_end = today + pd.Timedelta(days=(4 - today.weekday()) % 7)
    if today < week_end:
        weekly = weekly[weekly["date"] < week_end]
    return weekly


def bist_radar():
    st.subheader("🇹🇷 BIST Radar — Günlük ve Haftalık")
    st.warning("Midas'a bağlanan doğrulanmış bir genel API kullanılmıyor. Bu sekme yalnızca kullanım hakkına sahip olduğun günlük OHLCV CSV dosyasını yerel oturumda analiz eder; Midas hesabına erişmez ve web sayfasından veri kazımaz.")
    st.caption("CSV sütunları: symbol,date,open,high,low,close,volume. Her hisse için ayrı günlük satırlar; örn. THYAO,2025-01-02,300,310,295,305,12000000. Fiyatlar TL, volume adet olmalı. Bölünme/temettü düzeltmelerinin tutarlı olması gerekir.")
    upload = st.file_uploader("📂 BIST günlük fiyat/hacim CSV yükle", type=["csv"], key="bist_csv")
    if upload is None:
        st.info("Veri yüklenmedi. Canlı BIST veya Midas fiyatı gösterilmiyor; dosya yüklendiğinde teknik tarama açılır.")
        return
    try:
        data = bist_parse_csv(upload)
        results, issues = [], []
        for symbol, group in data.groupby("symbol"):
            daily = group.rename(columns={"volume": "quote_volume"}).copy()
            weekly = bist_resample_weekly(group).rename(columns={"volume": "quote_volume"})
            for tf, frame in (("Günlük", daily), ("Haftalık", weekly)):
                try:
                    if len(frame) < 205:
                        raise ValueError(f"{len(frame)}/205 kapanmış mum")
                    frame["confirm"] = "1"
                    reading = market_trend_reading(frame, symbol, tf)
                    close = frame.close
                    vr = float(frame.quote_volume.iloc[-1] / frame.quote_volume.iloc[-21:-1].mean()) if frame.quote_volume.iloc[-21:-1].mean() > 0 else np.nan
                    reading["Hacim katı"] = round(vr, 2) if np.isfinite(vr) else np.nan
                    reading["Veri son tarihi"] = frame.date.iloc[-1]
                    results.append(reading)
                except Exception as exc:
                    issues.append(f"{symbol} {tf}: {exc}")
        if not results:
            st.warning("Yeterli günlük/haftalık geçmiş yok. Günlük analiz için 205 işlem günü; haftalık analiz için yaklaşık 4 yıllık haftalık veri gerekli.")
        else:
            table = pd.DataFrame(results)
            for tf in ("Günlük", "Haftalık"):
                sub = table[table["Zaman dilimi"] == tf]
                if sub.empty:
                    continue
                up = int((sub["Trend"] == "🟢 Yükseliş").sum())
                down = int((sub["Trend"] == "🔴 Düşüş").sum())
                c1, c2, c3 = st.columns(3)
                c1.metric(f"{tf} yükseliş payı", f"{up/len(sub):.0%}")
                c2.metric(f"{tf} düşüş payı", f"{down/len(sub):.0%}")
                c3.metric(f"{tf} incelenen hisse", len(sub))
            st.dataframe(table, hide_index=True, use_container_width=True)
            st.download_button("📥 BIST analiz CSV", table.to_csv(index=False).encode("utf-8-sig"),
                               "burak_bist_radar.csv", "text/csv", key="bist_download")
            st.caption("Piyasa genişliği yalnızca yüklenen ve yeterli geçmişe sahip hisseleri kapsar; otomatik BIST 100 evreni değildir. Günlük ve haftalık sinyaller kapanmış mumlara dayanır.")
        if issues:
            with st.expander(f"Eksik analizler ({len(issues)})"):
                st.write("\n".join(issues[:250]))
    except Exception as exc:
        st.error(f"BIST CSV analiz edilemedi: {type(exc).__name__}: {exc}")


with bist_tab:
    bist_radar()


with futures:
    st.subheader("Kaldıraçlı işlemlerde senaryo ve risk")
    st.info("V2 vadeli risk ekranı hesaplama amaçlıdır. Fonlama oranı ve açık pozisyon (OI) henüz canlı bağlanmadı; bunlar spot verilerinden türetilmez.")
    st.warning("Bu ekran vadeli işlem sinyali üretmez ve pozisyon açmaz. Spot piyasa verileri vadeli piyasa fonlama, açık pozisyon veya likidasyon verisi yerine geçmez.")
    margin = st.number_input("Teminat ($)", min_value=1., value=100., step=25.)
    leverage = st.slider("Kaldıraç", 1, 20, 5)
    direction = st.radio("Yön", ["Long", "Short"], horizontal=True)
    move = st.slider("Fiyat değişimi (%)", -50., 50., -10., .5)
    signed = move if direction == "Long" else -move
    pnl = margin * leverage * signed / 100
    st.metric("Yaklaşık brüt P&L ($)", f"{pnl:,.2f}", f"{signed*leverage:.1f}% teminat değişimi")
    st.caption("Komisyon, fonlama, slippage, bakım teminatı ve borsaya özgü likidasyon kuralları dahil değildir. Likidasyon bu basit hesaplamadan daha önce gerçekleşebilir.")


with methodology:
    st.markdown("""**V4 canlı aday sinyaller:** OKX USDT perpetual ticker fiyatı ve oluşmakta olan mum (confirm=0). Fiyat/mum önbelleği 15 saniye, fonlama ve OI 60 saniye. Açık sekme seçtiğin otomatik yenileme aralığında yenilenir. Ticker UTC, OKX fiyat zaman damgasıdır. REST veri kaynakları tam eşzamanlı olmayabilir.

**RSI eşiği (orijinal):** LONG için RSI14 45–68, SHORT için RSI14 32–55. Teknik skorun RSI puanlaması ayrı bir bilgi göstergesidir, sinyal koşulu değildir.\n\n**Göstergeler:** EMA, MACD, RSI, ADX ve ATR için önceki mumlar matematiksel olarak gereklidir; geçmiş performans testi yapılmaz. Açık mum hacmi geçen süreye göre yaklaşık tam mum hacmine ölçeklenir (ilk %10 için tahmin özellikle belirsizdir). Mum kapanmadan sinyal değişebilir. ATR stop/hedef canlı ticker fiyatına göre varsayımsaldır, gerçek emir gerçekleşmesi değildir. Funding, spread, komisyon, kayma, kaldıraç ve likidasyon dahil değildir. Otomatik emir gönderilmez.
""")
