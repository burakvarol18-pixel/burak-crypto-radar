"""BURAK CRYPTO RADAR V4.3 — OKX LIVE USDT perpetual market research only.
No orders, account access, or leverage execution.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Burak Crypto Radar V4.3 — OKX LIVE", page_icon="📡", layout="wide")
HEADERS = {"User-Agent": "BurakCryptoRadar/1.0", "accept": "application/json"}
STABLE = {"usdt", "usdc", "dai", "fdusd", "tusd", "usde", "usdd", "pyusd", "frax"}


def get_json(url, params=None, headers=None):
    resp = requests.get(url, params=params, headers=headers or HEADERS, timeout=18)
    resp.raise_for_status()
    return resp.json()


def technical(df):
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
    # Independent directional conditions; fundamental growth score does not decide short.
    bullish = (c > e20.iloc[-1] > e50.iloc[-1]
               and macd.iloc[-1] > signal.iloc[-1]
               and plus_di.iloc[-1] > minus_di.iloc[-1]
               and a >= 20 and 45 <= r <= 68 and v >= 1.2)
    bearish = (c < e20.iloc[-1] < e50.iloc[-1]
               and macd.iloc[-1] < signal.iloc[-1]
               and minus_di.iloc[-1] > plus_di.iloc[-1]
               and a >= 20 and 32 <= r <= 55 and v >= 1.2)
    direction = "🟢 LONG" if bullish else ("🔴 SHORT" if bearish else "⚪ BEKLE")

    long_tests = {"EMA": bool(c > e20.iloc[-1] > e50.iloc[-1]),
                  "MACD": bool(macd.iloc[-1] > signal.iloc[-1]),
                  "DI": bool(plus_di.iloc[-1] > minus_di.iloc[-1]),
                  "ADX": bool(a >= 20),
                  "RSI": bool(45 <= r <= 68),
                  "Hacim": bool(v >= 1.2)}
    short_tests = {"EMA": bool(c < e20.iloc[-1] < e50.iloc[-1]),
                   "MACD": bool(macd.iloc[-1] < signal.iloc[-1]),
                   "DI": bool(minus_di.iloc[-1] > plus_di.iloc[-1]),
                   "ADX": bool(a >= 20),
                   "RSI": bool(32 <= r <= 55),
                   "Hacim": bool(v >= 1.2)}
    lc, sc = sum(long_tests.values()), sum(short_tests.values())
    if bullish:
        stage, missing = "🟢 LONG", "—"
    elif bearish:
        stage, missing = "🔴 SHORT", "—"
    elif lc == 5 and sc < 5:
        stage = "🟡 LONG adayı"
        missing = ", ".join(k for k, ok in long_tests.items() if not ok)
    elif sc == 5 and lc < 5:
        stage = "🟠 SHORT adayı"
        missing = ", ".join(k for k, ok in short_tests.items() if not ok)
    elif lc == 5 and sc == 5:
        stage, missing = "⚪ Çelişkili aday", "LONG ve SHORT farklı koşullarda eksik"
    else:
        stage, missing = "⚪ BEKLE", "LONG: " + ", ".join(k for k, ok in long_tests.items() if not ok) + " | SHORT: " + ", ".join(k for k, ok in short_tests.items() if not ok)
    # The live candle may change before close.
    candle_time = df["date"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC")
    # 30 trend + 10 RSI + 15 MACD + 15 ADX + 20 volume = 90.
    return {"Sinyal": direction, "Fırsat durumu": stage, "Eksik koşul": missing,
            "LONG koşul": f"{lc}/6", "SHORT koşul": f"{sc}/6", "Sinyal mumu": candle_time,
            "Teknik skor": round(score / 90 * 100), "RSI": round(r, 1),
            "ADX": round(a, 1), "Hacim katı": round(v, 2),
            "EMA20 üstü": bool(c > e20.iloc[-1]), "EMA50 üstü": bool(c > e50.iloc[-1]),
            "EMA200 üstü": bool(c > e200.iloc[-1]), "Kapanış": c}


def fibonacci_check(df, price, side):
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
    return distance <= 1, level, distance


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



st.title("📡 BURAK CRYPTO RADAR V3 — OKX")
st.caption("Yalnızca OKX USDT perpetual verileri • LONG / SHORT araştırma sinyalleri • Otomatik emir göndermez")
with st.sidebar:
    st.header("OKX veri ayarları")
    st.caption("Canlı radar yaklaşık 20 saniyede yenilenir. Ticker ve mum önbelleği 15 sn, fonlama/OI 60 sn.")
    if st.button("🔄 OKX verilerini yenile"):
        st.cache_data.clear()
        st.rerun()

radar_tab, futures, methodology = st.tabs(["🟢🔴 OKX Perpetual Radar", "⚠️ Vadeli risk ekranı", "ℹ️ Metodoloji"])

@st.fragment(run_every="20s")
def analyze_okx_coin(item, okx_interval, stop_mult, target_mult):
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
    indicators = technical(frame)
    if indicators is None:
        raise ValueError("Yeterli teknik mum verisi yok")
    fib = {}
    for tf in ("1h", "4h"):
        fib_frame = frame if tf == okx_interval else okx_candles(inst, tf)
        if fib_frame.iloc[-1]["confirm"] != "0":
            raise ValueError("Fibonacci için açık mum yok")
        for side in ("LONG", "SHORT"):
            fib[(tf, side)] = fibonacci_check(fib_frame, live_price, side)
    old_signal = indicators["Sinyal"]
    long_n = int(indicators["LONG koşul"].split("/")[0])
    short_n = int(indicators["SHORT koşul"].split("/")[0])
    long_n += int(fib[("1h", "LONG")][0]) + int(fib[("4h", "LONG")][0])
    short_n += int(fib[("1h", "SHORT")][0]) + int(fib[("4h", "SHORT")][0])
    indicators["LONG koşul"] = f"{long_n}/8"
    indicators["SHORT koşul"] = f"{short_n}/8"
    for tf in ("1h", "4h"):
        for side in ("LONG", "SHORT"):
            passed, level, distance = fib[(tf, side)]
            indicators[f"Fib {tf} {side}"] = passed
            indicators[f"Fib {tf} {side} seviye ($)"] = level
            indicators[f"Fib {tf} {side} uzaklık ATR"] = round(distance, 2) if np.isfinite(distance) else np.nan
    missing_long = [f"Fib {tf}" for tf in ("1h", "4h") if not fib[(tf, "LONG")][0]]
    missing_short = [f"Fib {tf}" for tf in ("1h", "4h") if not fib[(tf, "SHORT")][0]]
    old_missing = indicators["Eksik koşul"]
    if old_signal == "🟢 LONG" and long_n == 8:
        indicators["Fırsat durumu"] = "🟢 LONG"
        indicators["Eksik koşul"] = "—"
    elif old_signal == "🔴 SHORT" and short_n == 8:
        indicators["Fırsat durumu"] = "🔴 SHORT"
        indicators["Eksik koşul"] = "—"
    elif long_n == 7 and short_n < 7:
        indicators["Fırsat durumu"] = "🟡 LONG adayı"
        indicators["Eksik koşul"] = ", ".join(missing_long) if missing_long else old_missing
    elif short_n == 7 and long_n < 7:
        indicators["Fırsat durumu"] = "🟠 SHORT adayı"
        indicators["Eksik koşul"] = ", ".join(missing_short) if missing_short else old_missing
    else:
        indicators["Fırsat durumu"] = "⚪ BEKLE"
        indicators["Eksik koşul"] = "LONG Fib: " + (", ".join(missing_long) or "OK") + " | SHORT Fib: " + (", ".join(missing_short) or "OK") + " | Teknik: " + old_missing
    indicators["Sinyal"] = indicators["Fırsat durumu"] if indicators["Fırsat durumu"] in ("🟢 LONG", "🔴 SHORT") else "⚪ BEKLE"
    try:
        derivative = okx_derivatives(inst)
    except Exception:
        derivative = {"Funding %": np.nan, "OI ($)": np.nan}
    levels = levels_and_risks(frame, {})
    scenario = atr_scenario(indicators, levels, stop_mult, target_mult)
    indicators["Sinyal mumu"] += " (açık / geçici)"
    return {**item.to_dict(), **indicators, **levels, **scenario, **derivative}


def live_radar():
        st.subheader("OKX USDT Perpetual — LONG / SHORT Radar")
        st.caption("OKX canlı ticker ve açık perpetual mumundan geçici LONG/SHORT adayları. Hesap bağlanmaz, emir gönderilmez.")
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
                        okx_rows.append(analyze_okx_coin(item, okx_interval, stop_mult, target_mult))
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
                show = ["Parite", "Fırsat durumu", "LONG koşul", "SHORT koşul", "Eksik koşul", "Fib 1h LONG", "Fib 4h LONG", "Fib 1h SHORT", "Fib 4h SHORT", "Sinyal", "Ticker UTC", "Sinyal mumu", "Fiyat ($)",
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
                st.caption("Funding % mevcut fonlama oranıdır; tek başına LONG/SHORT koşuluna katılmaz. OI ($) mevcut açık pozisyon anlık görüntüsüdür; OI değişimi değildir. OKX hacmi yaklaşık USD cinsindedir.")
                st.download_button("📥 OKX perpetual radar CSV",
                                   radar.to_csv(index=False).encode("utf-8-sig"),
                                   "burak_okx_perpetual_radar.csv", "text/csv")
                st.warning("Canlı aday sinyal: Açık mum kapanmadan LONG/SHORT değişebilir. Ticker UTC zamanını kontrol et; eski fiyatı işlem referansı alma.")
        except Exception as exc:
            st.error(f"OKX radar yüklenemedi: {type(exc).__name__}: {exc}")

        st.divider()
        st.subheader("🔎 Kendi coinini analiz et")
        st.caption("Ana tarama listesinden bağımsızdır. OKX USDT perpetual paritesini gir; aynı teknik + 1h/4h Fibonacci 8 koşuluyla değerlendirilir. Ana taramanın hacim ve ilk-N sınırına tabi değildir.")
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
                        custom = analyze_okx_coin(matched.iloc[0], okx_interval, stop_mult, target_mult)
                    st.metric("Fırsat durumu", custom["Fırsat durumu"])
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Canlı fiyat ($)", f'{custom["Fiyat ($)"]:,.6g}')
                    c2.metric("LONG koşul", custom["LONG koşul"])
                    c3.metric("SHORT koşul", custom["SHORT koşul"])
                    st.write("**Eksik koşul:**", custom["Eksik koşul"])
                    fields = ["Parite", "Ticker UTC", "Sinyal mumu", "Fırsat durumu", "Eksik koşul",
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
    live_radar()


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
    st.markdown("""**V4 canlı aday sinyaller:** OKX USDT perpetual ticker fiyatı ve oluşmakta olan mum (confirm=0). Fiyat/mum önbelleği 15 saniye, fonlama ve OI 60 saniye. Açık sekme yaklaşık 20 saniyede yenilenir. Ticker UTC, OKX fiyat zaman damgasıdır. REST veri kaynakları tam eşzamanlı olmayabilir.

**Göstergeler:** EMA, MACD, RSI, ADX ve ATR için önceki mumlar matematiksel olarak gereklidir; geçmiş performans testi yapılmaz. Açık mum hacmi geçen süreye göre yaklaşık tam mum hacmine ölçeklenir (ilk %10 için tahmin özellikle belirsizdir). Mum kapanmadan sinyal değişebilir. ATR stop/hedef canlı ticker fiyatına göre varsayımsaldır, gerçek emir gerçekleşmesi değildir. Funding, spread, komisyon, kayma, kaldıraç ve likidasyon dahil değildir. Otomatik emir gönderilmez.
""")
