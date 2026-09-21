"""Burak Crypto Radar: educational crypto market research dashboard.
Public data only; no orders, account access, or leverage execution.
"""
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Burak Crypto Radar V2.2", page_icon="📡", layout="wide")
CG = "https://api.coingecko.com/api/v3"
HEADERS = {"User-Agent": "BurakCryptoRadar/1.0", "accept": "application/json"}
STABLE = {"usdt", "usdc", "dai", "fdusd", "tusd", "usde", "usdd", "pyusd", "frax"}


def get_json(url, params=None, headers=None):
    resp = requests.get(url, params=params, headers=headers or HEADERS, timeout=18)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=3600, show_spinner=False)
def market_data(pages, api_key):
    result = []
    hdr = dict(HEADERS)
    if api_key:
        hdr["x-cg-demo-api-key"] = api_key
    for page in range(1, pages + 1):
        result.extend(get_json(f"{CG}/coins/markets", {
            "vs_currency": "usd", "order": "market_cap_desc", "per_page": 250,
            "page": page, "sparkline": "false", "price_change_percentage": "24h,7d"
        }, hdr))
        if page < pages:
            time.sleep(1.3)
    return pd.DataFrame(result)


@st.cache_data(ttl=3600, show_spinner=False)
def exchange_symbols():
    """Kraken public spot USD/USDT pairs. No Binance dependency."""
    payload = get_json("https://api.kraken.com/0/public/AssetPairs")
    if payload.get("error"):
        raise RuntimeError("; ".join(payload["error"]))
    result = {}
    for pair_id, item in payload.get("result", {}).items():
        if item.get("status") != "online" or item.get("wsname") is None:
            continue
        base, sep, quote = item["wsname"].partition("/")
        if not sep or quote not in ("USD", "USDT"):
            continue
        base = {"XBT": "BTC", "XDG": "DOGE"}.get(base, base)
        # Prefer USD over USDT for broad coverage, but keep existing USD.
        if base not in result or quote == "USD":
            result[base] = (pair_id, quote)
    return result


@st.cache_data(ttl=300, show_spinner=False)
def candles(pair, interval):
    minutes = {"1d": 1440, "4h": 240, "1h": 60}[interval]
    payload = get_json("https://api.kraken.com/0/public/OHLC",
                       {"pair": pair, "interval": minutes})
    if payload.get("error"):
        raise RuntimeError("; ".join(payload["error"]))
    series = next((v for k, v in payload.get("result", {}).items()
                   if k != "last"), None)
    if not series:
        raise ValueError("Kraken OHLC data unavailable")
    df = pd.DataFrame(series, columns=["ts", "open", "high", "low", "close",
                                        "vwap", "volume", "trades"])
    for c in ["open", "high", "low", "close", "vwap", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["quote_volume"] = df["volume"] * df["vwap"]
    df["date"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    # Kraken returns the current uncommitted candle as the final entry.
    return df.iloc[:-1].copy()


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
    # Last fully completed candle; not live tick-by-tick.
    candle_time = df["date"].iloc[-1].strftime("%Y-%m-%d %H:%M UTC")
    # 30 trend + 10 RSI + 15 MACD + 15 ADX + 20 volume = 90.
    return {"Sinyal": direction, "Sinyal mumu": candle_time,
            "Teknik skor": round(score / 90 * 100), "RSI": round(r, 1),
            "ADX": round(a, 1), "Hacim katı": round(v, 2),
            "EMA20 üstü": bool(c > e20.iloc[-1]), "EMA50 üstü": bool(c > e50.iloc[-1]),
            "EMA200 üstü": bool(c > e200.iloc[-1]), "Kapanış": c}


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


def fundamental(row):
    mc, fdv, vol = row.get("market_cap"), row.get("fully_diluted_valuation"), row.get("total_volume")
    if pd.isna(mc) or mc <= 0 or pd.isna(fdv) or fdv <= 0 or pd.isna(vol):
        return np.nan
    ratio, turnover = fdv / mc, vol / mc
    # Transparent screening heuristic, NOT a valuation forecast.
    cap_points = 25 if 20e6 <= mc <= 300e6 else (15 if mc <= 500e6 else 0)
    dilution_points = 25 if ratio <= 1.3 else (15 if ratio <= 2 else 0)
    turnover_points = 25 if turnover >= .15 else (15 if turnover >= .05 else 0)
    liquid_points = 25 if vol >= 20e6 else (15 if vol >= 5e6 else 0)
    return cap_points + dilution_points + turnover_points + liquid_points


st.title("📡 BURAK CRYPTO RADAR V2.2")
st.caption("Piyasa araştırması • Spot aday taraması ve ayrı vadeli risk görünümü • Emir göndermez")
with st.sidebar:
    st.header("Tarama ayarları")
    api_key = st.text_input("CoinGecko Demo API anahtarı (isteğe bağlı)", type="password",
                            value=st.secrets.get("COINGECKO_API_KEY", os.getenv("COINGECKO_API_KEY", "")))
    pages = st.select_slider("Piyasa evreni", options=[250, 500, 750], value=500)
    low, high = st.slider("Market cap ($ milyon)", 0, 2000, (20, 500), step=10)
    fdv_max = st.slider("En yüksek FDV / MC", 1.0, 10.0, 2.0, .1)
    min_volume = st.number_input("En düşük 24s hacim ($ milyon)", 0., 500., 5., 1.)
    min_turnover = st.slider("En düşük hacim / MC (%)", 0, 100, 5)
    interval = st.selectbox("Teknik zaman dilimi", ["1d", "4h", "1h"], index=0)
    limit = st.slider("Teknik analiz yapılacak aday sayısı", 5, 60, 25, 5)
    st.caption("CoinGecko önbelleği 1 saat; Kraken mumları 5 dakika. Sinyaller son kapanmış muma göredir, anlık fiyat akışı değildir.")
    with st.expander("🧪 Binance bağlantı testi"):
        st.caption("Test, bu Streamlit sunucusundan yapılır; API anahtarı veya emir yetkisi gerekmez.")
        if st.button("Binance Spot ve Futures erişimini test et"):
            checks = [
                ("Spot ping", "https://api.binance.com/api/v3/ping"),
                ("Futures ping", "https://fapi.binance.com/fapi/v1/ping"),
                ("Futures sunucu saati", "https://fapi.binance.com/fapi/v1/time"),
                ("Futures pariteler", "https://fapi.binance.com/fapi/v1/exchangeInfo"),
                ("Futures BTCUSDT mum", "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1h&limit=2"),
                ("Futures fonlama", "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT"),
                ("Futures açık pozisyon", "https://fapi.binance.com/fapi/v1/openInterest?symbol=BTCUSDT"),
                ("Kraken kontrol", "https://api.kraken.com/0/public/Time"),
            ]
            for label, url in checks:
                try:
                    response = requests.get(url, headers=HEADERS, timeout=7)
                    status = response.status_code
                    if status == 200:
                        st.success(f"{label}: HTTP 200 — erişim var")
                    else:
                        st.error(f"{label}: HTTP {status} — {response.text[:180]}")
                except requests.exceptions.RequestException as exc:
                    st.error(f"{label}: bağlantı hatası — {type(exc).__name__}: {str(exc)[:130]}")
    with st.expander("🌐 Binance dil başlığı testi"):
        st.caption("Aynı Binance Futures ping adresine aynı sunucudan üç ayrı Accept-Language başlığıyla istek gönderir. API anahtarı gerekmez.")
        if st.button("Varsayılan / İngilizce / Türkçe karşılaştır"):
            url = "https://fapi.binance.com/fapi/v1/ping"
            for label, language in [("Varsayılan", None), ("İngilizce (en-US)", "en-US,en;q=0.9"), ("Türkçe (tr-TR)", "tr-TR,tr;q=0.9")]:
                hdr = dict(HEADERS)
                if language is not None:
                    hdr["Accept-Language"] = language
                try:
                    response = requests.get(url, headers=hdr, timeout=8)
                    message = f"{label}: HTTP {response.status_code}"
                    if response.status_code == 200:
                        st.success(message + " — erişim var")
                    else:
                        st.error(message + " — " + response.text[:220])
                except requests.exceptions.RequestException as exc:
                    st.error(f"{label}: {type(exc).__name__}: {str(exc)[:140]}")
    if st.button("🔄 Önbelleği temizle ve yeniden tara"):
        st.cache_data.clear()
        st.rerun()

try:
    with st.spinner("CoinGecko piyasa verileri yükleniyor..."):
        raw = market_data(pages // 250, api_key)
except Exception as exc:
    st.error(f"CoinGecko verisi alınamadı: {exc}")
    st.info("API kota sınırı veya geçici erişim sorunu olabilir. Demo API anahtarı ekleyip daha sonra yeniden deneyin.")
    st.stop()

if raw.empty:
    st.warning("Veri bulunamadı.")
    st.stop()

raw = raw[~raw.symbol.str.lower().isin(STABLE)].copy()
for col in ["market_cap", "fully_diluted_valuation", "total_volume", "current_price"]:
    raw[col] = pd.to_numeric(raw[col], errors="coerce")
raw["FDV/MC"] = raw.fully_diluted_valuation / raw.market_cap.replace(0, np.nan)
raw["Hacim/MC %"] = 100 * raw.total_volume / raw.market_cap.replace(0, np.nan)
raw["Temel ön skor"] = raw.apply(fundamental, axis=1)
selected = raw[(raw.market_cap.between(low*1e6, high*1e6)) &
               (raw["FDV/MC"] <= fdv_max) &
               (raw.total_volume >= min_volume*1e6) &
               (raw["Hacim/MC %"] >= min_turnover)].copy()
selected = selected.sort_values(["Temel ön skor", "total_volume"], ascending=False)

c1, c2, c3, c4 = st.columns(4)
c1.metric("İncelenen piyasa", len(raw))
c2.metric("Filtreyi geçen", len(selected))
c3.metric("Teknik analiz limiti", min(limit, len(selected)))
c4.metric("Son yükleme (UTC)", datetime.now(timezone.utc).strftime("%H:%M"))

spot, backtest_tab, futures, methodology = st.tabs(["🔎 Spot araştırma", "📊 Sinyal geçmiş testi", "⚠️ Vadeli risk ekranı", "ℹ️ Metodoloji"])
with st.spinner("Seçili adaylar için teknik veriler hesaplanıyor..."):
    try:
        symbols = exchange_symbols()
    except Exception as exc:
        symbols = {}
        st.warning(f"Kraken spot sembolleri alınamadı; teknik analiz boş kalabilir: {exc}")
    tech_rows = []
    errors = 0
    unmatched = 0
    for _, row in selected.head(limit).iterrows():
        sym = str(row.symbol).upper()
        if sym not in symbols:
            unmatched += 1
            continue
        try:
            price_df = candles(symbols[sym][0], interval)
            t = technical(price_df)
            if t:
                tech_rows.append({"id": row.id, **t, **levels_and_risks(price_df, row)})
        except Exception:
            errors += 1
    tech = pd.DataFrame(tech_rows)
if unmatched:
    st.info(f"{unmatched} aday için Kraken USD/USDT spot paritesi bulunamadı; temel verileri görüntülenebilir.")
if errors:
    st.warning(f"{errors} sembolün teknik verisi alınamadı. Kraken erişimi, veri geçmişi veya istek limiti etkili olabilir.")

if not selected.empty:
    view = selected.merge(tech, on="id", how="left") if not tech.empty else selected.copy()
    view["Birleşik araştırma skoru"] = (0.6*view["Temel ön skor"] + 0.4*view["Teknik skor"]).round() if not tech.empty else np.nan
    view = view.sort_values("Birleşik araştırma skoru", ascending=False, na_position="last")
    if "Sinyal" in view.columns:
        view["Sinyal"] = view["Sinyal"].fillna("⚪ VERİ YOK")
else:
    view = selected.copy()

with spot:
    st.subheader("Piyasa ön elemesi ve teknik durum")
    st.caption("🟢 LONG: yükseliş koşulları • 🔴 SHORT: düşüş koşulları • ⚪ BEKLE: koşullar yetersiz. Son tamamlanmış mum kullanılır; işlem emri değildir.")
    if "Sinyal" in view:
        counts = view["Sinyal"].value_counts()
        a1, a2, a3 = st.columns(3)
        a1.metric("🟢 LONG", int(counts.get("🟢 LONG", 0)))
        a2.metric("🔴 SHORT", int(counts.get("🔴 SHORT", 0)))
        a3.metric("⚪ BEKLE / VERİ YOK", int(counts.get("⚪ BEKLE", 0) + counts.get("⚪ VERİ YOK", 0)))
    st.caption("Birleşik skor yalnızca iki veri grubu mevcutsa hesaplanır. Eksik teknik veri sıfır sayılmaz.")
    cols = ["name", "symbol"] + (["Sinyal", "Sinyal mumu"] if "Sinyal" in view else []) + ["market_cap", "fully_diluted_valuation", "FDV/MC", "total_volume", "Hacim/MC %", "Temel ön skor"]
    cols += [x for x in ["Teknik skor", "Birleşik araştırma skoru", "RSI", "ADX", "Hacim katı", "Dolaşım %", "EMA20 ($)", "EMA50 ($)", "20 mum destek ($)", "20 mum direnç ($)", "ATR14 %", "Risk notları"] if x in view]
    table = view[cols].rename(columns={"name":"Coin", "symbol":"Sembol", "market_cap":"MC ($)",
                    "fully_diluted_valuation":"FDV ($)", "total_volume":"24s hacim ($)"})
    if "Sinyal" in table:
        def signal_color(row):
            status = row["Sinyal"]
            bg = ("background-color: #143d2b; color: #e6fff0" if status == "🟢 LONG"
                  else "background-color: #52232b; color: #fff0f0" if status == "🔴 SHORT"
                  else "")
            return [bg] * len(row)
        st.dataframe(table.style.apply(signal_color, axis=1),
                     hide_index=True, use_container_width=True)
    else:
        st.dataframe(table, hide_index=True, use_container_width=True)
    if view.empty:
        st.info("Filtreye uyan coin yok. Filtreleri genişletebilirsin.")
    elif not tech.empty:
        available = view[view.id.isin(tech.id)]
        if not available.empty:
            choice = st.selectbox("Detay grafiği", available.id.tolist(),
                                  format_func=lambda x: available.loc[available.id == x, "name"].iloc[0])
            row = available[available.id == choice].iloc[0]
            try:
                chart = candles(symbols[str(row.symbol).upper()][0], interval)
                fig = go.Figure(go.Candlestick(x=chart.date, open=chart.open, high=chart.high,
                                               low=chart.low, close=chart.close, name="Fiyat"))
                for n in (20, 50, 200):
                    fig.add_trace(go.Scatter(x=chart.date, y=chart.close.ewm(span=n, adjust=False).mean(),
                                             name=f"EMA {n}", mode="lines"))
                if "20 mum destek ($)" in row and pd.notna(row["20 mum destek ($)"]):
                    fig.add_hline(y=float(row["20 mum destek ($)"]), line_dash="dash",
                                  annotation_text="20 mum destek")
                    fig.add_hline(y=float(row["20 mum direnç ($)"]), line_dash="dash",
                                  annotation_text="20 mum direnç")
                fig.update_layout(height=520, xaxis_rangeslider_visible=False, template="plotly_dark")
                st.caption("Destek/direnç önceki 20 tamamlanmış mumdan hesaplanır; "
                           "EMA seviyeleri olası izleme bölgeleridir, alım emri değildir.")
                st.dataframe(pd.DataFrame([{"Gösterge": k, "Değer": row[k]}
                    for k in ["EMA20 ($)", "EMA50 ($)", "EMA200 ($)",
                              "20 mum destek ($)", "20 mum direnç ($)",
                              "ATR14 ($)", "ATR14 %", "Dolaşım %", "Risk notları"]
                    if k in row.index]), hide_index=True, use_container_width=True)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.warning(f"Grafik alınamadı: {exc}")
    st.download_button("📥 Filtre sonuçlarını CSV indir", view.to_csv(index=False).encode("utf-8-sig"),
                       "burak_crypto_radar.csv", "text/csv", disabled=view.empty)


with backtest_tab:
    st.subheader("LONG / SHORT geçmiş sinyal testi")
    st.caption("Son kapanmış mumda oluşan YENİ yön sinyali → sonraki mum açılışında varsayımsal giriş → seçilen mum sayısı sonunda kapanışta çıkış. Çakışan pozisyonlar alınmaz.")
    st.warning("Bu bir geçmiş veri simülasyonudur; kârlılık garantisi, canlı emir veya vadeli piyasa backtesti değildir. Kraken spot fiyatları kullanılır; fonlama, likidasyon ve emir defteri etkileri dahil değildir.")
    if tech.empty:
        st.info("Teknik veri yok. Önce Spot araştırma için geçerli Kraken paritesi bulunan coinleri tara.")
    else:
        bt_options = view[view.id.isin(tech.id)]
        bt_coin = st.selectbox("Test edilecek coin", bt_options.id.tolist(),
                               format_func=lambda x: bt_options.loc[bt_options.id == x, "name"].iloc[0],
                               key="backtest_coin")
        b1, b2, b3 = st.columns(3)
        with b1:
            hold = st.selectbox("Pozisyon süresi (mum)", [1, 2, 4, 8, 12, 24], index=2)
        with b2:
            fee = st.number_input("Tek yön komisyon (%)", min_value=0.0, max_value=2.0,
                                  value=0.1, step=0.01)
        with b3:
            slip = st.number_input("Tek yön kayma (%)", min_value=0.0, max_value=2.0,
                                   value=0.05, step=0.01)
        bt_row = bt_options.loc[bt_options.id == bt_coin].iloc[0]
        try:
            bt_df = candles(symbols[str(bt_row.symbol).upper()][0], interval)
            trades = backtest_directional(bt_df, hold, fee, slip)
            st.caption(f"Veri aralığı: {bt_df.date.iloc[0]:%Y-%m-%d} – {bt_df.date.iloc[-1]:%Y-%m-%d} UTC | {interval} | {len(bt_df)} kapanmış mum. İlk 205 mum indikatör ısınması için ayrılır.")
            if trades.empty:
                st.info("Bu dönemde kuralları karşılayan yeni ve tamamlanmış işlem bulunamadı. Başka coin veya zaman dilimi seç.")
            else:
                n = len(trades)
                wins = int((trades["Net %"] > 0).sum())
                compounded = (1 + trades["Net %"] / 100).cumprod()
                dd = (compounded / compounded.cummax() - 1) * 100
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("İşlem", n)
                m2.metric("Net kazançlı işlem", f"%{100*wins/n:.1f}")
                m3.metric("Ortalama net / işlem", f"%{trades['Net %'].mean():.2f}")
                m4.metric("En yüksek gerileme", f"%{dd.min():.2f}")
                st.metric("Bileşik varsayımsal sonuç (1x, tüm sermaye)", f"%{(compounded.iloc[-1]-1)*100:.2f}")
                st.line_chart(pd.DataFrame({"Bileşik çarpan": compounded.to_numpy()},
                                           index=trades["Çıkış UTC"]))
                st.dataframe(trades.sort_values("Sinyal UTC", ascending=False),
                             hide_index=True, use_container_width=True)
                st.download_button("📥 Geçmiş test CSV", trades.to_csv(index=False).encode("utf-8-sig"),
                                   f"backtest_{bt_coin}_{interval}.csv", "text/csv")
                if n < 30:
                    st.warning(f"Yalnızca {n} işlem var; örneklem küçük. Sonuçlar istatistiksel olarak güvenilir kabul edilmemeli.")
                st.caption("Bileşik sonuç, her işlemde sermayenin tamamının 1x kullanıldığını ve pozisyonların çakışmadığını varsayar. SHORT sonuçları spot mumlardan sentetik hesaplanır. Gerçek kaldıraç, marjin, fonlama, likidasyon, spread değişimi ve vergiler hesaplanmaz.")
        except Exception as exc:
            st.error(f"Geçmiş test hesaplanamadı: {exc}")

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
    st.markdown("""**V2.2 geçmiş test:** Aynı LONG/SHORT koşulları geçmiş kapanmış mumlarda tekrar hesaplanır. Yalnızca yeni sinyalde, bir sonraki mum açılışında giriş varsayılır; seçilen süre sonunda kapanışta çıkılır. Tek yön komisyon ve kayma iki kez düşülür. Çakışan işlemler atlanır. Başarı oranı yalnızca net getirisi pozitif işlemlerin payıdır. Sonuçlar sınırlı Kraken OHLC geçmişine ve seçilen coin/zaman dilimine özgüdür; ileriye dönük performans göstermez.\n\n**V2.1 yön etiketleri:** LONG için kapanış > EMA20 > EMA50, MACD > sinyal çizgisi, +DI > -DI, ADX ≥20, RSI 45–68 ve son kapanmış mumun hacmi önceki 20 mum ortalamasının ≥1,2 katı olmalı. SHORT için kapanış < EMA20 < EMA50, MACD < sinyal çizgisi, -DI > +DI, ADX ≥20, RSI 32–55 ve aynı hacim koşulu aranır. Diğer durumlar BEKLE; veri eksikse VERİ YOK. Birleşik araştırma skoru yön etiketini belirlemez. Bu koşullar geriye dönük test edilmemiştir; kaldıraçlı işlem önerisi değildir. Kraken OHLC 5 dakika önbellekli ve sadece kapanmış mum kullanılır; 1d seçimi gün içinde sürekli değişen sinyal vermez.\n\n**V2 yenilikleri:** Son 20 tamamlanmış mumun destek/direnç seviyeleri, EMA20/50/200, ATR14 volatilitesi, dolaşımdaki arz oranı ve açıklanabilir risk notları. Bunlar fiyat hedefi veya işlem sinyali değildir.\n\n**Veri kaynakları:** CoinGecko `/coins/markets` (market cap, FDV, 24 saatlik hacim); Kraken public `/AssetPairs` ve `/OHLC` (OHLCV). Kraken spot USD/USDT paritesi bulunmayan coinlerde teknik skor boş kalır. CoinGecko ve Kraken farklı fiyat/arz anlık görüntüleri sunabilir.

**Temel ön skor (0–100):** Market cap bandı 25, FDV/MC 25, hacim/MC 25, mutlak hacim 25. Bunlar kullanıcı tarafından değiştirilebilir filtrelere ek, sabit ve açıklanabilir araştırma puanlarıdır.

**Teknik skor (0–100):** EMA/trend 30, RSI 10, MACD 15, ADX 15, göreli hacim 20; toplam 90 ham puan 100'e normalize edilir. Hacim karşılaştırması Kraken USD/USDT işlem hacmi (yaklaşık VWAP × baz hacim) üzerinden yapılır. Tamamlanmış son mum kullanılır.

**Birleşik araştırma skoru:** %60 temel ön skor + %40 teknik skor. Bu, 10x/20x olasılığı veya getiri tahmini değildir; geçmiş performans testi yapılmamıştır.

**V2 kapsam dışı:** Doğrulanmış tarihli token unlock takvimi, gerçek emir defteri derinliği, TVL, protokol geliri, kullanıcı sayısı, narrative/catalyst doğrulaması, canlı fonlama ve açık pozisyon. Dolaşım yüzdesi token unlock tarihi veya miktarı değildir. Bu alanlara veri uydurulmaz. Büyük fiyat düşüşleri ve sermayenin tamamının kaybı mümkündür.

**Tarama sıklığı:** Streamlit önbelleği 1 saat; uygulama açıkken veya kullanıcı tekrar açtığında veri yenilenir. Sunucu arka planda sürekli tarama veya bildirim gönderme yapmaz.
""")
