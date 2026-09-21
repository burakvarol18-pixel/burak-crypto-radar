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

st.set_page_config(page_title="Burak Crypto Radar", page_icon="📡", layout="wide")
CG = "https://api.coingecko.com/api/v3"
BINANCE = "https://api.binance.com/api/v3"
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
    info = get_json(f"{BINANCE}/exchangeInfo")
    return {x["baseAsset"]: x["symbol"] for x in info["symbols"]
            if x["quoteAsset"] == "USDT" and x["status"] == "TRADING"
            and x.get("isSpotTradingAllowed", True)}


@st.cache_data(ttl=3600, show_spinner=False)
def candles(symbol, interval):
    data = get_json(f"{BINANCE}/klines", {"symbol": symbol, "interval": interval, "limit": 230})
    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume",
                                     "close_ts", "quote_volume", "trades", "buy_base", "buy_quote", "ignore"])
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    # Ignore the current incomplete candle to prevent unstable signals.
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
    # 30 trend + 10 RSI + 15 MACD + 15 ADX + 20 volume = 90.
    return {"Teknik skor": round(score / 90 * 100), "RSI": round(r, 1),
            "ADX": round(a, 1), "Hacim katı": round(v, 2),
            "EMA20 üstü": bool(c > e20.iloc[-1]), "EMA50 üstü": bool(c > e50.iloc[-1]),
            "EMA200 üstü": bool(c > e200.iloc[-1]), "Kapanış": c}


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


st.title("📡 BURAK CRYPTO RADAR")
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
    st.caption("Veri önbelleği 1 saat. Binance fiyat verisinde kapanmamış mum dışlanır.")
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

spot, futures, methodology = st.tabs(["🔎 Spot araştırma", "⚠️ Vadeli risk ekranı", "ℹ️ Metodoloji"])
with st.spinner("Seçili adaylar için teknik veriler hesaplanıyor..."):
    try:
        symbols = exchange_symbols()
    except Exception as exc:
        symbols = {}
        st.warning(f"Binance spot sembolleri alınamadı; teknik analiz boş kalabilir: {exc}")
    tech_rows = []
    errors = 0
    for _, row in selected.head(limit).iterrows():
        sym = str(row.symbol).upper()
        if sym not in symbols:
            continue
        try:
            t = technical(candles(symbols[sym], interval))
            if t:
                tech_rows.append({"id": row.id, **t})
        except Exception:
            errors += 1
    tech = pd.DataFrame(tech_rows)
if errors:
    st.warning(f"{errors} sembolün teknik verisi alınamadı. Binance bölgesel erişimi veya istek limiti etkili olabilir.")

if not selected.empty:
    view = selected.merge(tech, on="id", how="left") if not tech.empty else selected.copy()
    view["Birleşik araştırma skoru"] = (0.6*view["Temel ön skor"] + 0.4*view["Teknik skor"]).round() if not tech.empty else np.nan
    view = view.sort_values("Birleşik araştırma skoru", ascending=False, na_position="last")
else:
    view = selected.copy()

with spot:
    st.subheader("Piyasa ön elemesi ve teknik durum")
    st.caption("Birleşik skor yalnızca iki veri grubu mevcutsa hesaplanır. Eksik teknik veri sıfır sayılmaz.")
    cols = ["name", "symbol", "market_cap", "fully_diluted_valuation", "FDV/MC", "total_volume", "Hacim/MC %", "Temel ön skor"]
    cols += [x for x in ["Teknik skor", "Birleşik araştırma skoru", "RSI", "ADX", "Hacim katı"] if x in view]
    st.dataframe(view[cols].rename(columns={"name":"Coin", "symbol":"Sembol", "market_cap":"MC ($)",
                    "fully_diluted_valuation":"FDV ($)", "total_volume":"24s hacim ($)"}),
                 hide_index=True, use_container_width=True)
    if view.empty:
        st.info("Filtreye uyan coin yok. Filtreleri genişletebilirsin.")
    elif not tech.empty:
        available = view[view.id.isin(tech.id)]
        if not available.empty:
            choice = st.selectbox("Detay grafiği", available.id.tolist(),
                                  format_func=lambda x: available.loc[available.id == x, "name"].iloc[0])
            row = available[available.id == choice].iloc[0]
            try:
                chart = candles(symbols[str(row.symbol).upper()], interval)
                fig = go.Figure(go.Candlestick(x=chart.date, open=chart.open, high=chart.high,
                                               low=chart.low, close=chart.close, name="Fiyat"))
                for n in (20, 50, 200):
                    fig.add_trace(go.Scatter(x=chart.date, y=chart.close.ewm(span=n, adjust=False).mean(),
                                             name=f"EMA {n}", mode="lines"))
                fig.update_layout(height=520, xaxis_rangeslider_visible=False, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.warning(f"Grafik alınamadı: {exc}")
    st.download_button("📥 Filtre sonuçlarını CSV indir", view.to_csv(index=False).encode("utf-8-sig"),
                       "burak_crypto_radar.csv", "text/csv", disabled=view.empty)

with futures:
    st.subheader("Kaldıraçlı işlemlerde senaryo ve risk")
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
    st.markdown("""**Veri kaynakları:** CoinGecko `/coins/markets` (market cap, FDV, 24 saatlik hacim); Binance spot `/exchangeInfo` ve `/klines` (OHLCV). CoinGecko ve Binance farklı fiyat/arz anlık görüntüleri sunabilir.

**Temel ön skor (0–100):** Market cap bandı 25, FDV/MC 25, hacim/MC 25, mutlak hacim 25. Bunlar kullanıcı tarafından değiştirilebilir filtrelere ek, sabit ve açıklanabilir araştırma puanlarıdır.

**Teknik skor (0–100):** EMA/trend 30, RSI 10, MACD 15, ADX 15, göreli hacim 20; toplam 90 ham puan 100'e normalize edilir. Hacim karşılaştırması Binance USDT işlem hacmi üzerinden yapılır. Tamamlanmış son mum kullanılır.

**Birleşik araştırma skoru:** %60 temel ön skor + %40 teknik skor. Bu, 10x/20x olasılığı veya getiri tahmini değildir; geçmiş performans testi yapılmamıştır.

**V1 kapsam dışı:** Token unlock takvimi, gerçek emir defteri derinliği, TVL, protokol geliri, kullanıcı sayısı, narrative/catalyst doğrulaması, fonlama ve açık pozisyon. Bu alanlara veri uydurulmaz. Büyük fiyat düşüşleri ve sermayenin tamamının kaybı mümkündür.

**Tarama sıklığı:** Streamlit önbelleği 1 saat; uygulama açıkken veya kullanıcı tekrar açtığında veri yenilenir. Sunucu arka planda sürekli tarama veya bildirim gönderme yapmaz.
""")
