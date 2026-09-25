import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google import genai
from google.genai import types
from groq import Groq as GroqClient
from openai import OpenAI

# Load environment variables
load_dotenv(".env.trade", override=True)

# FIXED: Corrected Gemini model name to a valid, active model
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
AIRFORCE_MODEL = os.getenv("AIRFORCE_MODEL", "claude-3-haiku")

st.set_page_config(page_title="Crypto Monitor & AI Advisor", layout="wide", initial_sidebar_state="expanded")
st.title("🚀 Crypto Scanner & AI Advisor (Long / Short)")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    scan_mode = st.radio("Scan Mode", ["🔍 Auto-Discover Viable Coins", "📝 Manual Coin List"], index=0)
    
    if scan_mode == "📝 Manual Coin List":
        coin_input = st.text_area(
            "Coins to Monitor (comma-separated)",
            value="BTC,ETH,SOL,BNB,XRP,ADA,AVAX,LINK,DOGE,KAS,NEAR,RUNE,SUI,THETA,ATOM,ZCASH",
            height=90, help="Enter symbols without USDT."
        )
    else:
        st.info("🔍 Auto-Scanner will find the top 5-10 coins based on Volume, RSI, S/R, and Volatility.")
        coin_input = ""

    timeframes = st.multiselect("Timeframes to Analyze", ["1m", "5m", "15m", "1h", "4h", "1d"], default=["1h", "4h"])
    
    st.subheader("Signal Thresholds")
    price_boost_pct = st.slider("Price Threshold %", 0.0, 50.0, 3.0, 0.5)
    volume_boost_pct = st.slider("Volume Spike %", 0.0, 200.0, 40.0, 10.0)
    rsi_threshold = st.slider("RSI Boundary (Upper / Lower)", 50, 80, 55, 1)
    
    st.subheader("💾 Options")
    enable_logging = st.checkbox("Log Scan Results to CSV", value=True)
    
    st.subheader("🤖 AI Advisor")
    ai_provider = st.radio(
        "Select AI Provider", ["Gemini (Default)", "Groq (Fast)", "Apiforce (DeepSeek V3)"],
        help="Falls back automatically if unavailable."
    )

@st.cache_resource
def init_binance():
    api_key = os.getenv('BINANCE_API_KEY')
    secret_key = os.getenv('BINANCE_SECRET_KEY')
    config = {'enableRateLimit': True}
    if api_key and secret_key:
        config['apiKey'] = api_key
        config['secret'] = secret_key
    try:
        return ccxt.binance(config)
    except Exception as e:
        st.error(f"Binance connection failed: {str(e)}")
        st.stop()

# --- INDICATOR CALCULATIONS ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_adx(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    plus_dm = high.diff().clip(lower=0)
    minus_dm = low.diff().abs().clip(lower=0) # Corrected for minus DM
    
    tr_smooth = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / tr_smooth)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / tr_smooth)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx

def calculate_vwap(df):
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    cum_vol = df['volume'].cumsum()
    cum_tp_vol = (typical_price * df['volume']).cumsum()
    return cum_tp_vol / cum_vol

def calculate_stoch_rsi(series, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):
    rsi = calculate_rsi(series, rsi_period)
    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min)
    stoch_rsi_k = stoch_rsi.fillna(0.5).rolling(k_smooth).mean() * 100
    stoch_rsi_d = stoch_rsi_k.rolling(d_smooth).mean()
    return stoch_rsi_k, stoch_rsi_d

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal, macd - macd_signal

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.ewm(span=period, adjust=False).mean()

def find_support_resistance(df, window=10, tolerance=0.03):
    highs = df['high'].rolling(window=window, center=True).max()
    lows = df['low'].rolling(window=window, center=True).min()
    current_price = df['close'].iloc[-1]
    recent_highs = highs.dropna().iloc[-5:]
    recent_lows = lows.dropna().iloc[-5:]
    resistance = recent_highs[recent_highs > current_price].min() if not recent_highs[recent_highs > current_price].empty else np.nan
    support = recent_lows[recent_lows < current_price].max() if not recent_lows[recent_lows < current_price].empty else np.nan
    dist_res = abs(resistance - current_price) / current_price if not pd.isna(resistance) else 1
    dist_sup = abs(support - current_price) / current_price if not pd.isna(support) else 1
    return (dist_res <= tolerance) or (dist_sup <= tolerance), support, resistance

# --- DATA FETCHING & CACHING ---
@st.cache_data(ttl=120, show_spinner=False)
def fetch_coin_data(symbol, timeframe='1h', limit=100):
    exchange = init_binance()
    try:
        ohlcv = exchange.fetch_ohlcv(f"{symbol}/USDT", timeframe, limit=limit)
        if not ohlcv or len(ohlcv) < 30: return None
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Calculate all indicators
        df['rsi'] = calculate_rsi(df['close'])
        df['adx'] = calculate_adx(df)
        df['vwap'] = calculate_vwap(df)
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean() if len(df) >= 200 else np.nan
        macd, macd_signal, macd_hist = calculate_macd(df['close'])
        df['macd'], df['macd_signal'], df['macd_hist'] = macd, macd_signal, macd_hist
        df['atr'] = calculate_atr(df)
        df['stoch_rsi_k'], df['stoch_rsi_d'] = calculate_stoch_rsi(df['close'])
        return df
    except Exception:
        return None

# --- BTC REGIME FILTER (Crucial for Win Rate) ---
@st.cache_data(ttl=300, show_spinner=False)
def get_btc_regime():
    """Checks BTC 4h trend to filter out bad altcoin longs"""
    df = fetch_coin_data('BTC', '4h', limit=100)
    if df is None: return "NEUTRAL", 0
    price = df['close'].iloc[-1]
    ema20 = df['ema_20'].iloc[-1]
    ema50 = df['ema_50'].iloc[-1]
    if price > ema20 > ema50: return "BULLISH 🟢", price
    elif price < ema20 < ema50: return "BEARISH 🔴", price
    return "NEUTRAL ⚪", price

# --- ANALYSIS LOGIC ---
def analyze_coin(symbol, timeframes_list, price_boost, volume_boost, rsi_thresh, btc_regime):
    results = {}
    raw_dfs = {}
    for tf in timeframes_list:
        limit = 210 if tf in ['1h', '4h', '1d'] else 100
        df = fetch_coin_data(symbol, tf, limit=limit)
        if df is None: continue
        raw_dfs[tf] = df
        
        current_price = df['close'].iloc[-1]
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].iloc[:-1].mean()
        volume_change_pct = ((current_volume - avg_volume) / avg_volume * 100) if avg_volume > 0 else 0
        
        rsi = df['rsi'].iloc[-1]
        adx = df['adx'].iloc[-1]
        vwap = df['vwap'].iloc[-1]
        ema_20, ema_50, ema_200 = df['ema_20'].iloc[-1], df['ema_50'].iloc[-1], df['ema_200'].iloc[-1]
        if pd.isna(ema_200): ema_200 = current_price
        
        near_sr, support, resistance = find_support_resistance(df)
        change_24h = ((current_price - df['close'].iloc[0]) / df['close'].iloc[0] * 100) if len(df) > 1 else 0

        # HIGH WIN-RATE SIGNAL LOGIC
        # LONG: Needs Trend (EMA), Momentum (RSI/Price), Strength (ADX > 20), Institutional Support (Price > VWAP), and BTC Approval
        is_long = (
            change_24h >= price_boost and 
            volume_change_pct >= volume_boost and 
            (pd.isna(rsi) or rsi > rsi_thresh) and
            (current_price > ema_20 and ema_20 > ema_50) and
            (adx > 20) and 
            (current_price > vwap) and
            (btc_regime != "BEARISH 🔴") # Block altcoin longs if BTC is dumping
        )
        
        # SHORT: Inverse logic
        is_short = (
            change_24h <= -price_boost and 
            volume_change_pct >= volume_boost and 
            (pd.isna(rsi) or rsi < (100 - rsi_thresh)) and
            (current_price < ema_20 and ema_20 < ema_50) and
            (adx > 20) and 
            (current_price < vwap) and
            (btc_regime != "BULLISH 🟢") # Block altcoin shorts if BTC is pumping
        )
        
        signal = "LONG 🟢" if is_long else ("SHORT 🔴" if is_short else "NEUTRAL ⚪")

        results[tf] = {
            'price': current_price, 'change_24h': change_24h, 'rsi': rsi, 'adx': adx, 'vwap': vwap,
            'volume_change_pct': volume_change_pct, 'ema_20': ema_20, 'ema_50': ema_50, 'ema_200': ema_200,
            'macd_hist': df['macd_hist'].iloc[-1], 'atr_pct': (df['atr'].iloc[-1] / current_price) * 100,
            'stoch_k': df['stoch_rsi_k'].iloc[-1], 'stoch_d': df['stoch_rsi_d'].iloc[-1],
            'near_sr': near_sr, 'support': support, 'resistance': resistance, 'signal': signal
        }
    return results, raw_dfs

def auto_discover_coins(target_count=10):
    exchange = init_binance()
    st.info("🔍 Auto-discovering viable coins...")
    progress = st.progress(0)
    tickers = exchange.fetch_tickers()
    usdt_tickers = {k: v for k, v in tickers.items() if k.endswith('/USDT') and v.get('quoteVolume')}
    sorted_tickers = sorted(usdt_tickers.items(), key=lambda x: x[1]['quoteVolume'], reverse=True)[:50]
    
    viable_coins = []
    for idx, (symbol, ticker) in enumerate(sorted_tickers):
        coin = symbol.split('/')[0]
        df = fetch_coin_data(coin, '1h', limit=50)
        if df is None or len(df) < 30: continue
            
        rsi = df['rsi'].iloc[-1]
        atr_pct = (df['atr'].iloc[-1] / df['close'].iloc[-1]) * 100
        near_sr, _, _ = find_support_resistance(df)
        
        if ticker['quoteVolume'] < 10_000_000 or atr_pct < 1.5: continue
        if rsi < 25 or rsi > 75 or not near_sr: continue
        
        viable_coins.append({'coin': coin, 'score': (atr_pct * 10) + 50, 'vol': ticker['quoteVolume']})
        if (idx + 1) % 10 == 0: progress.progress((idx + 1) / len(sorted_tickers))
            
    progress.empty()
    viable_coins.sort(key=lambda x: x['score'], reverse=True)
    return [c['coin'] for c in viable_coins[:target_count]]

def get_coins_scan(coins_list, timeframes_list, price_boost, volume_boost, rsi_thresh, btc_regime):
    coins_data = []
    progress_bar = st.progress(0)
    for idx, coin in enumerate(coins_list):
        analysis, raw_dfs = analyze_coin(coin, timeframes_list, price_boost, volume_boost, rsi_thresh, btc_regime)
        if analysis:
            signals = [data.get('signal') for data in analysis.values()]
            dominant_signal = "LONG 🟢" if "LONG 🟢" in signals else ("SHORT 🔴" if "SHORT 🔴" in signals else "NEUTRAL ⚪")
            coins_data.append({
                'symbol': coin, 'price': analysis[timeframes_list[0]]['price'],
                'change_24h': np.mean([data.get('change_24h', 0) for data in analysis.values()]),
                'avg_rsi': np.nanmean([data.get('rsi', 50) for data in analysis.values()]),
                'signal': dominant_signal, 'analysis': analysis, 'raw_dfs': raw_dfs
            })
        progress_bar.progress((idx + 1) / len(coins_list))
    progress_bar.empty()
    return sorted(coins_data, key=lambda x: abs(x['change_24h']), reverse=True)

# --- AI GENERATION (FIXED GEMINI & ENHANCED PROMPT) ---
def generate_ai_advice_with_fallback(symbol, timeframe, metrics, btc_regime, primary_provider="Gemini"):
    def generate_gemini_advice():
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: return "⚠️ GEMINI_API_KEY missing. Check .env.trade (Ensure NO quotes around the key)."
        try:
            client = genai.Client(api_key=api_key)
            prompt = f"""You are an expert quantitative crypto trader. Analyze {symbol}/USDT on {timeframe}:
Price: ${metrics['price']:.4f} | 24h Change: {metrics['change_24h']:.2f}% | Vol Spike: {metrics['volume_change_pct']:.2f}%
RSI: {metrics['rsi']:.1f} | ADX (Trend Strength): {metrics['adx']:.1f} | VWAP: ${metrics['vwap']:.4f}
EMAs: 20(${metrics['ema_20']:.4f}) | 50(${metrics['ema_50']:.4f}) | 200(${metrics['ema_200']:.4f})
MACD Hist: {metrics['macd_hist']:.4f} | ATR Volatility: {metrics['atr_pct']:.2f}%
Support: ${metrics['support']:.4f} | Resistance: ${metrics['resistance']:.4f}
BTC Market Regime: {btc_regime}
Provide: 1. Trade Recommendation (LONG/SHORT/WAIT). 2. Technical Explanation (Focus on ADX trend strength and VWAP position). 3. Targets (Entry, SL, TP 1:2 RR). 4. Primary Risk."""
            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config=types.GenerateContentConfig(automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)))
            return response.text
        except Exception as e: return f"Error (Gemini): {str(e)}"

    def generate_groq_advice():
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key: return "⚠️ GROQ_API_KEY missing."
        try:
            client = GroqClient(api_key=api_key)
            prompt = f"Analyze {symbol}/USDT ({timeframe}). Price: ${metrics['price']:.4f}, RSI: {metrics['rsi']:.1f}, ADX: {metrics['adx']:.1f}, VWAP: ${metrics['vwap']:.4f}, BTC Regime: {btc_regime}. Give LONG/SHORT/WAIT advice, 3 bullet points, and 1:2 RR targets."
            
            # ADD max_tokens=800 to stay under the 1000 OTPM limit
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}], 
                model=GROQ_MODEL,
                max_tokens=800
            )
            return response.choices[0].message.content
        except Exception as e: return f"Error (Groq): {str(e)}"

    def generate_airforce_advice():
        api_key = os.getenv("AIRFORCE_API_KEY")
        if not api_key: return "⚠️ AIRFORCE_API_KEY missing."
        try:
            client = OpenAI(api_key=api_key, base_url="https://api.airforce/v1")
            prompt = f"Analyze {symbol}/USDT ({timeframe}). Price: ${metrics['price']:.4f}, RSI: {metrics['rsi']:.1f}, ADX: {metrics['adx']:.1f}, VWAP: ${metrics['vwap']:.4f}, BTC Regime: {btc_regime}. Give LONG/SHORT/WAIT advice, 3 bullet points, and 1:2 RR targets."
            return client.chat.completions.create(model=AIRFORCE_MODEL, messages=[{"role": "user", "content": prompt}]).choices[0].message.content
        except Exception as e: return f"Error (Apiforce): {str(e)}"

    providers = {"Gemini": generate_gemini_advice, "Groq": generate_groq_advice, "Apiforce": generate_airforce_advice}
    fallback_order = [primary_provider] + [p for p in providers if p != primary_provider]
    for idx, name in enumerate(fallback_order):
        advice = providers[name]()
        if "Error" not in advice and "missing" not in advice.lower():
            return f"**📊 Analysis by: {name if idx == 0 else name + ' (Fallback)'}**\n\n{advice}"
        if idx < len(fallback_order) - 1: st.warning(f"{name} failed ({advice}), attempting fallback...")
    return f"**⚠️ All providers failed.**\n\n{advice}"

def plot_interactive_chart(df, symbol, timeframe):
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=[0.30, 0.10, 0.15, 0.15, 0.15, 0.15],
        subplot_titles=(f"{symbol}/USDT ({timeframe.upper()}) Price & EMAs", "Volume", "RSI & ADX", "MACD", "Stoch RSI", "VWAP vs Price"))
    
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema_20'], line=dict(color='orange', width=1), name="EMA 20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema_50'], line=dict(color='cyan', width=1), name="EMA 50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['vwap'], line=dict(color='yellow', width=1.5, dash='dash'), name="VWAP"), row=1, col=1)
    
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], name="Volume", marker_color='rgba(128,128,128,0.4)'), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi'], line=dict(color='purple', width=1.5), name="RSI"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['adx'], line=dict(color='gold', width=1.5), name="ADX"), row=3, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="red", row=3, col=1) # ADX Trend Threshold
    
    colors = ['#00E676' if val >= 0 else '#FF5252' for val in df['macd_hist']]
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['macd_hist'], marker_color=colors, name="MACD Hist"), row=4, col=1)
    
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['stoch_rsi_k'], line=dict(color='#00E5FF', width=1), name="Stoch %K"), row=5, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['stoch_rsi_d'], line=dict(color='#FF9100', width=1), name="Stoch %D"), row=5, col=1)

    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['close'], line=dict(color='white', width=1), name="Close"), row=6, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['vwap'], line=dict(color='yellow', width=2), name="VWAP"), row=6, col=1)

    fig.update_layout(height=900, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# --- MAIN EXECUTION ---
try:
    # 1. Check BTC Regime First
    btc_regime, btc_price = get_btc_regime()
    st.markdown(f"### 🌍 Global Market Context: Bitcoin (4h) is **{btc_regime}** @ ${btc_price:,.2f}")
    if "BEARISH" in btc_regime:
        st.warning("⚠️ **BTC is in a confirmed downtrend.** Altcoin LONG signals are automatically blocked to protect your win rate.")
    elif "BULLISH" in btc_regime:
        st.success("✅ **BTC is in a confirmed uptrend.** Altcoin SHORT signals are automatically blocked.")
    st.divider()

    if scan_mode == "🔍 Auto-Discover Viable Coins":
        coins_list = auto_discover_coins(target_count=10)
        if not coins_list: st.error("Auto-discovery failed."); st.stop()
    else:
        coins_list = [c.strip().upper() for c in coin_input.split(',') if c.strip()]
        
    if not coins_list or not timeframes: st.error("Select at least one coin and one timeframe."); st.stop()

    coins_analysis = get_coins_scan(coins_list, timeframes, price_boost_pct, volume_boost_pct, rsi_threshold, btc_regime)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🟢 Long Signals", sum(1 for c in coins_analysis if "LONG" in c['signal']))
    col2.metric("🔴 Short Signals", sum(1 for c in coins_analysis if "SHORT" in c['signal']))
    col3.metric("⚪ Neutral", sum(1 for c in coins_analysis if "NEUTRAL" in c['signal']))
    col4.metric("📊 Total Scanned", len(coins_analysis))
    st.divider()

    tab1, tab2 = st.tabs(["📊 Scan Table", "🤖 Deep Analysis & AI Advice"])
    
    with tab1:
        table_rows = [{'Symbol': c['symbol'], 'Price': f"{c['price']:.4f}" if c['price'] < 1 else f"{c['price']:.2f}", 
                       '24h %': f"{c['change_24h']:.2f}%", 'Avg RSI': f"{c['avg_rsi']:.1f}", 'Signal': c['signal']} for c in coins_analysis]
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    with tab2:
        selected_coin = st.selectbox("Select a coin for chart & AI evaluation:", options=[c['symbol'] for c in coins_analysis])
        if selected_coin:
            coin_data = next(c for c in coins_analysis if c['symbol'] == selected_coin)
            selected_tf = st.radio("Chart Timeframe", options=timeframes, horizontal=True)
            
            if selected_tf in coin_data['raw_dfs']:
                raw_df = coin_data['raw_dfs'][selected_tf]
                metrics = coin_data['analysis'].get(selected_tf, {})
                
                fig = plot_interactive_chart(raw_df, selected_coin, selected_tf)
                st.plotly_chart(fig, use_container_width=True)
                st.divider()
                
                st.subheader("🤖 AI Trade Analysis (Context-Aware)")
                provider_name = "Gemini" if ai_provider == "Gemini (Default)" else ("Groq" if ai_provider == "Groq (Fast)" else "Apiforce")
                
                if st.button(f"Generate AI Advice for {selected_coin} ({selected_tf})", type="primary"):
                    with st.spinner(f"Analyzing market structure with {provider_name}..."):
                        ai_response = generate_ai_advice_with_fallback(selected_coin, selected_tf, metrics, btc_regime, primary_provider=provider_name)
                        st.markdown(ai_response)
                        
except Exception as e:
    st.error(f"Execution Error: {str(e)}")