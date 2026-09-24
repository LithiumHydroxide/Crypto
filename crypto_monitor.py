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
load_dotenv(".env.trade")

# FIXED: Updated to currently valid, working model names
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
AIRFORCE_MODEL = os.getenv("AIRFORCE_MODEL", "claude-3-haiku")

# Page config
st.set_page_config(page_title="Crypto Monitor & AI Trade Advisor", layout="wide", initial_sidebar_state="expanded")
st.title("🚀 Crypto Scanner & AI Advisor (Long / Short)")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    scan_mode = st.radio("Scan Mode", ["🔍 Auto-Discover Viable Coins", "📝 Manual Coin List"], index=0)
    
    if scan_mode == "📝 Manual Coin List":
        coin_input = st.text_area(
            "Coins to Monitor (comma-separated)",
            value="BTC,ETH,SOL,BNB,XRP,ADA,AVAX,LINK,DOGE,KAS,NEAR,RUNE,SUI,THETA,ATOM,ZCASH",
            height=90,
            help="Enter symbols without USDT."
        )
    else:
        st.info("🔍 Auto-Scanner will find the top 5-10 coins based on Volume, RSI, S/R, and Volatility.")
        coin_input = ""

    timeframes = st.multiselect(
        "Timeframes to Analyze",
        ["1m", "5m", "15m", "1h", "4h", "1d"],
        default=["1h", "4h"]
    )
    
    st.subheader("Signal Thresholds")
    price_boost_pct = st.slider("Price Threshold %", 0.0, 50.0, 3.0, 0.5)
    volume_boost_pct = st.slider("Volume Spike %", 0.0, 200.0, 40.0, 10.0)
    rsi_threshold = st.slider("RSI Boundary (Upper / Lower)", 50, 80, 55, 1)
    
    st.subheader("💾 Options")
    enable_logging = st.checkbox("Log Scan Results to CSV", value=True)
    
    st.subheader("🤖 AI Advisor")
    ai_provider = st.radio(
        "Select AI Provider for Trade Analysis",
        ["Gemini (Default)", "Groq (Fast)", "Apiforce (DeepSeek V3)"],
        help="Gemini for detailed analysis, Groq/Apiforce for faster responses. Falls back automatically if unavailable."
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

# PERFORMANCE FIX: Added caching. Data is cached for 120 seconds. 
# Moving sliders will now be instant instead of re-fetching from Binance.
@st.cache_data(ttl=120, show_spinner=False)
def fetch_coin_data(symbol, timeframe='1h', limit=100):
    exchange = init_binance()
    try:
        ohlcv = exchange.fetch_ohlcv(f"{symbol}/USDT", timeframe, limit=limit)
        if not ohlcv or len(ohlcv) < 30:
            return None

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Indicators
        df['rsi'] = calculate_rsi(df['close'])
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean() if len(df) >= 200 else np.nan
        
        macd, macd_signal, macd_hist = calculate_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        df['atr'] = calculate_atr(df)
        stoch_k, stoch_d = calculate_stoch_rsi(df['close'])
        df['stoch_rsi_k'] = stoch_k
        df['stoch_rsi_d'] = stoch_d
        
        return df
    except Exception:
        return None

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_stoch_rsi(series, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):
    rsi = calculate_rsi(series, rsi_period)
    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min)
    stoch_rsi = stoch_rsi.fillna(0.5)
    stoch_rsi_k = stoch_rsi.rolling(k_smooth).mean() * 100
    stoch_rsi_d = stoch_rsi_k.rolling(d_smooth).mean()
    return stoch_rsi_k, stoch_rsi_d

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

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
    
    near_sr = (dist_res <= tolerance) or (dist_sup <= tolerance)
    return near_sr, dist_res, dist_sup, support, resistance

def analyze_coin(symbol, timeframes_list, price_boost, volume_boost, rsi_thresh):
    results = {}
    raw_dfs = {}
    for tf in timeframes_list:
        # Fetch with enough data for 200 EMA if possible, else fallback
        limit = 210 if tf in ['1h', '4h', '1d'] else 100
        df = fetch_coin_data(symbol, tf, limit=limit)
        if df is None:
            continue
        raw_dfs[tf] = df
        
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-25] if len(df) > 25 else df['close'].iloc[0]
        price_change_pct = ((current_price - prev_price) / prev_price * 100)
        
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].iloc[:-1].mean()
        volume_change_pct = ((current_volume - avg_volume) / avg_volume * 100) if avg_volume > 0 else 0
        
        rsi = df['rsi'].iloc[-1]
        ema_20 = df['ema_20'].iloc[-1]
        ema_50 = df['ema_50'].iloc[-1]
        ema_200 = df['ema_200'].iloc[-1] if not pd.isna(df['ema_200'].iloc[-1]) else current_price
        macd_hist = df['macd_hist'].iloc[-1]
        
        atr = df['atr'].iloc[-1]
        atr_pct = (atr / current_price) * 100
        stoch_k = df['stoch_rsi_k'].iloc[-1]
        stoch_d = df['stoch_rsi_d'].iloc[-1]
        
        near_sr, dist_res, dist_sup, support, resistance = find_support_resistance(df)
        
        change_24h = ((current_price - df['close'].iloc[0]) / df['close'].iloc[0] * 100) if len(df) > 1 else 0

        is_long = (
            change_24h >= price_boost and 
            volume_change_pct >= volume_boost and 
            (pd.isna(rsi) or rsi > rsi_thresh) and
            (current_price > ema_20 and ema_20 > ema_50 and current_price > ema_200)
        )
        is_short = (
            change_24h <= -price_boost and 
            volume_change_pct >= volume_boost and 
            (pd.isna(rsi) or rsi < (100 - rsi_thresh)) and
            (current_price < ema_20 and ema_20 < ema_50 and current_price < ema_200)
        )
        
        signal = "LONG 🟢" if is_long else ("SHORT 🔴" if is_short else "NEUTRAL ⚪")

        results[tf] = {
            'price': current_price,
            'price_change_pct': price_change_pct,
            'change_24h': change_24h,
            'rsi': rsi,
            'volume': current_volume,
            'volume_change_pct': volume_change_pct,
            'ema_20': ema_20, 'ema_50': ema_50, 'ema_200': ema_200,
            'macd_hist': macd_hist,
            'atr_pct': atr_pct,
            'stoch_k': stoch_k, 'stoch_d': stoch_d,
            'near_sr': near_sr, 'support': support, 'resistance': resistance,
            'signal': signal
        }
    return results, raw_dfs

def auto_discover_coins(target_count=10):
    exchange = init_binance()
    st.info("🔍 Auto-discovering viable coins... (This may take 10-20 seconds on first run)")
    progress = st.progress(0)
    
    tickers = exchange.fetch_tickers()
    usdt_tickers = {k: v for k, v in tickers.items() if k.endswith('/USDT') and v.get('quoteVolume')}
    
    # Sort by 24h quote volume and take top 50 to avoid API timeouts
    sorted_tickers = sorted(usdt_tickers.items(), key=lambda x: x[1]['quoteVolume'], reverse=True)[:50]
    
    viable_coins = []
    total = len(sorted_tickers)
    
    for idx, (symbol, ticker) in enumerate(sorted_tickers):
        coin = symbol.split('/')[0]
        # PERFORMANCE FIX: Only fetch 50 candles for the initial scan. 
        # 50 is enough for RSI(14) and ATR(14), cuts API load by 75%.
        df = fetch_coin_data(coin, '1h', limit=50)
        
        if df is None or len(df) < 30:
            continue
            
        current_price = df['close'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        atr = df['atr'].iloc[-1]
        atr_pct = (atr / current_price) * 100
        near_sr, _, _, _, _ = find_support_resistance(df)
        quote_vol = ticker['quoteVolume']
        
        if quote_vol < 10_000_000: continue
        if atr_pct < 1.5: continue
        if rsi < 25 or rsi > 75: continue
        if not near_sr: continue
        
        sr_score = 50 
        rsi_score = 100 - abs(rsi - 50) 
        score = (atr_pct * 10) + sr_score + (rsi_score * 0.5)
        
        viable_coins.append({'coin': coin, 'score': score, 'vol': quote_vol})
        
        if (idx + 1) % 10 == 0:
            progress.progress((idx + 1) / total)
            
    progress.empty()
    
    viable_coins.sort(key=lambda x: x['score'], reverse=True)
    top_coins = [c['coin'] for c in viable_coins[:target_count]]
    
    if not top_coins:
        st.warning("Strict criteria yielded 0 results. Falling back to top coins by Volume...")
        fallback = sorted(viable_coins, key=lambda x: x['vol'], reverse=True)
        top_coins = [c['coin'] for c in fallback[:target_count]]
        
    return top_coins

def get_coins_scan(coins_list, timeframes_list, price_boost, volume_boost, rsi_thresh):
    coins_data = []
    progress_bar = st.progress(0)
    for idx, coin in enumerate(coins_list):
        analysis, raw_dfs = analyze_coin(coin, timeframes_list, price_boost, volume_boost, rsi_thresh)
        if analysis:
            signals = [data.get('signal') for data in analysis.values()]
            dominant_signal = "LONG 🟢" if "LONG 🟢" in signals else ("SHORT 🔴" if "SHORT 🔴" in signals else "NEUTRAL ⚪")
            avg_rsi = np.nanmean([data.get('rsi', 50) for data in analysis.values()])
            avg_change_24h = np.mean([data.get('change_24h', 0) for data in analysis.values()])
            coins_data.append({
                'symbol': coin,
                'price': analysis[timeframes_list[0]]['price'] if timeframes_list else 0,
                'change_24h': avg_change_24h,
                'avg_rsi': avg_rsi,
                'signal': dominant_signal,
                'analysis': analysis,
                'raw_dfs': raw_dfs
            })
        progress_bar.progress((idx + 1) / len(coins_list))
    progress_bar.empty()
    return sorted(coins_data, key=lambda x: abs(x['change_24h']), reverse=True)

def generate_ai_advice_with_fallback(symbol, timeframe, metrics, primary_provider="Gemini"):
    def generate_gemini_advice():
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key: return "⚠️ GEMINI_API_KEY missing in .env.trade."
        try:
            client = genai.Client(api_key=api_key)
            prompt = f"""You are an expert quantitative crypto trader. Analyze {symbol}/USDT on {timeframe}:
Price: ${metrics['price']:.4f} | 24h Change: {metrics['change_24h']:.2f}% | Vol Spike: {metrics['volume_change_pct']:.2f}%
RSI: {metrics['rsi']:.1f} | Stoch RSI (%K/%D): {metrics['stoch_k']:.1f}/{metrics['stoch_d']:.1f}
EMAs: 20(${metrics['ema_20']:.4f}) | 50(${metrics['ema_50']:.4f}) | 200(${metrics['ema_200']:.4f})
MACD Hist: {metrics['macd_hist']:.4f} | ATR Volatility: {metrics['atr_pct']:.2f}%
Support: ${metrics['support']:.4f} | Resistance: ${metrics['resistance']:.4f}
Provide: 1. Trade Recommendation (LONG/SHORT/WAIT). 2. Technical Explanation (3 bullets). 3. Targets (Entry, SL, TP 1:2 RR). 4. Primary Risk."""
            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config=types.GenerateContentConfig(automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)))
            return response.text
        except Exception as e: return f"Error (Gemini): {str(e)}"

    def generate_groq_advice():
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key: return "⚠️ GROQ_API_KEY missing in .env.trade."
        try:
            client = GroqClient(api_key=api_key)
            prompt = f"""You are an expert quantitative crypto trader. Analyze {symbol}/USDT on {timeframe}:
Price: ${metrics['price']:.4f} | 24h Change: {metrics['change_24h']:.2f}% | Vol Spike: {metrics['volume_change_pct']:.2f}%
RSI: {metrics['rsi']:.1f} | Stoch RSI (%K/%D): {metrics['stoch_k']:.1f}/{metrics['stoch_d']:.1f}
EMAs: 20(${metrics['ema_20']:.4f}) | 50(${metrics['ema_50']:.4f}) | 200(${metrics['ema_200']:.4f})
MACD Hist: {metrics['macd_hist']:.4f} | ATR Volatility: {metrics['atr_pct']:.2f}%
Support: ${metrics['support']:.4f} | Resistance: ${metrics['resistance']:.4f}
Provide: 1. Trade Recommendation (LONG/SHORT/WAIT). 2. Technical Explanation (3 bullets). 3. Targets (Entry, SL, TP 1:2 RR). 4. Primary Risk."""
            message = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=GROQ_MODEL)
            return message.choices[0].message.content
        except Exception as e: return f"Error (Groq): {str(e)}"

    def generate_airforce_advice():
        api_key = os.getenv("AIRFORCE_API_KEY")
        if not api_key: return "⚠️ AIRFORCE_API_KEY missing in .env.trade."
        try:
            client = OpenAI(api_key=api_key, base_url="https://api.airforce/v1")
            prompt = f"""You are an expert quantitative crypto trader. Analyze {symbol}/USDT on {timeframe}:
Price: ${metrics['price']:.4f} | 24h Change: {metrics['change_24h']:.2f}% | Vol Spike: {metrics['volume_change_pct']:.2f}%
RSI: {metrics['rsi']:.1f} | Stoch RSI (%K/%D): {metrics['stoch_k']:.1f}/{metrics['stoch_d']:.1f}
EMAs: 20(${metrics['ema_20']:.4f}) | 50(${metrics['ema_50']:.4f}) | 200(${metrics['ema_200']:.4f})
MACD Hist: {metrics['macd_hist']:.4f} | ATR Volatility: {metrics['atr_pct']:.2f}%
Support: ${metrics['support']:.4f} | Resistance: ${metrics['resistance']:.4f}
Provide: 1. Trade Recommendation (LONG/SHORT/WAIT). 2. Technical Explanation (3 bullets). 3. Targets (Entry, SL, TP 1:2 RR). 4. Primary Risk."""
            message = client.chat.completions.create(model=AIRFORCE_MODEL, messages=[{"role": "user", "content": prompt}])
            return message.choices[0].message.content
        except Exception as e: return f"Error (Apiforce): {str(e)}"

    providers = {"Gemini": generate_gemini_advice, "Groq": generate_groq_advice, "Apiforce": generate_airforce_advice}
    if primary_provider not in providers: return "**⚠️ Unsupported AI provider.**"
    
    fallback_order = [primary_provider] + [p for p in providers if p != primary_provider]
    for idx, name in enumerate(fallback_order):
        advice = providers[name]()
        if "Error" not in advice and "missing" not in advice.lower():
            label = name if idx == 0 else f"{name} (Fallback)"
            return f"**📊 Analysis by: {label}**\n\n{advice}"
        if idx < len(fallback_order) - 1: 
            st.warning(f"{name} failed ({advice}), attempting fallback...")
    return f"**⚠️ All providers failed.**\n\n{advice}"

def plot_interactive_chart(df, symbol, timeframe):
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.35, 0.15, 0.15, 0.15, 0.20],
        subplot_titles=(f"{symbol}/USDT ({timeframe.upper()}) Price & EMAs", "Volume", "RSI", "MACD", "Stochastic RSI")
    )
    
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema_20'], line=dict(color='orange', width=1.2), name="EMA 20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema_50'], line=dict(color='cyan', width=1.2), name="EMA 50"), row=1, col=1)
    if 'ema_200' in df.columns and not df['ema_200'].isna().all():
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema_200'], line=dict(color='magenta', width=1.5, dash='dash'), name="EMA 200"), row=1, col=1)
    
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], name="Volume", marker_color='rgba(128,128,128,0.4)'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi'], line=dict(color='purple', width=1.5), name="RSI"), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
    
    colors = ['#00E676' if val >= 0 else '#FF5252' for val in df['macd_hist']]
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['macd_hist'], marker_color=colors, name="MACD Hist"), row=4, col=1)
    
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['stoch_rsi_k'], line=dict(color='#00E5FF', width=1.5), name="Stoch RSI %K"), row=5, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['stoch_rsi_d'], line=dict(color='#FF9100', width=1.5), name="Stoch RSI %D"), row=5, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="red", row=5, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="green", row=5, col=1)

    fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# Main App Execution
try:
    if scan_mode == "🔍 Auto-Discover Viable Coins":
        coins_list = auto_discover_coins(target_count=10)
        if not coins_list:
            st.error("Auto-discovery failed to find viable coins. Check connection or try Manual mode.")
            st.stop()
    else:
        coins_list = [c.strip().upper() for c in coin_input.split(',') if c.strip()]
        
    if not coins_list or not timeframes:
        st.error("Select at least one coin and one timeframe.")
        st.stop()

    coins_analysis = get_coins_scan(coins_list, timeframes, price_boost_pct, volume_boost_pct, rsi_threshold)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🟢 Long Signals", sum(1 for c in coins_analysis if "LONG" in c['signal']))
    col2.metric("🔴 Short Signals", sum(1 for c in coins_analysis if "SHORT" in c['signal']))
    col3.metric("⚪ Neutral", sum(1 for c in coins_analysis if "NEUTRAL" in c['signal']))
    col4.metric("📊 Total Scanned", len(coins_analysis))
    st.divider()

    tab1, tab2 = st.tabs(["📊 Scan Table", "🤖 Deep Analysis & AI Advice"])
    
    with tab1:
        table_rows = []
        for c in coins_analysis:
            table_rows.append({
                'Symbol': c['symbol'],
                'Price (USDT)': f"{c['price']:.4f}" if c['price'] < 1 else f"{c['price']:.2f}",
                '24h Change %': f"{c['change_24h']:.2f}%",
                'Avg RSI': f"{c['avg_rsi']:.1f}",
                'Signal': c['signal']
            })
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
                
                st.subheader("🤖 AI Trade Analysis")
                provider_name = "Gemini" if ai_provider == "Gemini (Default)" else ("Groq" if ai_provider == "Groq (Fast)" else "Apiforce")
                
                if st.button(f"Generate AI Advice for {selected_coin} ({selected_tf})", type="primary"):
                    with st.spinner(f"Analyzing market structure with {provider_name}..."):
                        ai_response = generate_ai_advice_with_fallback(
                            selected_coin, selected_tf, metrics, primary_provider=provider_name
                        )
                        st.markdown(ai_response)
                        
except Exception as e:
    st.error(f"Execution Error: {str(e)}")