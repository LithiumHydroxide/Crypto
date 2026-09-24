# Crypto Coin Monitor - Setup Guide

A real-time Streamlit app that scans your configured coins on Binance and identifies bullish signals based on price momentum, volume spikes, and RSI indicators.

## 🔐 Security - API Keys Setup

**IMPORTANT:** Never hardcode API keys in your code. This app uses environment variables for security.

### Step 1: Get Your Binance API Keys

1. Go to [Binance API Management](https://www.binance.com/en/account/api-management)
2. Create a new API key (e.g., name it "CryptoMonitor")
3. **Important:** Restrict permissions to:
   - ✅ Enable Reading (for fetching data)
   - ✅ Spot Trading (if you plan to trade later)
   - ✅ Margin Trading (optional)
   - ❌ Disable Withdrawal
   - ❌ Disable IP Restriction (or whitelist your IP)
4. Copy your API Key and Secret Key

### Step 2: Create `.env` File

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your keys:
   ```
   BINANCE_API_KEY=your_actual_api_key_here
   BINANCE_SECRET_KEY=your_actual_secret_key_here
   ```

3. **Make sure `.env` is in `.gitignore`** (never commit this file):
   ```bash
   echo ".env" >> .gitignore
   ```

## 📦 Installation

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Running the App

```bash
streamlit run crypto_monitor.py
```

This opens the app in your browser (usually at `http://localhost:8501`).

## 📊 How It Works

### Configuration (Left Sidebar)

- **Coins to Monitor**: Enter comma-separated symbols (e.g., `BTC,ETH,SOL,BNB`)
- **Timeframes**: Select which candle intervals to analyze (`1m`, `5m`, `15m`, `1h`, `4h`, `1d`)
- **Bullish Thresholds**:
  - **Price Boost %**: Coins must gain this much in 24h
  - **Volume Spike %**: Volume must surge this much vs 7-day average
  - **RSI Threshold**: RSI must exceed this value to confirm bullish momentum

### Output

**Three Views:**
1. **🔥 Bullish Coins**: Only coins meeting ALL bullish criteria
2. **📊 All Coins**: Complete list sorted by 24h gain
3. **📈 Detailed Analysis**: Deep dive into individual coin metrics by timeframe

### Metrics Explained

- **24h Change %**: Price movement over last 24 hours
- **Avg RSI**: Relative Strength Index (50-70 = bullish, >70 = overbought)
- **Volume Spike %**: How much current volume exceeds 7-day average
- **Price Change %**: Momentum over the analyzed period

## 💡 Tips for Using

### Finding Good Coins

**Conservative Settings** (fewer false positives):
```
Price Boost: 8%+
Volume Spike: 80%+
RSI Threshold: 60
Timeframes: 4h, 1d
```

**Aggressive Settings** (catch early breakouts):
```
Price Boost: 2%+
Volume Spike: 30%+
RSI Threshold: 50
Timeframes: 15m, 1h, 4h
```

### Workflow

1. **Load the app** → Scanner runs automatically
2. **Review bullish coins** → Check 4h and 1d timeframes first
3. **Click "Detailed Analysis"** → Drill into individual coins
4. **Refresh page** → Re-scan with latest data
5. **Use charts** → (Optional: add to later) for visual confirmation

## 🔄 Integration with Your Trading Agent

This app is **Stage 1** of your crypto trading agent architecture:
- ✅ Market Scanner (this app)
- 🔜 Rule-based Strategy Engine
- 🔜 Backtesting Module
- 🔜 ML Predictor (XGBoost)
- 🔜 Trade Scoring Engine
- 🔜 LLM Explanation Layer
- 🔜 Paper Trading

**Next Steps:**
1. Use this app daily to build intuition on which coins move predictably
2. Log the outputs to a CSV for later backtesting
3. Once you have patterns, move to Stage 2: build strategy rules
4. Feed these bullish coins into your ML model

## 📝 Example Setup

**Start with default coins (BTC, ETH, SOL, BNB, etc.) on 1h + 4h:**
- Good for swing trading
- Catches medium-term momentum
- Low false signals

**Then create a specialized version for:**
- **Altcoins only**: DOGE, SHIB, MATIC, etc. (more volatile, faster moves)
- **Blue chips**: BTC, ETH, BNB (lower volatility, more reliable)
- **Emerging projects**: LINK, AVAX, ATOM (balance of volatility & trend strength)

## ⚠️ Important Notes

- **Real-time data**: Updates only when you refresh the page
- **API rate limits**: Binance allows ~1200 requests/minute (this app is efficient)
- **Paper trading first**: Use this to plan trades, not to execute automatically yet
- **Risk management**: Always set stop losses before trading

## 🐛 Troubleshooting

**"Failed to connect to Binance"**
- Check your `.env` file has the correct keys
- Verify API key is enabled on Binance
- Check internet connection

**"No bullish coins found"**
- Adjust thresholds (lower them)
- Try different timeframes
- Market may be in consolidation phase

**Slow performance**
- Reduce number of coins
- Switch to longer timeframes (4h, 1d vs 1m, 5m)
- Run during off-peak hours

## 📚 Resources

- [Binance API Docs](https://binance-docs.github.io/apidocs/)
- [CCXT Docs](https://docs.ccxt.com)
- [Streamlit Docs](https://docs.streamlit.io)
- [RSI Indicator Explained](https://www.investopedia.com/terms/r/rsi.asp)

---

**Happy trading! 🚀**
