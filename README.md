# NSE/BSE Stock Fair Value & Swing Trading Analyzer

A comprehensive Streamlit web application for analyzing Indian stocks listed on NSE and BSE, with fair value calculations, technical indicators, and swing trading signals for profitable short-term trades (1 day to 1 month).

## 🚀 Features

### 📊 Swing Trading Analysis (1 Day - 1 Month)
Perfect for day traders and swing traders looking for short-term profitable opportunities:

**Customizable Technical Indicators:**
Choose from 20+ technical indicators and display only what you need:

- **Moving Averages:** SMA 5, 10, 20, 50 | EMA 9, 21, 50
- **Oscillators:** RSI, MACD, Stochastic, CCI, Williams %R, ADX
- **Volatility Indicators:** Bollinger Bands, ATR, Keltner Channels
- **Volume Indicators:** Volume, OBV (On-Balance Volume), Volume SMA
- **Trend Indicators:** ADX, Supertrend, Parabolic SAR

**Smart Trading Signals:**
- ✅ **Automated Entry/Exit:** Clear buy/sell recommendations
- 🎯 **Multiple Targets:** Three target levels with risk-reward ratios (1.5:1, 2.5:1, 3.5:1)
- 🛡️ **Risk Management:** ATR-based stop loss calculations
- 📍 **Support & Resistance:** Automatically identified key price levels
- 📈 **Signal Score:** Composite score from all indicators (-10 to +10)
- 📉 **Volume Analysis:** Confirms trend strength

**Visual Charts:**
- Candlestick charts with all indicators
- MACD histogram and signal lines
- RSI with overbought/oversold zones
- Stochastic oscillator
- Target and stop loss levels marked on chart

### 💰 Fundamental Analysis (Long-term Investment)
- **Multiple Valuation Models:** DCF, P/E, P/B ratio methods
- **Fair Value Calculation:** Average of multiple valuation approaches
- **Buy/Sell Recommendations:** Based on discount/premium to fair value
- **Key Metrics:** P/E, P/B, PEG, ROE, Debt-to-Equity, Dividend Yield
- **Price History:** Interactive charts with fair value overlay

## 📋 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Steps

1. **Download the files** to your local directory

2. **Install required packages:**
```bash
pip install -r requirements.txt
```

## 🎯 Running the Application

1. **Navigate to the project directory:**
```bash
cd path/to/your/project
```

2. **Run the Streamlit app:**
```bash
streamlit run stock_analyzer.py
```

3. **Access the app:**
   - The app will automatically open in your default browser
   - If not, go to: `http://localhost:8501`

## 📖 How to Use

### 1. Select Analysis Type

Choose between two analysis modes:

- **Swing Trading (1 Day - 1 Month):** For short-term technical trading
- **Fundamental (Long-term):** For long-term value investing

### 2. Enter Stock Symbol

- **NSE stocks:** Add `.NS` suffix (e.g., `RELIANCE.NS`, `TCS.NS`)
- **BSE stocks:** Add `.BO` suffix (e.g., `RELIANCE.BO`, `TCS.BO`)

### 3. Select Timeframe

Choose from: 1 Week, 1 Month, 3 Months, 6 Months, or 1 Year

### 4. Customize Indicators (Swing Trading Mode Only)

In the sidebar, select which indicators you want to display:

- **Moving Averages:** Choose from SMA 5/10/20/50 and EMA 9/21/50
- **Oscillators & Momentum:** RSI, MACD, Stochastic, CCI, Williams %R, ADX
- **Volatility & Bands:** Bollinger Bands, ATR, Keltner Channels
- **Volume Indicators:** Volume bars, OBV, Volume SMA
- **Trend Indicators:** ADX, Supertrend, Parabolic SAR

**Tip:** Start with the defaults (EMA 9/21, SMA 20, RSI, MACD, Stochastic, Bollinger Bands, ATR) and add more as needed.

### 5. Analyze Results

#### For Swing Trading:
- **Signal Score:** -10 to +10 score indicating strength
- **Entry Price:** Current market price
- **Stop Loss:** Where to exit if trade goes against you
- **Targets:** Three target levels with risk-reward ratios
- **Support/Resistance:** Key levels to watch
- **Technical Signals:** All indicator signals explained

#### For Fundamental:
- **Fair Value:** Average from multiple valuation models
- **Upside/Downside:** Percentage difference from current price
- **Recommendation:** Strong Buy to Strong Sell based on valuation

## 📊 Understanding Swing Trading Signals

### Signal Recommendations:
- **STRONG BUY (Score ≥5):** Multiple bullish indicators aligned
- **BUY (Score 2-4):** Bullish bias, good entry opportunity
- **HOLD (Score -2 to 2):** Neutral, wait for clearer signals
- **SELL (Score -5 to -2):** Bearish bias, consider exiting
- **STRONG SELL (Score ≤-5):** Multiple bearish indicators aligned

### Key Technical Signals:
✅ **Buy Signals:**
- RSI below 30 (oversold)
- MACD bullish crossover
- EMA Golden Cross (9 crosses above 21)
- Price touches lower Bollinger Band
- Stochastic below 20
- Price above 20 SMA (uptrend)
- High volume confirms moves

❌ **Sell Signals:**
- RSI above 70 (overbought)
- MACD bearish crossover
- EMA Death Cross (9 crosses below 21)
- Price touches upper Bollinger Band
- Stochastic above 80
- Price below 20 SMA (downtrend)

### Risk Management:
1. **Stop Loss:** Set automatically at 2x ATR below entry (or nearest support)
2. **Target 1:** 1.5:1 risk-reward ratio
3. **Target 2:** 2.5:1 risk-reward ratio
4. **Target 3:** 3.5:1 risk-reward ratio

**Example:**
- Entry: ₹1000
- Stop Loss: ₹950 (5% risk)
- Target 1: ₹1075 (7.5% reward, 1.5:1 R:R)
- Target 2: ₹1125 (12.5% reward, 2.5:1 R:R)
- Target 3: ₹1175 (17.5% reward, 3.5:1 R:R)

## 📈 Popular Stock Symbols

### NSE Large Caps
- `RELIANCE.NS` - Reliance Industries
- `TCS.NS` - Tata Consultancy Services
- `INFY.NS` - Infosys
- `HDFCBANK.NS` - HDFC Bank
- `ICICIBANK.NS` - ICICI Bank
- `BHARTIARTL.NS` - Bharti Airtel
- `ITC.NS` - ITC Limited
- `SBIN.NS` - State Bank of India
- `HINDUNILVR.NS` - Hindustan Unilever
- `KOTAKBANK.NS` - Kotak Mahindra Bank

### NSE Mid/Small Caps (Good for Swing Trading)
- `ADANIPORTS.NS` - Adani Ports
- `TATAMOTORS.NS` - Tata Motors
- `SAIL.NS` - SAIL
- `IDEA.NS` - Vodafone Idea
- `YESBANK.NS` - Yes Bank

## 🎓 Swing Trading Tips

1. **Timeframe Selection:**
   - 1 Week: For very short-term day trades
   - 1 Month: Optimal for swing trading (recommended)
   - 3 Months: For medium-term position trading

2. **Signal Confirmation:**
   - Don't rely on just one indicator
   - Wait for multiple signals to align (score ≥3 or ≤-3)
   - Higher signal scores = higher confidence

3. **Risk Management:**
   - Never risk more than 2% of capital per trade
   - Always use stop losses
   - Book partial profits at Target 1, hold rest for higher targets
   - Trail stop loss to entry after Target 1 is hit

4. **Volume Confirmation:**
   - High volume (>1.5x average) confirms trend strength
   - Low volume moves are less reliable

5. **Support/Resistance:**
   - Use these levels to time entries and exits
   - Breakouts above resistance with volume are bullish
   - Breakdowns below support are bearish

## 🛠️ Troubleshooting

### Common Issues

1. **"Unable to fetch stock data" error:**
   - Verify stock symbol is correct
   - Ensure .NS or .BO suffix is added
   - Check internet connection
   - Try a different stock symbol

2. **"N/A" in technical indicators:**
   - Need sufficient data points (minimum 20 days for most indicators)
   - Try selecting a longer timeframe

3. **Slow loading:**
   - First load fetches fresh data and takes time
   - Subsequent loads are faster

## 🌐 Deployment to Streamlit Cloud

1. **Push code to GitHub:**
   - Create a new repository
   - Upload `stock_analyzer.py` and `requirements.txt`

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file as `stock_analyzer.py`
   - Click "Deploy"

## ⚠️ Disclaimer

**IMPORTANT:** This tool is for educational and informational purposes only. It is NOT financial advice.

- Always conduct your own research before making investment decisions
- Consult with a qualified financial advisor
- Past performance does not guarantee future results
- Stock market trading involves substantial risk of loss
- Never invest money you cannot afford to lose
- Technical indicators are not 100% accurate
- Markets can remain irrational longer than you can remain solvent

**The developers and contributors are not responsible for any financial losses incurred from using this tool.**

## 🔧 Technical Stack

- **Streamlit:** Web framework
- **yfinance:** Stock data API (Yahoo Finance)
- **Pandas:** Data manipulation and analysis
- **NumPy:** Numerical computations
- **Plotly:** Interactive charts and visualizations

## 📚 Learning Resources

### Understanding Technical Indicators

**Oscillators (Identify Overbought/Oversold):**
- **RSI:** 0-100 scale. <30 oversold, >70 overbought
- **CCI:** >100 overbought, <-100 oversold, good for trending markets
- **Williams %R:** <-80 oversold, >-20 overbought
- **Stochastic:** <20 oversold, >80 overbought, %K crosses %D for signals

**Trend Indicators:**
- **MACD:** Crossovers indicate trend changes
- **ADX:** >25 indicates strong trend, <20 weak trend
- **Moving Averages:** Price above MA = uptrend, below = downtrend
- **Supertrend:** Dynamic support/resistance that changes with volatility
- **Parabolic SAR:** Dots flip sides when trend reverses

**Volatility Indicators:**
- **Bollinger Bands:** Price touching bands indicates extreme moves
- **ATR:** Higher values = more volatile (use for stop loss sizing)
- **Keltner Channels:** Similar to Bollinger but uses ATR

**Volume Indicators:**
- **OBV:** Rising OBV confirms uptrend, falling confirms downtrend
- **Volume:** High volume confirms price moves

### Indicator Combinations (Best Setups)

**For Swing Trading (1-4 weeks):**
- RSI + MACD + Moving Averages (9/21)
- Bollinger Bands + Stochastic + Volume
- ADX + Supertrend + ATR

**For Day Trading (Intraday):**
- Stochastic + CCI + Volume
- MACD + Williams %R + Short MAs (5/10)
- Bollinger Bands + ATR + Volume

**For Position Trading (1-3 months):**
- Long MAs (20/50) + ADX + OBV
- MACD + RSI + Support/Resistance

To improve your swing trading skills:
- Learn about technical analysis fundamentals
- Study candlestick patterns
- Understand risk management principles
- Practice with paper trading first
- Keep a trading journal
- Study market psychology

## 🤝 Support

For issues or questions:
- Check the troubleshooting section
- Verify all dependencies are installed
- Ensure Python 3.8+
- Try with popular stocks first (RELIANCE.NS, TCS.NS)

---

**Happy Trading! 📈💰**

*Remember: The best traders are risk managers first, profit seekers second.*
