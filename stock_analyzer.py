import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(page_title="NSE/BSE Stock Analyzer", layout="wide", page_icon="📈")

# Title and description
st.title("📊 NSE/BSE Stock Fair Value & Swing Trading Analyzer")
st.markdown("*Analyze Indian stocks with valuation metrics, technical indicators, and swing trading signals*")

# Sidebar inputs
st.sidebar.header("Stock Selection")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., RELIANCE.NS, TCS.BO)", "RELIANCE.NS").upper()
st.sidebar.markdown("*Use .NS for NSE, .BO for BSE*")

st.sidebar.header("Analysis Settings")
analysis_type = st.sidebar.radio("Analysis Type", ["Swing Trading (1 Day - 1 Month)", "Fundamental (Long-term)"], index=0)
timeframe = st.sidebar.selectbox("Timeframe", ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year"], index=1)

if analysis_type == "Swing Trading (1 Day - 1 Month)":
    st.sidebar.header("📊 Select Indicators")
    
    # Moving Averages
    show_ma = st.sidebar.multiselect(
        "Moving Averages",
        ["SMA 5", "SMA 10", "SMA 20", "EMA 9", "EMA 21", "EMA 50"],
        default=["EMA 9", "EMA 21", "SMA 20"]
    )
    
    # Oscillators
    show_oscillators = st.sidebar.multiselect(
        "Oscillators & Momentum",
        ["RSI", "MACD", "Stochastic", "CCI", "Williams %R", "ADX"],
        default=["RSI", "MACD", "Stochastic"]
    )
    
    # Volatility Indicators
    show_volatility = st.sidebar.multiselect(
        "Volatility & Bands",
        ["Bollinger Bands", "ATR", "Keltner Channels"],
        default=["Bollinger Bands", "ATR"]
    )
    
    # Volume Indicators
    show_volume = st.sidebar.multiselect(
        "Volume Indicators",
        ["Volume", "OBV", "Volume SMA"],
        default=["Volume"]
    )
    
    # Trend Indicators
    show_trend = st.sidebar.multiselect(
        "Trend Indicators",
        ["ADX", "Supertrend", "Parabolic SAR"],
        default=[]
    )
else:
    show_ma = []
    show_oscillators = []
    show_volatility = []
    show_volume = []
    show_trend = []

# Helper functions
def get_stock_data(symbol, period="1mo"):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period=period, interval="1d")
        return stock, info, hist
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None, None

def calculate_technical_indicators(df):
    """Calculate technical indicators for swing trading"""
    # Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # ATR (Average True Range) for volatility
    df['High-Low'] = df['High'] - df['Low']
    df['High-PrevClose'] = abs(df['High'] - df['Close'].shift(1))
    df['Low-PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # Keltner Channels
    df['KC_Middle'] = df['EMA_21']
    df['KC_Upper'] = df['KC_Middle'] + (2 * df['ATR'])
    df['KC_Lower'] = df['KC_Middle'] - (2 * df['ATR'])
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # OBV (On-Balance Volume)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stoch_%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['Stoch_%D'] = df['Stoch_%K'].rolling(window=3).mean()
    
    # CCI (Commodity Channel Index)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['CCI'] = (tp - tp.rolling(window=20).mean()) / (0.015 * tp.rolling(window=20).std())
    
    # Williams %R
    hh = df['High'].rolling(window=14).max()
    ll = df['Low'].rolling(window=14).min()
    df['Williams_%R'] = -100 * ((hh - df['Close']) / (hh - ll))
    
    # ADX (Average Directional Index)
    df['High_Diff'] = df['High'].diff()
    df['Low_Diff'] = df['Low'].diff()
    df['+DM'] = np.where((df['High_Diff'] > df['Low_Diff']) & (df['High_Diff'] > 0), df['High_Diff'], 0)
    df['-DM'] = np.where((df['Low_Diff'] > df['High_Diff']) & (df['Low_Diff'] > 0), df['Low_Diff'], 0)
    
    df['+DI'] = 100 * (df['+DM'].rolling(window=14).mean() / df['ATR'])
    df['-DI'] = 100 * (df['-DM'].rolling(window=14).mean() / df['ATR'])
    df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df['ADX'] = df['DX'].rolling(window=14).mean()
    
    # Supertrend
    df['Supertrend_Basic_Upper'] = ((df['High'] + df['Low']) / 2) + (3 * df['ATR'])
    df['Supertrend_Basic_Lower'] = ((df['High'] + df['Low']) / 2) - (3 * df['ATR'])
    
    # Initialize Supertrend
    df['Supertrend'] = 0.0
    df['Supertrend_Direction'] = 1
    
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Supertrend_Basic_Upper'].iloc[i-1]:
            df.loc[df.index[i], 'Supertrend'] = df['Supertrend_Basic_Lower'].iloc[i]
            df.loc[df.index[i], 'Supertrend_Direction'] = 1
        elif df['Close'].iloc[i] < df['Supertrend_Basic_Lower'].iloc[i-1]:
            df.loc[df.index[i], 'Supertrend'] = df['Supertrend_Basic_Upper'].iloc[i]
            df.loc[df.index[i], 'Supertrend_Direction'] = -1
        else:
            if df['Supertrend_Direction'].iloc[i-1] == 1:
                df.loc[df.index[i], 'Supertrend'] = df['Supertrend_Basic_Lower'].iloc[i]
                df.loc[df.index[i], 'Supertrend_Direction'] = 1
            else:
                df.loc[df.index[i], 'Supertrend'] = df['Supertrend_Basic_Upper'].iloc[i]
                df.loc[df.index[i], 'Supertrend_Direction'] = -1
    
    # Parabolic SAR
    df['SAR'] = df['Close'].copy()
    df['EP'] = df['Close'].copy()
    df['AF'] = 0.02
    
    for i in range(2, len(df)):
        if df['Supertrend_Direction'].iloc[i-1] == 1:  # Uptrend
            df.loc[df.index[i], 'SAR'] = df['SAR'].iloc[i-1] + df['AF'].iloc[i-1] * (df['EP'].iloc[i-1] - df['SAR'].iloc[i-1])
            if df['High'].iloc[i] > df['EP'].iloc[i-1]:
                df.loc[df.index[i], 'EP'] = df['High'].iloc[i]
                df.loc[df.index[i], 'AF'] = min(0.2, df['AF'].iloc[i-1] + 0.02)
            else:
                df.loc[df.index[i], 'EP'] = df['EP'].iloc[i-1]
                df.loc[df.index[i], 'AF'] = df['AF'].iloc[i-1]
        else:  # Downtrend
            df.loc[df.index[i], 'SAR'] = df['SAR'].iloc[i-1] - df['AF'].iloc[i-1] * (df['SAR'].iloc[i-1] - df['EP'].iloc[i-1])
            if df['Low'].iloc[i] < df['EP'].iloc[i-1]:
                df.loc[df.index[i], 'EP'] = df['Low'].iloc[i]
                df.loc[df.index[i], 'AF'] = min(0.2, df['AF'].iloc[i-1] + 0.02)
            else:
                df.loc[df.index[i], 'EP'] = df['EP'].iloc[i-1]
                df.loc[df.index[i], 'AF'] = df['AF'].iloc[i-1]
    
    return df

def find_support_resistance(df, window=10):
    """Find support and resistance levels"""
    highs = df['High'].rolling(window=window, center=True).max()
    lows = df['Low'].rolling(window=window, center=True).min()
    
    resistance_levels = df[df['High'] == highs]['High'].unique()
    support_levels = df[df['Low'] == lows]['Low'].unique()
    
    # Get recent levels (last 5)
    resistance = sorted(resistance_levels[-5:], reverse=True)[:3]
    support = sorted(support_levels[-5:])[:3]
    
    return support, resistance

def get_swing_trading_signal(df, current_price, selected_oscillators):
    """Generate swing trading signals based on technical indicators"""
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    signals = []
    score = 0
    
    # RSI Analysis
    if "RSI" in selected_oscillators:
        if latest['RSI'] < 30:
            signals.append("🟢 RSI Oversold (<30) - Potential Buy")
            score += 2
        elif latest['RSI'] > 70:
            signals.append("🔴 RSI Overbought (>70) - Potential Sell")
            score -= 2
        elif 40 <= latest['RSI'] <= 60:
            signals.append("🟡 RSI Neutral (40-60)")
    
    # MACD Analysis
    if "MACD" in selected_oscillators:
        if latest['MACD'] > latest['Signal_Line'] and prev['MACD'] <= prev['Signal_Line']:
            signals.append("🟢 MACD Bullish Crossover - Buy Signal")
            score += 2
        elif latest['MACD'] < latest['Signal_Line'] and prev['MACD'] >= prev['Signal_Line']:
            signals.append("🔴 MACD Bearish Crossover - Sell Signal")
            score -= 2
    
    # Moving Average Analysis
    if latest['EMA_9'] > latest['EMA_21'] and prev['EMA_9'] <= prev['EMA_21']:
        signals.append("🟢 EMA Golden Cross - Strong Buy")
        score += 3
    elif latest['EMA_9'] < latest['EMA_21'] and prev['EMA_9'] >= prev['EMA_21']:
        signals.append("🔴 EMA Death Cross - Strong Sell")
        score -= 3
    
    # Bollinger Bands
    if current_price < latest['BB_Lower']:
        signals.append("🟢 Price Below Lower Bollinger Band - Oversold")
        score += 1
    elif current_price > latest['BB_Upper']:
        signals.append("🔴 Price Above Upper Bollinger Band - Overbought")
        score -= 1
    
    # Volume Analysis
    if latest['Volume_Ratio'] > 1.5:
        signals.append("📊 High Volume (1.5x avg) - Strong Move")
        score += 1
    
    # Stochastic
    if "Stochastic" in selected_oscillators:
        if latest['Stoch_%K'] < 20:
            signals.append("🟢 Stochastic Oversold (<20)")
            score += 1
        elif latest['Stoch_%K'] > 80:
            signals.append("🔴 Stochastic Overbought (>80)")
            score -= 1
    
    # CCI Analysis
    if "CCI" in selected_oscillators:
        if latest['CCI'] < -100:
            signals.append("🟢 CCI Oversold (<-100) - Buy Signal")
            score += 2
        elif latest['CCI'] > 100:
            signals.append("🔴 CCI Overbought (>100) - Sell Signal")
            score -= 2
    
    # Williams %R
    if "Williams %R" in selected_oscillators:
        if latest['Williams_%R'] < -80:
            signals.append("🟢 Williams %R Oversold (<-80)")
            score += 1
        elif latest['Williams_%R'] > -20:
            signals.append("🔴 Williams %R Overbought (>-20)")
            score -= 1
    
    # ADX Trend Strength
    if "ADX" in selected_oscillators:
        if latest['ADX'] > 25:
            if latest['+DI'] > latest['-DI']:
                signals.append("📈 ADX Strong Uptrend (>25)")
                score += 2
            else:
                signals.append("📉 ADX Strong Downtrend (>25)")
                score -= 2
        else:
            signals.append("🟡 ADX Weak Trend (<25)")
    
    # Trend Analysis
    if latest['Close'] > latest['SMA_20']:
        signals.append("📈 Price Above 20 SMA - Uptrend")
        score += 1
    else:
        signals.append("📉 Price Below 20 SMA - Downtrend")
        score -= 1
    
    # Final recommendation
    if score >= 5:
        recommendation = "STRONG BUY"
        color = "#00FF00"
    elif score >= 2:
        recommendation = "BUY"
        color = "#90EE90"
    elif score >= -2:
        recommendation = "HOLD"
        color = "#FFA500"
    elif score >= -5:
        recommendation = "SELL"
        color = "#FFB6C1"
    else:
        recommendation = "STRONG SELL"
        color = "#FF0000"
    
    return recommendation, signals, score, color

def calculate_swing_targets(df, current_price):
    """Calculate entry, stop-loss, and target prices for swing trading"""
    latest = df.iloc[-1]
    atr = latest['ATR']
    
    # Support and Resistance
    support_levels, resistance_levels = find_support_resistance(df)
    
    # Entry price (current price)
    entry = current_price
    
    # Stop Loss (2 ATR below entry for long, or nearest support)
    stop_loss = entry - (2 * atr)
    if support_levels:
        nearest_support = [s for s in support_levels if s < entry]
        if nearest_support:
            stop_loss = max(stop_loss, min(nearest_support))
    
    # Target prices
    target_1 = entry + (1.5 * atr)  # 1.5:1 risk-reward
    target_2 = entry + (2.5 * atr)  # 2.5:1 risk-reward
    target_3 = entry + (3.5 * atr)  # 3.5:1 risk-reward
    
    if resistance_levels:
        # Adjust targets based on resistance
        for i, resistance in enumerate(resistance_levels):
            if resistance > entry:
                if i == 0:
                    target_1 = min(target_1, resistance)
                elif i == 1:
                    target_2 = min(target_2, resistance)
                elif i == 2:
                    target_3 = min(target_3, resistance)
    
    risk = entry - stop_loss
    reward_1 = target_1 - entry
    reward_2 = target_2 - entry
    reward_3 = target_3 - entry
    
    rr_ratio_1 = reward_1 / risk if risk > 0 else 0
    rr_ratio_2 = reward_2 / risk if risk > 0 else 0
    rr_ratio_3 = reward_3 / risk if risk > 0 else 0
    
    return {
        'entry': entry,
        'stop_loss': stop_loss,
        'target_1': target_1,
        'target_2': target_2,
        'target_3': target_3,
        'rr_ratio_1': rr_ratio_1,
        'rr_ratio_2': rr_ratio_2,
        'rr_ratio_3': rr_ratio_3,
        'support': support_levels,
        'resistance': resistance_levels,
        'atr': atr
    }

def calculate_intrinsic_value_dcf(info):
    """Calculate intrinsic value using simplified DCF model"""
    try:
        fcf = info.get('freeCashflow', 0)
        if fcf == 0:
            return None
        
        growth_rate = 0.10  # Assume 10% growth
        discount_rate = 0.12  # 12% discount rate
        terminal_growth = 0.03  # 3% terminal growth
        years = 5
        
        # Project cash flows
        cash_flows = []
        for year in range(1, years + 1):
            cash_flows.append(fcf * ((1 + growth_rate) ** year))
        
        # Calculate terminal value
        terminal_value = (cash_flows[-1] * (1 + terminal_growth)) / (discount_rate - terminal_growth)
        
        # Discount cash flows
        pv_cash_flows = sum([cf / ((1 + discount_rate) ** (i + 1)) for i, cf in enumerate(cash_flows)])
        pv_terminal = terminal_value / ((1 + discount_rate) ** years)
        
        enterprise_value = pv_cash_flows + pv_terminal
        
        # Calculate equity value
        cash = info.get('totalCash', 0)
        debt = info.get('totalDebt', 0)
        equity_value = enterprise_value + cash - debt
        
        shares_outstanding = info.get('sharesOutstanding', 1)
        intrinsic_value_per_share = equity_value / shares_outstanding
        
        return intrinsic_value_per_share
    except:
        return None

def calculate_fair_value_pe(info):
    """Calculate fair value using P/E ratio method"""
    try:
        eps = info.get('trailingEps', 0)
        industry_pe = 20  # Average industry P/E (can be customized)
        
        if eps > 0:
            return eps * industry_pe
        return None
    except:
        return None

def calculate_fair_value_pb(info):
    """Calculate fair value using P/B ratio method"""
    try:
        book_value = info.get('bookValue', 0)
        industry_pb = 3  # Average industry P/B (can be customized)
        
        if book_value > 0:
            return book_value * industry_pb
        return None
    except:
        return None

def get_valuation_metrics(info, current_price):
    """Calculate various valuation metrics"""
    metrics = {}
    
    # P/E Ratio
    eps = info.get('trailingEps', 0)
    metrics['P/E Ratio'] = current_price / eps if eps > 0 else 'N/A'
    metrics['Forward P/E'] = info.get('forwardPE', 'N/A')
    
    # P/B Ratio
    book_value = info.get('bookValue', 0)
    metrics['P/B Ratio'] = current_price / book_value if book_value > 0 else 'N/A'
    
    # Other metrics
    metrics['PEG Ratio'] = info.get('pegRatio', 'N/A')
    metrics['Dividend Yield'] = f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else 'N/A'
    metrics['ROE'] = f"{info.get('returnOnEquity', 0) * 100:.2f}%" if info.get('returnOnEquity') else 'N/A'
    metrics['Debt to Equity'] = info.get('debtToEquity', 'N/A')
    
    return metrics

def get_recommendation(current_price, fair_values):
    """Generate buy/sell recommendation"""
    valid_values = [v for v in fair_values.values() if v is not None and v > 0]
    
    if not valid_values:
        return "HOLD", "Insufficient data for valuation", "#FFA500", 0
    
    avg_fair_value = np.mean(valid_values)
    upside = ((avg_fair_value - current_price) / current_price) * 100
    
    if upside > 20:
        return "STRONG BUY", f"Undervalued by {upside:.1f}%", "#00FF00", upside
    elif upside > 10:
        return "BUY", f"Undervalued by {upside:.1f}%", "#90EE90", upside
    elif upside > -10:
        return "HOLD", f"Fairly valued ({upside:.1f}%)", "#FFA500", upside
    elif upside > -20:
        return "SELL", f"Overvalued by {abs(upside):.1f}%", "#FFB6C1", upside
    else:
        return "STRONG SELL", f"Overvalued by {abs(upside):.1f}%", "#FF0000", upside

# Main analysis
if stock_symbol:
    # Determine period based on timeframe
    period_map = {
        "1 Week": "5d",
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y"
    }
    selected_period = period_map[timeframe]
    
    stock, info, hist = get_stock_data(stock_symbol, selected_period)
    
    if info and hist is not None and not hist.empty:
        current_price = info.get('currentPrice', hist['Close'].iloc[-1])
        
        # Display company info
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.subheader(f"{info.get('longName', 'N/A')}")
            st.write(f"**Sector:** {info.get('sector', 'N/A')} | **Industry:** {info.get('industry', 'N/A')}")
        
        with col2:
            change = hist['Close'].iloc[-1] - hist['Close'].iloc[-2] if len(hist) > 1 else 0
            change_pct = (change / hist['Close'].iloc[-2] * 100) if len(hist) > 1 else 0
            st.metric("Current Price", f"₹{current_price:.2f}", f"{change_pct:+.2f}%")
        
        with col3:
            market_cap = info.get('marketCap', 0)
            st.metric("Market Cap", f"₹{market_cap/1e7:.2f} Cr" if market_cap else 'N/A')
        
        st.divider()
        
        # Calculate technical indicators for swing trading
        hist_with_indicators = calculate_technical_indicators(hist.copy())
        
        if analysis_type == "Swing Trading (1 Day - 1 Month)":
            # SWING TRADING ANALYSIS
            st.header("🎯 Swing Trading Analysis")
            
            # Get trading signals
            recommendation, signals, score, color = get_swing_trading_signal(hist_with_indicators, current_price, show_oscillators)
            targets = calculate_swing_targets(hist_with_indicators, current_price)
            
            # Display recommendation
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"### **{recommendation}**")
                st.markdown(f"<p style='color: {color}; font-size: 18px; font-weight: bold;'>Signal Score: {score}</p>", unsafe_allow_html=True)
            
            with col2:
                st.metric("Entry Price", f"₹{targets['entry']:.2f}")
                st.caption("Current market price")
            
            with col3:
                st.metric("Stop Loss", f"₹{targets['stop_loss']:.2f}")
                risk_pct = ((targets['entry'] - targets['stop_loss']) / targets['entry'] * 100)
                st.caption(f"Risk: {risk_pct:.1f}%")
            
            with col4:
                st.metric("Primary Target", f"₹{targets['target_1']:.2f}")
                reward_pct = ((targets['target_1'] - targets['entry']) / targets['entry'] * 100)
                st.caption(f"Reward: {reward_pct:.1f}%")
            
            st.divider()
            
            # Trading Signals
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📊 Technical Signals")
                for signal in signals:
                    st.write(signal)
            
            with col2:
                st.subheader("🎯 Price Targets & Risk Management")
                
                st.write(f"**Entry:** ₹{targets['entry']:.2f}")
                st.write(f"**Stop Loss:** ₹{targets['stop_loss']:.2f} (Risk: {((targets['entry']-targets['stop_loss'])/targets['entry']*100):.1f}%)")
                st.write("")
                st.write(f"**Target 1:** ₹{targets['target_1']:.2f} (R:R = 1:{targets['rr_ratio_1']:.1f})")
                st.write(f"**Target 2:** ₹{targets['target_2']:.2f} (R:R = 1:{targets['rr_ratio_2']:.1f})")
                st.write(f"**Target 3:** ₹{targets['target_3']:.2f} (R:R = 1:{targets['rr_ratio_3']:.1f})")
                st.write("")
                st.write(f"**ATR (Volatility):** ₹{targets['atr']:.2f}")
            
            st.divider()
            
            # Support and Resistance
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📉 Support Levels")
                if targets['support']:
                    for i, level in enumerate(targets['support'], 1):
                        distance = ((current_price - level) / current_price * 100)
                        st.write(f"S{i}: ₹{level:.2f} ({distance:.1f}% below)")
                else:
                    st.write("No recent support levels identified")
            
            with col2:
                st.subheader("📈 Resistance Levels")
                if targets['resistance']:
                    for i, level in enumerate(targets['resistance'], 1):
                        distance = ((level - current_price) / current_price * 100)
                        st.write(f"R{i}: ₹{level:.2f} ({distance:.1f}% above)")
                else:
                    st.write("No recent resistance levels identified")
            
            st.divider()
            
            # Technical Indicators Chart
            st.subheader("📈 Technical Analysis Charts")
            
            # Determine how many subplots we need
            num_oscillator_plots = 0
            if "RSI" in show_oscillators:
                num_oscillator_plots += 1
            if "MACD" in show_oscillators:
                num_oscillator_plots += 1
            if "Stochastic" in show_oscillators:
                num_oscillator_plots += 1
            if "CCI" in show_oscillators:
                num_oscillator_plots += 1
            if "Williams %R" in show_oscillators:
                num_oscillator_plots += 1
            if "ADX" in show_oscillators:
                num_oscillator_plots += 1
            
            total_rows = 1 + num_oscillator_plots  # Price chart + oscillators
            
            # Calculate row heights
            price_chart_height = 0.5
            oscillator_height = 0.5 / num_oscillator_plots if num_oscillator_plots > 0 else 0
            row_heights = [price_chart_height] + [oscillator_height] * num_oscillator_plots
            
            # Create subplot titles
            subplot_titles = ['Price & Moving Averages']
            if "MACD" in show_oscillators:
                subplot_titles.append('MACD')
            if "RSI" in show_oscillators:
                subplot_titles.append('RSI')
            if "Stochastic" in show_oscillators:
                subplot_titles.append('Stochastic')
            if "CCI" in show_oscillators:
                subplot_titles.append('CCI')
            if "Williams %R" in show_oscillators:
                subplot_titles.append('Williams %R')
            if "ADX" in show_oscillators:
                subplot_titles.append('ADX')
            
            # Create subplots
            fig = make_subplots(
                rows=total_rows, cols=1,
                row_heights=row_heights,
                subplot_titles=subplot_titles,
                vertical_spacing=0.03,
                specs=[[{"secondary_y": False}]] * total_rows
            )
            
            # === ROW 1: PRICE CHART ===
            # Candlestick chart
            fig.add_trace(go.Candlestick(
                x=hist_with_indicators.index,
                open=hist_with_indicators['Open'],
                high=hist_with_indicators['High'],
                low=hist_with_indicators['Low'],
                close=hist_with_indicators['Close'],
                name='Price',
                showlegend=False
            ), row=1, col=1)
            
            # Add selected moving averages
            ma_colors = {
                'SMA 5': 'lightblue',
                'SMA 10': 'cyan',
                'SMA 20': 'purple',
                'SMA 50': 'pink',
                'EMA 9': 'orange',
                'EMA 21': 'blue',
                'EMA 50': 'red'
            }
            
            for ma in show_ma:
                col_name = ma.replace(' ', '_')
                if col_name in hist_with_indicators.columns:
                    fig.add_trace(go.Scatter(
                        x=hist_with_indicators.index,
                        y=hist_with_indicators[col_name],
                        name=ma,
                        line=dict(color=ma_colors.get(ma, 'gray'), width=1.5)
                    ), row=1, col=1)
            
            # Add Bollinger Bands if selected
            if "Bollinger Bands" in show_volatility:
                fig.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['BB_Upper'],
                    name='BB Upper',
                    line=dict(color='gray', width=1, dash='dot'),
                    showlegend=True
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['BB_Lower'],
                    name='BB Lower',
                    line=dict(color='gray', width=1, dash='dot'),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)',
                    showlegend=True
                ), row=1, col=1)
            
            # Add Keltner Channels if selected
            if "Keltner Channels" in show_volatility:
                fig.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['KC_Upper'],
                    name='KC Upper',
                    line=dict(color='green', width=1, dash='dot')
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['KC_Lower'],
                    name='KC Lower',
                    line=dict(color='red', width=1, dash='dot')
                ), row=1, col=1)
            
            # Add Supertrend if selected
            if "Supertrend" in show_trend:
                supertrend_colors = ['green' if x == 1 else 'red' for x in hist_with_indicators['Supertrend_Direction']]
                fig.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['Supertrend'],
                    name='Supertrend',
                    line=dict(color='purple', width=2),
                    mode='lines'
                ), row=1, col=1)
            
            # Add Parabolic SAR if selected
            if "Parabolic SAR" in show_trend:
                fig.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['SAR'],
                    name='Parabolic SAR',
                    mode='markers',
                    marker=dict(size=3, color='blue')
                ), row=1, col=1)
            
            # Add target and stop loss lines
            fig.add_hline(y=targets['target_1'], line_dash="dash", line_color="green",
                         annotation_text=f"T1: ₹{targets['target_1']:.2f}", row=1, col=1)
            fig.add_hline(y=targets['stop_loss'], line_dash="dash", line_color="red",
                         annotation_text=f"SL: ₹{targets['stop_loss']:.2f}", row=1, col=1)
            
            # === ADD OSCILLATOR PLOTS ===
            current_row = 2
            
            # MACD
            if "MACD" in show_oscillators:
                colors = ['green' if val >= 0 else 'red' for val in hist_with_indicators['MACD_Histogram']]
                fig.add_trace(go.Bar(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['MACD_Histogram'],
                    name='MACD Histogram',
                    marker_color=colors,
                    showlegend=False
                ), row=current_row, col=1)
                fig.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['MACD'],
                    name='MACD',
                    line=dict(color='blue', width=1.5),
                    showlegend=False
                ), row=current_row, col=1)
                fig.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['Signal_Line'],
                    name='Signal',
                    line=dict(color='orange', width=1.5),
                    showlegend=False
                ), row=current_row, col=1)
                fig.update_yaxes(title_text="MACD", row=current_row, col=1)
                current_row += 1
            
            # RSI
            if "RSI" in show_oscillators:
                fig.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['RSI'],
                    name='RSI',
                    line=dict(color='purple', width=2),
                    showlegend=False
                ), row=current_row, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color="gray", row=current_row, col=1)
                fig.update_yaxes(title_text="RSI", row=current_row, col=1)
                current_row += 1
            
            # Stochastic
            if "Stochastic" in show_oscillators:
                fig.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['Stoch_%K'],
                    name='%K',
                    line=dict(color='blue', width=1.5),
                    showlegend=False
                ), row=current_row, col=1)
                fig.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['Stoch_%D'],
                    name='%D',
                    line=dict(color='orange', width=1.5),
                    showlegend=False
                ), row=current_row, col=1)
                fig.add_hline(y=80, line_dash="dash", line_color="red", row=current_row, col=1)
                fig.add_hline(y=20, line_dash="dash", line_color="green", row=current_row, col=1)
                fig.update_yaxes(title_text="Stochastic", row=current_row, col=1)
                current_row += 1
            
            # CCI
            if "CCI" in show_oscillators:
                fig.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['CCI'],
                    name='CCI',
                    line=dict(color='teal', width=2),
                    showlegend=False
                ), row=current_row, col=1)
                fig.add_hline(y=100, line_dash="dash", line_color="red", row=current_row, col=1)
                fig.add_hline(y=-100, line_dash="dash", line_color="green", row=current_row, col=1)
                fig.add_hline(y=0, line_dash="dot", line_color="gray", row=current_row, col=1)
                fig.update_yaxes(title_text="CCI", row=current_row, col=1)
                current_row += 1
            
            # Williams %R
            if "Williams %R" in show_oscillators:
                fig.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['Williams_%R'],
                    name='Williams %R',
                    line=dict(color='brown', width=2),
                    showlegend=False
                ), row=current_row, col=1)
                fig.add_hline(y=-20, line_dash="dash", line_color="red", row=current_row, col=1)
                fig.add_hline(y=-80, line_dash="dash", line_color="green", row=current_row, col=1)
                fig.add_hline(y=-50, line_dash="dot", line_color="gray", row=current_row, col=1)
                fig.update_yaxes(title_text="Williams %R", row=current_row, col=1)
                current_row += 1
            
            # ADX
            if "ADX" in show_oscillators:
                fig.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['ADX'],
                    name='ADX',
                    line=dict(color='black', width=2),
                    showlegend=False
                ), row=current_row, col=1)
                fig.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['+DI'],
                    name='+DI',
                    line=dict(color='green', width=1.5),
                    showlegend=False
                ), row=current_row, col=1)
                fig.add_trace(go.Scatter(
                    x=hist_with_indicators.index,
                    y=hist_with_indicators['-DI'],
                    name='-DI',
                    line=dict(color='red', width=1.5),
                    showlegend=False
                ), row=current_row, col=1)
                fig.add_hline(y=25, line_dash="dash", line_color="gray", row=current_row, col=1)
                fig.update_yaxes(title_text="ADX", row=current_row, col=1)
                current_row += 1
            
            chart_height = 400 + (num_oscillator_plots * 200)
            fig.update_layout(height=chart_height, showlegend=True, xaxis_rangeslider_visible=False)
            fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume Chart (if selected)
            if show_volume:
                st.subheader("📊 Volume Analysis")
                fig_vol = go.Figure()
                
                if "Volume" in show_volume:
                    colors = ['red' if hist_with_indicators['Close'].iloc[i] < hist_with_indicators['Open'].iloc[i] else 'green' 
                             for i in range(len(hist_with_indicators))]
                    fig_vol.add_trace(go.Bar(
                        x=hist_with_indicators.index,
                        y=hist_with_indicators['Volume'],
                        name='Volume',
                        marker_color=colors
                    ))
                
                if "Volume SMA" in show_volume:
                    fig_vol.add_trace(go.Scatter(
                        x=hist_with_indicators.index,
                        y=hist_with_indicators['Volume_SMA'],
                        name='Volume SMA',
                        line=dict(color='blue', width=2)
                    ))
                
                if "OBV" in show_volume:
                    fig_vol.add_trace(go.Scatter(
                        x=hist_with_indicators.index,
                        y=hist_with_indicators['OBV'],
                        name='OBV',
                        line=dict(color='purple', width=2),
                        yaxis='y2'
                    ))
                    fig_vol.update_layout(
                        yaxis2=dict(title="OBV", overlaying='y', side='right')
                    )
                
                fig_vol.update_layout(height=300, showlegend=True, yaxis_title="Volume")
                st.plotly_chart(fig_vol, use_container_width=True)
            
            # Key Metrics Table - Dynamic based on selected indicators
            st.subheader("📊 Current Indicator Values")
            latest = hist_with_indicators.iloc[-1]
            
            # Collect all selected indicators
            all_metrics = {}
            
            # Moving Averages
            for ma in show_ma:
                col_name = ma.replace(' ', '_')
                if col_name in hist_with_indicators.columns:
                    all_metrics[ma] = f"₹{latest[col_name]:.2f}"
            
            # Oscillators
            if "RSI" in show_oscillators:
                all_metrics["RSI"] = f"{latest['RSI']:.2f}"
            if "MACD" in show_oscillators:
                all_metrics["MACD"] = f"{latest['MACD']:.4f}"
                all_metrics["MACD Signal"] = f"{latest['Signal_Line']:.4f}"
            if "Stochastic" in show_oscillators:
                all_metrics["Stochastic %K"] = f"{latest['Stoch_%K']:.2f}"
                all_metrics["Stochastic %D"] = f"{latest['Stoch_%D']:.2f}"
            if "CCI" in show_oscillators:
                all_metrics["CCI"] = f"{latest['CCI']:.2f}"
            if "Williams %R" in show_oscillators:
                all_metrics["Williams %R"] = f"{latest['Williams_%R']:.2f}"
            if "ADX" in show_oscillators:
                all_metrics["ADX"] = f"{latest['ADX']:.2f}"
                all_metrics["+DI"] = f"{latest['+DI']:.2f}"
                all_metrics["-DI"] = f"{latest['-DI']:.2f}"
            
            # Volatility
            if "ATR" in show_volatility:
                all_metrics["ATR"] = f"₹{latest['ATR']:.2f}"
            if "Bollinger Bands" in show_volatility:
                all_metrics["BB Upper"] = f"₹{latest['BB_Upper']:.2f}"
                all_metrics["BB Lower"] = f"₹{latest['BB_Lower']:.2f}"
            if "Keltner Channels" in show_volatility:
                all_metrics["KC Upper"] = f"₹{latest['KC_Upper']:.2f}"
                all_metrics["KC Lower"] = f"₹{latest['KC_Lower']:.2f}"
            
            # Volume
            if "Volume" in show_volume or "Volume SMA" in show_volume:
                all_metrics["Volume Ratio"] = f"{latest['Volume_Ratio']:.2f}x"
            if "OBV" in show_volume:
                all_metrics["OBV"] = f"{latest['OBV']:.0f}"
            
            # Display metrics in columns
            if all_metrics:
                num_cols = min(4, len(all_metrics))
                cols = st.columns(num_cols)
                for i, (metric, value) in enumerate(all_metrics.items()):
                    with cols[i % num_cols]:
                        st.metric(metric, value)
        
        else:
            # FUNDAMENTAL ANALYSIS (Long-term)
            st.header("🎯 Valuation Analysis")
            
            fair_value_dcf = calculate_intrinsic_value_dcf(info)
            fair_value_pe = calculate_fair_value_pe(info)
            fair_value_pb = calculate_fair_value_pb(info)
            
            fair_values = {
                'DCF Model': fair_value_dcf,
                'P/E Based': fair_value_pe,
                'P/B Based': fair_value_pb
            }
            
            # Calculate average fair value
            valid_fair_values = [v for v in fair_values.values() if v is not None and v > 0]
            avg_fair_value = np.mean(valid_fair_values) if valid_fair_values else current_price
            
            # Get recommendation
            recommendation, reason, color, upside = get_recommendation(current_price, fair_values)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Average Fair Value", f"₹{avg_fair_value:.2f}")
            
            with col2:
                st.metric("Upside/Downside", f"{upside:.1f}%")
            
            with col3:
                st.markdown(f"### **{recommendation}**")
                st.markdown(f"<p style='color: {color}; font-size: 14px;'>{reason}</p>", unsafe_allow_html=True)
            
            with col4:
                # Visual indicator
                if "BUY" in recommendation:
                    st.success("✅ Attractive Valuation")
                elif "SELL" in recommendation:
                    st.error("⚠️ Expensive Valuation")
                else:
                    st.warning("⚪ Neutral Valuation")
            
            st.divider()
            
            # Fair value breakdown
            st.subheader("📊 Fair Value Breakdown")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Fair value table
                fv_data = []
                for method, value in fair_values.items():
                    if value and value > 0:
                        diff = ((value - current_price) / current_price) * 100
                        fv_data.append({
                            'Method': method,
                            'Fair Value (₹)': f"{value:.2f}",
                            'Difference': f"{diff:.1f}%"
                        })
                
                if fv_data:
                    df_fv = pd.DataFrame(fv_data)
                    st.dataframe(df_fv, use_container_width=True, hide_index=True)
            
            with col2:
                # Valuation chart
                fig = go.Figure()
                
                methods = []
                values_list = []
                colors_list = []
                
                for method, value in fair_values.items():
                    if value and value > 0:
                        methods.append(method)
                        values_list.append(value)
                        colors_list.append('#4CAF50' if value > current_price else '#F44336')
                
                methods.append('Current Price')
                values_list.append(current_price)
                colors_list.append('#2196F3')
                
                fig.add_trace(go.Bar(
                    x=methods,
                    y=values_list,
                    marker_color=colors_list,
                    text=[f"₹{v:.2f}" for v in values_list],
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title="Fair Value Comparison",
                    yaxis_title="Price (₹)",
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Valuation metrics
            st.subheader("📈 Key Valuation Metrics")
            
            metrics = get_valuation_metrics(info, current_price)
            
            cols = st.columns(4)
            for i, (metric, value) in enumerate(metrics.items()):
                with cols[i % 4]:
                    st.metric(metric, value)
            
            st.divider()
            
            # Price history chart
            st.subheader("📉 Price History")
            
            fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], 
                               subplot_titles=('Price Movement', 'Volume'),
                               vertical_spacing=0.05)
            
            # Price line
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['Close'],
                name='Close Price',
                line=dict(color='#2196F3', width=2)
            ), row=1, col=1)
            
            # Add fair value line
            fig.add_hline(y=avg_fair_value, line_dash="dash", line_color="green", 
                         annotation_text=f"Avg Fair Value: ₹{avg_fair_value:.2f}",
                         row=1, col=1)
            
            # Volume bars
            colors = ['red' if hist['Close'].iloc[i] < hist['Open'].iloc[i] else 'green' 
                     for i in range(len(hist))]
            
            fig.add_trace(go.Bar(
                x=hist.index,
                y=hist['Volume'],
                name='Volume',
                marker_color=colors,
                showlegend=False
            ), row=2, col=1)
            
            fig.update_layout(height=600, showlegend=True)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional info
            with st.expander("ℹ️ Company Information"):
                st.write(f"**Description:** {info.get('longBusinessSummary', 'N/A')}")
                st.write(f"**Website:** {info.get('website', 'N/A')}")
                st.write(f"**Employees:** {info.get('fullTimeEmployees', 'N/A')}")
    
    else:
        st.error("Unable to fetch stock data. Please check the symbol and try again.")
        st.info("Make sure to use .NS for NSE stocks (e.g., RELIANCE.NS) or .BO for BSE stocks (e.g., TCS.BO)")

# Footer
st.divider()
st.markdown("""
### 📝 Methodology

#### Swing Trading Analysis (1 Day - 1 Month):
- **Technical Indicators:** RSI, MACD, Moving Averages, Bollinger Bands, Stochastic
- **Support & Resistance:** Automatically identified key price levels
- **Risk Management:** ATR-based stop loss and target calculations
- **Signal Score:** Composite score from multiple technical indicators

**Key Signals:**
- **RSI:** Oversold (<30) = Buy, Overbought (>70) = Sell
- **MACD:** Bullish/Bearish crossovers indicate trend changes
- **Moving Averages:** Golden Cross (Buy) / Death Cross (Sell)
- **Bollinger Bands:** Price touching bands indicates potential reversal
- **Volume:** Above average volume confirms trend strength

**Risk-Reward:** Targets calculated with 1.5:1, 2.5:1, and 3.5:1 ratios

#### Fundamental Analysis (Long-term):
- **DCF Model:** Discounted Cash Flow analysis with 5-year projection
- **P/E Based:** Fair value using industry average P/E ratio (20x)
- **P/B Based:** Fair value using industry average P/B ratio (3x)

**Disclaimer:** This tool is for educational purposes only. It is NOT financial advice. Always conduct your own research and consult with a qualified financial advisor before making investment decisions. Past performance does not guarantee future results. Trading involves risk of loss.
""")
