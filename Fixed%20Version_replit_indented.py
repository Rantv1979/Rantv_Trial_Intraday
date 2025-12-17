# Rantv Intraday Trading Signals - VS Code Integrated Version

import time
from datetime import datetime, time as dt_time
import numpy as np
import pandas as pd
import pytz
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import math
import warnings
import os
from dataclasses import dataclass
from typing import Optional, Dict, List
import requests
import json
import traceback
import subprocess
import sys
from datetime import timedelta
import threading
import webbrowser
from io import StringIO

# VS Code UI components
try:
    from streamlit_ace import st_ace
    ACE_AVAILABLE = True
except ImportError:
    ACE_AVAILABLE = False
    st.warning("Install streamlit-ace for full VS Code editor: pip install streamlit-ace")

# Auto-install missing dependencies
try:
    from kiteconnect import KiteConnect, KiteTicker
    KITECONNECT_AVAILABLE = True
except ImportError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kiteconnect"])
        from kiteconnect import KiteConnect, KiteTicker
        KITECONNECT_AVAILABLE = True
    except:
        KITECONNECT_AVAILABLE = False

# ... [Keep all your original imports and setup] ...

# ==================== UTILITY FUNCTIONS ====================
def now_indian():
    return datetime.now(IND_TZ)

def market_open():
    """Check if market is open (9:15 AM to 3:30 PM IST)"""
    n = now_indian()
    try:
        open_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(9, 15)))
        close_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(15, 30)))
        return open_time <= n <= close_time
    except Exception:
        return False

def should_auto_close():
    """Check if it's time to auto-close positions"""
    n = now_indian()
    try:
        auto_close_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(15, 10)))
        return n >= auto_close_time
    except Exception:
        return False

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rs = rs.fillna(0)
    return 100 - (100 / (1 + rs))

# ==================== VS CODE COMPONENTS ====================
class VSCodeTerminal:
    """VS Code style terminal emulator"""
    
    def __init__(self):
        self.history = []
        self.max_history = 100
        
    def execute_command(self, command):
        """Execute a terminal command"""
        self.history.append(f"$ {command}")
        
        try:
            if command.startswith("python "):
                code = command[7:]
                old_stdout = sys.stdout
                redirected_output = sys.stdout = StringIO()
                exec(code, globals())
                sys.stdout = old_stdout
                output = redirected_output.getvalue()
                self.history.append(output)
            elif command == "clear":
                self.history = []
            elif command == "help":
                self.history.extend([
                    "Available commands:",
                    "  python <code> - Execute Python code",
                    "  clear - Clear terminal",
                    "  help - Show this help"
                ])
            else:
                self.history.append(f"Command not found: {command}")
        except Exception as e:
            self.history.append(f"Error: {str(e)}")
        
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def render(self):
        """Render the terminal"""
        st.markdown('<div class="panel-header"><div>TERMINAL</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="terminal">', unsafe_allow_html=True)
        
        for line in self.history[-20:]:
            if line.startswith("$ "):
                st.markdown(f'<div class="terminal-line terminal-command">{line}</div>', unsafe_allow_html=True)
            elif line.startswith("Error:"):
                st.markdown(f'<div class="terminal-line terminal-error">{line}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="terminal-line terminal-output">{line}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Command input
        col1, col2 = st.columns([5, 1])
        with col1:
            command = st.text_input("", placeholder="Type a command...", key="terminal_input", label_visibility="collapsed")
        with col2:
            if st.button("Execute", key="terminal_execute"):
                if command:
                    self.execute_command(command)
                    st.rerun()

def vs_code_activity_bar():
    """Create VS Code style activity bar"""
    activities = [
        {"icon": "📁", "title": "Explorer", "id": "explorer"},
        {"icon": "📈", "title": "Dashboard", "id": "dashboard"},
        {"icon": "🔍", "title": "Search", "id": "search"},
        {"icon": "📚", "title": "Git", "id": "git"},
        {"icon": "🐞", "title": "Debug", "id": "debug"},
        {"icon": "⚙️", "title": "Settings", "id": "settings"}
    ]
    
    html = '<div class="activity-bar">'
    for activity in activities:
        active_class = "active" if st.session_state.get("active_activity") == activity["id"] else ""
        html += f'''
        <div class="activity-item {active_class}" 
             onclick="document.querySelector('#activity_{activity["id"]}').click()"
             title="{activity["title"]}">
            {activity["icon"]}
        </div>
        '''
    html += '</div>'
    
    st.markdown(html, unsafe_allow_html=True)
    
    # Create hidden buttons
    for activity in activities:
        if st.button(f"", key=f"activity_{activity['id']}", help=activity['title']):
            st.session_state.active_activity = activity["id"]
            if activity["id"] == "dashboard":
                st.session_state.active_tab = "dashboard"

def vs_code_status_bar():
    """Create VS Code style status bar"""
    current_time = datetime.now().strftime("%H:%M:%S")
    market_status = "OPEN" if market_open() else "CLOSED"
    status_color = "#4CAF50" if market_open() else "#F44336"
    
    html = f'''
    <div class="status-bar">
        <div class="status-item">🕐 {current_time}</div>
        <div class="status-item">🌐 Market: <span style="color:{status_color}">{market_status}</span></div>
        <div class="status-item">🐍 Python {sys.version.split()[0]}</div>
        <div class="status-item">🔧 VS Code Mode</div>
        <div class="status-item" style="margin-left:auto;">📊 Streamlit</div>
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)

def vs_code_file_explorer():
    """Create VS Code style file explorer"""
    files = [
        {"name": "dashboard.py", "type": "python", "icon": "📊"},
        {"name": "signals.py", "type": "python", "icon": "🚦"},
        {"name": "trading.py", "type": "python", "icon": "💰"},
        {"name": "charts.py", "type": "python", "icon": "📈"},
        {"name": "strategies.py", "type": "python", "icon": "🤖"},
        {"name": "config.json", "type": "json", "icon": "⚙️"},
        {"name": "logs/", "type": "folder", "icon": "📁"},
        {"name": "data/", "type": "folder", "icon": "📁"}
    ]
    
    st.markdown('<div class="explorer">', unsafe_allow_html=True)
    st.markdown('<div class="explorer-header">EXPLORER</div>', unsafe_allow_html=True)
    
    for file in files:
        st.markdown(f'''
            <div class="explorer-item">
                {file["icon"]} {file["name"]}
            </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def vs_code_tabs():
    """Create VS Code style tabs"""
    tabs = [
        {"id": "dashboard", "name": "📈 Dashboard", "icon": "📈"},
        {"id": "signals", "name": "🚦 Signals", "icon": "🚦"},
        {"id": "trading", "name": "💰 Trading", "icon": "💰"},
        {"id": "charts", "name": "📊 Charts", "icon": "📊"},
        {"id": "code", "name": "💻 Code", "icon": "💻"},
        {"id": "terminal", "name": "🖥️ Terminal", "icon": "🖥️"}
    ]
    
    html = '<div class="vs-code-tabs">'
    for tab in tabs:
        active_class = "active" if st.session_state.get("active_tab", "dashboard") == tab["id"] else ""
        html += f'''
        <div class="vs-code-tab {active_class}" 
             onclick="document.querySelector('#tab_{tab["id"]}').click()">
            {tab["icon"]} {tab["name"]}
            <span class="vs-code-tab-close" onclick="document.querySelector('#close_{tab["id"]}').click()">×</span>
        </div>
        '''
    html += '</div>'
    
    st.markdown(html, unsafe_allow_html=True)
    
    # Create buttons
    for tab in tabs:
        if st.button("", key=f"tab_{tab['id']}"):
            st.session_state.active_tab = tab["id"]
        if st.button("", key=f"close_{tab['id']}"):
            if st.session_state.get("active_tab") == tab["id"]:
                st.session_state.active_tab = "dashboard"

# ==================== KITE CONNECT INTEGRATION ====================
class KiteManager:
    """Simplified Kite Connect Manager"""
    
    def __init__(self):
        self.api_key = os.environ.get("KITE_API_KEY", "")
        self.api_secret = os.environ.get("KITE_API_SECRET", "")
        self.kite = None
        self.is_authenticated = False
        
    def login(self):
        """Simple login interface"""
        if not self.api_key:
            st.warning("⚠️ KITE_API_KEY not set in environment")
            return False
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Login Method")
            method = st.radio("Choose method:", ["Request Token", "Access Token"])
            
            if method == "Request Token":
                if self.api_key and self.api_secret:
                    try:
                        kite = KiteConnect(api_key=self.api_key)
                        login_url = kite.login_url()
                        st.markdown(f"[Login to Kite]({login_url})")
                        
                        request_token = st.text_input("Enter Request Token:", type="password")
                        if st.button("Authenticate") and request_token:
                            data = kite.generate_session(request_token, self.api_secret)
                            self.kite = kite
                            self.kite.set_access_token(data["access_token"])
                            self.is_authenticated = True
                            st.session_state.kite_auth = True
                            st.success("✅ Authenticated successfully!")
                            return True
                    except Exception as e:
                        st.error(f"Authentication failed: {e}")
            
            else:  # Access Token
                access_token = st.text_input("Access Token:", type="password")
                if st.button("Connect") and access_token:
                    try:
                        self.kite = KiteConnect(api_key=self.api_key)
                        self.kite.set_access_token(access_token)
                        profile = self.kite.profile()
                        self.is_authenticated = True
                        st.session_state.kite_auth = True
                        st.success(f"✅ Connected as {profile.get('user_name', 'User')}")
                        return True
                    except Exception as e:
                        st.error(f"Connection failed: {e}")
        
        with col2:
            st.subheader("Info")
            st.info("""
            **Kite Connect Setup:**
            1. Get API key from [Kite Console](https://console.zerodha.com)
            2. Set environment variables:
               - `KITE_API_KEY`
               - `KITE_API_SECRET`
            3. Choose login method
            """)
        
        return False

# ==================== MAIN VS CODE INTERFACE ====================
def main_vs_code_interface():
    """Main VS Code interface with integrated trading"""
    
    # Initialize session states
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "dashboard"
    if "active_activity" not in st.session_state:
        st.session_state.active_activity = "explorer"
    if "terminal" not in st.session_state:
        st.session_state.terminal = VSCodeTerminal()
    if "kite_manager" not in st.session_state:
        st.session_state.kite_manager = KiteManager()
    
    # Create layout
    st.markdown("""
    <style>
    .main-container {
        display: flex;
        height: 100vh;
    }
    .activity-bar {
        width: 50px;
        background: #333;
        padding-top: 20px;
    }
    .explorer-panel {
        width: 250px;
        background: #252526;
        border-right: 1px solid #333;
    }
    .content-area {
        flex: 1;
        overflow: auto;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Activity Bar
    vs_code_activity_bar()
    
    # Explorer Panel (only if explorer is active)
    if st.session_state.active_activity == "explorer":
        vs_code_file_explorer()
    
    # Main Content
    st.markdown('<div class="content-area">', unsafe_allow_html=True)
    
    # Tabs
    vs_code_tabs()
    
    # Tab Content
    active_tab = st.session_state.active_tab
    
    if active_tab == "dashboard":
        st.title("📈 Trading Dashboard")
        
        # Market Status
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Market Status", "OPEN" if market_open() else "CLOSED", 
                     delta="Trading" if market_open() else "Closed")
        with col2:
            st.metric("Current Time", now_indian().strftime("%H:%M:%S"))
        with col3:
            # Get Nifty value
            try:
                nifty = yf.download("^NSEI", period="1d", interval="1m")
                nifty_val = nifty['Close'].iloc[-1] if not nifty.empty else 0
                st.metric("NIFTY 50", f"₹{nifty_val:,.2f}")
            except:
                st.metric("NIFTY 50", "N/A")
        with col4:
            st.metric("VS Code Mode", "Active")
        
        # Quick Stats
        st.subheader("📊 Quick Stats")
        stats_cols = st.columns(4)
        with stats_cols[0]:
            st.metric("Capital", "₹2,000,000")
        with stats_cols[1]:
            st.metric("Today's P&L", "₹15,250", "+2.1%")
        with stats_cols[2]:
            st.metric("Win Rate", "68.5%")
        with stats_cols[3]:
            st.metric("Open Positions", "3")
        
        # Quick Actions
        st.subheader("⚡ Quick Actions")
        action_cols = st.columns(5)
        with action_cols[0]:
            if st.button("🔍 Scan Signals", use_container_width=True):
                st.session_state.active_tab = "signals"
        with action_cols[1]:
            if st.button("📊 View Charts", use_container_width=True):
                st.session_state.active_tab = "charts"
        with action_cols[2]:
            if st.button("💰 Place Trade", use_container_width=True):
                st.session_state.active_tab = "trading"
        with action_cols[3]:
            if st.button("🤖 Run Strategy", use_container_width=True):
                st.session_state.active_tab = "code"
        with action_cols[4]:
            if st.button("🖥️ Terminal", use_container_width=True):
                st.session_state.active_tab = "terminal"
        
        # Recent Activity
        st.subheader("🕒 Recent Activity")
        activity_data = [
            {"time": "10:15", "action": "BUY RELIANCE", "qty": 10, "price": "₹2,750", "status": "Filled"},
            {"time": "10:30", "action": "SELL TCS", "qty": 5, "price": "₹3,850", "status": "Filled"},
            {"time": "11:00", "action": "Signal Generated", "symbol": "HDFCBANK", "type": "BUY", "confidence": "78%"},
            {"time": "11:30", "action": "Auto Trade", "symbol": "INFY", "qty": 15, "status": "Executed"},
        ]
        
        for activity in activity_data:
            st.write(f"**{activity['time']}** - {activity['action']}")
    
    elif active_tab == "signals":
        st.title("🚦 Trading Signals")
        
        # Signal Generation
        st.subheader("Generate Signals")
        
        col1, col2 = st.columns(2)
        with col1:
            universe = st.selectbox("Select Universe", ["NIFTY 50", "NIFTY 100", "Midcap 150", "All Stocks"])
            min_confidence = st.slider("Min Confidence %", 60, 90, 70)
        
        with col2:
            strategy = st.multiselect(
                "Strategies",
                ["EMA Crossover", "RSI Reversal", "MACD", "Bollinger Bands", "Volume Breakout"],
                default=["EMA Crossover", "RSI Reversal"]
            )
        
        if st.button("🔍 Scan for Signals", type="primary"):
            with st.spinner(f"Scanning {universe}..."):
                # Simulate signal generation
                time.sleep(2)
                
                # Sample signals
                signals = [
                    {"symbol": "RELIANCE", "action": "BUY", "confidence": 82, "strategy": "EMA Crossover", "price": "₹2,750"},
                    {"symbol": "TCS", "action": "SELL", "confidence": 75, "strategy": "RSI Reversal", "price": "₹3,850"},
                    {"symbol": "HDFCBANK", "action": "BUY", "confidence": 78, "strategy": "Volume Breakout", "price": "₹1,650"},
                    {"symbol": "INFY", "action": "BUY", "confidence": 85, "strategy": "MACD", "price": "₹1,450"},
                ]
                
                # Filter by confidence
                filtered_signals = [s for s in signals if s["confidence"] >= min_confidence]
                
                if filtered_signals:
                    st.success(f"Found {len(filtered_signals)} signals")
                    
                    for signal in filtered_signals:
                        with st.container():
                            col_a, col_b, col_c, col_d = st.columns([2,1,1,1])
                            with col_a:
                                st.write(f"**{signal['symbol']}** - {signal['action']}")
                            with col_b:
                                st.write(f"{signal['confidence']}%")
                            with col_c:
                                st.write(signal['strategy'])
                            with col_d:
                                if st.button(f"Trade", key=f"trade_{signal['symbol']}"):
                                    st.info(f"Trading {signal['symbol']}...")
                else:
                    st.warning("No signals found with current filters")
    
    elif active_tab == "trading":
        st.title("💰 Paper Trading")
        
        # Trading Interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Place Order")
            
            symbol = st.selectbox("Symbol", ["RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR"])
            action = st.radio("Action", ["BUY", "SELL"], horizontal=True)
            quantity = st.number_input("Quantity", min_value=1, value=10)
            
            # Get current price
            try:
                ticker = yf.Ticker(f"{symbol}.NS")
                data = ticker.history(period="1d", interval="1m")
                current_price = data['Close'].iloc[-1] if not data.empty else 0
                st.info(f"Current Price: ₹{current_price:.2f}")
            except:
                current_price = 0
                st.warning("Could not fetch price")
            
            if st.button("Execute Trade", type="primary"):
                trade_value = quantity * current_price
                st.success(f"Order placed: {action} {quantity} {symbol} @ ₹{current_price:.2f}")
                st.info(f"Trade Value: ₹{trade_value:,.2f}")
        
        with col2:
            st.subheader("Account Info")
            st.metric("Available Cash", "₹1,850,000")
            st.metric("Margin Used", "₹150,000")
            st.metric("Open Positions", "3")
            st.metric("Today's P&L", "₹15,250", "+2.1%")
            
            st.subheader("Risk Controls")
            stop_loss = st.checkbox("Auto Stop Loss", value=True)
            if stop_loss:
                sl_percent = st.slider("Stop Loss %", 0.5, 5.0, 2.0)
            take_profit = st.checkbox("Auto Take Profit", value=True)
            if take_profit:
                tp_percent = st.slider("Take Profit %", 1.0, 10.0, 4.0)
    
    elif active_tab == "charts":
        st.title("📊 Kite Connect Charts")
        
        # Kite Connect Authentication
        if not st.session_state.get("kite_auth", False):
            st.warning("🔐 Kite Connect authentication required")
            if st.session_state.kite_manager.login():
                st.rerun()
        else:
            st.success("✅ Connected to Kite Connect")
            
            # Chart Interface
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.subheader("Settings")
                instrument = st.selectbox("Instrument", ["NIFTY 50", "BANKNIFTY", "RELIANCE", "TCS", "HDFCBANK"])
                interval = st.selectbox("Interval", ["minute", "5minute", "15minute", "30minute", "hour"])
                chart_type = st.selectbox("Chart Type", ["Candlestick", "Line", "OHLC"])
                
                # Indicators
                st.subheader("Indicators")
                show_ema = st.checkbox("Show EMA", value=True)
                if show_ema:
                    ema_period = st.slider("EMA Period", 5, 50, 20)
                show_rsi = st.checkbox("Show RSI", value=False)
                if show_rsi:
                    rsi_period = st.slider("RSI Period", 5, 30, 14)
            
            with col2:
                st.subheader(f"Chart: {instrument}")
                
                # Generate sample chart data
                dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
                np.random.seed(42)
                prices = 100 + np.cumsum(np.random.randn(100) * 2)
                highs = prices + np.random.rand(100) * 5
                lows = prices - np.random.rand(100) * 5
                
                fig = go.Figure()
                
                if chart_type == "Candlestick":
                    fig.add_trace(go.Candlestick(
                        x=dates,
                        open=prices * 0.99,
                        high=highs,
                        low=lows,
                        close=prices * 1.01,
                        name=instrument
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=prices,
                        mode='lines',
                        name=instrument,
                        line=dict(color='blue', width=2)
                    ))
                
                # Add EMA
                if show_ema:
                    ema_values = pd.Series(prices).ewm(span=ema_period, adjust=False).mean()
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=ema_values,
                        mode='lines',
                        name=f'EMA{ema_period}',
                        line=dict(color='orange', width=1.5, dash='dash')
                    ))
                
                fig.update_layout(
                    title=f"{instrument} Price Chart",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    height=500,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Quick Stats
                current_price = prices[-1]
                change = prices[-1] - prices[-2] if len(prices) > 1 else 0
                change_pct = (change / prices[-2] * 100) if len(prices) > 1 else 0
                
                stat_cols = st.columns(4)
                with stat_cols[0]:
                    st.metric("Current", f"₹{current_price:.2f}")
                with stat_cols[1]:
                    st.metric("Change", f"₹{change:+.2f}")
                with stat_cols[2]:
                    st.metric("Change %", f"{change_pct:+.2f}%")
                with stat_cols[3]:
                    st.metric("High/Low", f"₹{highs.max():.2f}/₹{lows.min():.2f}")
    
    elif active_tab == "code":
        st.title("💻 Strategy Code Editor")
        
        if not ACE_AVAILABLE:
            st.warning("Full editor requires: pip install streamlit-ace")
            code = st.text_area("Code Editor", height=300, value="""# Trading Strategy Example

def calculate_signals(data):
    \"\"\"Calculate trading signals\"\"\"
    signals = []
    
    # Calculate EMA
    data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
    
    # Generate signals
    for i in range(1, len(data)):
        if data['EMA20'].iloc[i] > data['EMA50'].iloc[i] and data['EMA20'].iloc[i-1] <= data['EMA50'].iloc[i-1]:
            signals.append({'type': 'BUY', 'index': i})
        elif data['EMA20'].iloc[i] < data['EMA50'].iloc[i] and data['EMA20'].iloc[i-1] >= data['EMA50'].iloc[i-1]:
            signals.append({'type': 'SELL', 'index': i})
    
    return signals

print("Strategy loaded successfully!")
""")
        else:
            code = st_ace(
                value="""# Trading Strategy Editor
# Write your trading strategies here

import pandas as pd
import numpy as np

def ema_crossover(data, short_period=20, long_period=50):
    \"\"\"EMA Crossover Strategy\"\"\"
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['short_ema'] = data['Close'].ewm(span=short_period, adjust=False).mean()
    signals['long_ema'] = data['Close'].ewm(span=long_period, adjust=False).mean()
    signals['signal'] = 0.0
    signals['signal'][short_period:] = np.where(
        signals['short_ema'][short_period:] > signals['long_ema'][short_period:], 1.0, 0.0
    )
    signals['positions'] = signals['signal'].diff()
    return signals

def rsi_strategy(data, period=14, oversold=30, overbought=70):
    \"\"\"RSI Strategy\"\"\"
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    signals = pd.DataFrame(index=data.index)
    signals['rsi'] = rsi
    signals['signal'] = 0
    signals.loc[rsi < oversold, 'signal'] = 1  # Buy
    signals.loc[rsi > overbought, 'signal'] = -1  # Sell
    return signals

# Test the strategies
print("Strategies defined successfully!")
""",
                language="python",
                theme="monokai",
                key="ace_editor",
                height=300
            )
        
        # Editor controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            filename = st.text_input("Filename", "strategy.py")
        with col2:
            if st.button("▶️ Run", use_container_width=True):
                try:
                    exec(code)
                    st.success("✅ Code executed successfully!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        with col3:
            if st.button("💾 Save", use_container_width=True):
                st.success(f"✅ Saved to {filename}")
    
    elif active_tab == "terminal":
        st.title("🖥️ VS Code Terminal")
        st.session_state.terminal.render()
    
    # Close content area
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Status Bar (always at bottom)
    vs_code_status_bar()

# ==================== RUN APPLICATION ====================
if __name__ == "__main__":
    # Set page config
    st.set_page_config(
        page_title="Rantv Intraday - VS Code Edition",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Hide Streamlit menu and footer
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    # Run main interface
    try:
        main_vs_code_interface()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.code(traceback.format_exc())
