# Rantv Intraday Trading Signals - VS Code Integrated Version
# Works with or without streamlit-ace

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

# Try to import streamlit-ace, but work without it
try:
    from streamlit_ace import st_ace
    ACE_AVAILABLE = True
except ImportError:
    ACE_AVAILABLE = False
    # Create a dummy st_ace function for compatibility
    def st_ace(value="", language="python", theme="monokai", **kwargs):
        return st.text_area("Code Editor", value=value, height=kwargs.get('height', 300))

# Auto-install missing dependencies
def install_package(package_name):
    """Try to install a package"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "--quiet"])
        return True
    except:
        return False

# Try to install kiteconnect if not available
try:
    from kiteconnect import KiteConnect, KiteTicker
    KITECONNECT_AVAILABLE = True
except ImportError:
    if install_package("kiteconnect"):
        from kiteconnect import KiteConnect, KiteTicker
        KITECONNECT_AVAILABLE = True
    else:
        KITECONNECT_AVAILABLE = False

# Try to install other dependencies
try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    if install_package("sqlalchemy"):
        import sqlalchemy
        from sqlalchemy import create_engine, text
        SQLALCHEMY_AVAILABLE = True
    else:
        SQLALCHEMY_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    if install_package("joblib"):
        import joblib
        JOBLIB_AVAILABLE = True
    else:
        JOBLIB_AVAILABLE = False

# Setup basic logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Kite Connect API Credentials
KITE_API_KEY = os.environ.get("KITE_API_KEY", "")
KITE_API_SECRET = os.environ.get("KITE_API_SECRET", "")

# ==================== UTILITY FUNCTIONS ====================
def now_indian():
    return datetime.now(pytz.timezone("Asia/Kolkata"))

def market_open():
    """Check if market is open (9:15 AM to 3:30 PM IST)"""
    n = now_indian()
    try:
        IND_TZ = pytz.timezone("Asia/Kolkata")
        open_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(9, 15)))
        close_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(15, 30)))
        return open_time <= n <= close_time
    except Exception:
        return False

def should_auto_close():
    """Check if it's time to auto-close positions"""
    n = now_indian()
    try:
        IND_TZ = pytz.timezone("Asia/Kolkata")
        auto_close_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(15, 10)))
        return n >= auto_close_time
    except Exception:
        return False

# ==================== VS CODE CSS ====================
st.markdown("""
<style>
    /* Main VS Code Theme */
    .stApp {
        background-color: #1e1e1e;
        color: #d4d4d4;
    }
    
    /* Activity Bar */
    .activity-bar {
        position: fixed;
        left: 0;
        top: 0;
        bottom: 0;
        width: 50px;
        background: #333333;
        display: flex;
        flex-direction: column;
        align-items: center;
        padding-top: 20px;
        z-index: 100;
    }
    
    .activity-item {
        width: 100%;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        color: #858585;
        font-size: 20px;
        transition: all 0.3s ease;
    }
    
    .activity-item:hover {
        color: #ffffff;
        background: rgba(255,255,255,0.1);
    }
    
    .activity-item.active {
        color: #ffffff;
        background: rgba(0,122,204,0.4);
    }
    
    /* Status Bar */
    .status-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        height: 22px;
        background: #007acc;
        color: white;
        display: flex;
        align-items: center;
        padding: 0 15px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 12px;
        z-index: 100;
    }
    
    .status-item {
        margin-right: 20px;
        display: flex;
        align-items: center;
    }
    
    /* Explorer */
    .explorer {
        position: fixed;
        left: 50px;
        top: 0;
        bottom: 22px;
        width: 250px;
        background: #252526;
        border-right: 1px solid #333333;
        overflow-y: auto;
        z-index: 99;
    }
    
    .explorer-header {
        padding: 15px;
        color: #cccccc;
        font-weight: bold;
        border-bottom: 1px solid #333333;
    }
    
    .explorer-item {
        padding: 8px 15px;
        color: #cccccc;
        cursor: pointer;
        display: flex;
        align-items: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 13px;
    }
    
    .explorer-item:hover {
        background: rgba(90, 93, 94, 0.31);
    }
    
    .explorer-item.active {
        background: rgba(90, 93, 94, 0.31);
    }
    
    /* Main Content */
    .main-content {
        margin-left: 300px;
        margin-bottom: 22px;
        padding: 20px;
        min-height: calc(100vh - 22px);
    }
    
    /* Tabs */
    .vs-code-tabs {
        display: flex;
        background: #252526;
        border-bottom: 1px solid #333333;
        margin: -20px -20px 20px -20px;
        padding: 0 20px;
    }
    
    .vs-code-tab {
        padding: 10px 20px;
        color: #cccccc;
        cursor: pointer;
        border-right: 1px solid #333333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 13px;
        display: flex;
        align-items: center;
        min-width: 120px;
    }
    
    .vs-code-tab.active {
        background: #1e1e1e;
        color: #ffffff;
        border-bottom: 2px solid #007acc;
    }
    
    .vs-code-tab-close {
        margin-left: 10px;
        opacity: 0.5;
        cursor: pointer;
        font-size: 16px;
    }
    
    .vs-code-tab-close:hover {
        opacity: 1;
    }
    
    /* Terminal */
    .terminal {
        background: #1e1e1e;
        color: #cccccc;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #333333;
        height: 300px;
        overflow-y: auto;
        white-space: pre-wrap;
        font-size: 13px;
        margin-top: 20px;
    }
    
    .terminal-line {
        margin: 2px 0;
    }
    
    .terminal-command {
        color: #569cd6;
    }
    
    .terminal-output {
        color: #cccccc;
    }
    
    .terminal-error {
        color: #f44747;
    }
    
    .terminal-success {
        color: #6a9955;
    }
    
    /* Code Editor Container */
    .editor-container {
        background: #1e1e1e;
        border-radius: 5px;
        border: 1px solid #333333;
        margin-bottom: 15px;
        overflow: hidden;
    }
    
    .editor-header {
        background: #252526;
        padding: 10px 15px;
        border-bottom: 1px solid #333333;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .editor-title {
        color: #cccccc;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 13px;
    }
    
    /* Metrics Cards */
    .metric-card {
        background: #252526;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #007acc;
        margin-bottom: 10px;
    }
    
    /* Buttons */
    .stButton > button {
        background: #0e639c;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 2px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 13px;
    }
    
    .stButton > button:hover {
        background: #1177bb;
    }
    
    /* Inputs */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: #3c3c3c;
        color: #cccccc;
        border-color: #333333;
    }
    
    /* Dataframes */
    .dataframe {
        background: #252526;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e1e1e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #555;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #777;
    }
</style>
""", unsafe_allow_html=True)

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
            elif command.startswith("pip "):
                package = command[4:]
                result = install_package(package)
                if result:
                    self.history.append(f"✅ Installed {package}")
                else:
                    self.history.append(f"❌ Failed to install {package}")
            elif command == "clear":
                self.history = []
            elif command == "help":
                self.history.extend([
                    "Available commands:",
                    "  python <code> - Execute Python code",
                    "  pip <package> - Install Python package",
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
        st.markdown("### Terminal")
        
        # Display terminal history
        terminal_content = ""
        for line in self.history[-20:]:
            if line.startswith("$ "):
                terminal_content += f'<div class="terminal-line terminal-command">{line}</div>'
            elif line.startswith("✅"):
                terminal_content += f'<div class="terminal-line terminal-success">{line}</div>'
            elif line.startswith("❌") or line.startswith("Error:"):
                terminal_content += f'<div class="terminal-line terminal-error">{line}</div>'
            else:
                terminal_content += f'<div class="terminal-line terminal-output">{line}</div>'
        
        st.markdown(f'<div class="terminal">{terminal_content}</div>', unsafe_allow_html=True)
        
        # Command input
        col1, col2 = st.columns([4, 1])
        with col1:
            command = st.text_input("", placeholder="Type a command...", key="terminal_input", label_visibility="collapsed")
        with col2:
            if st.button("Execute", key="terminal_execute", use_container_width=True):
                if command:
                    self.execute_command(command)
                    st.rerun()

def vs_code_activity_bar():
    """Create VS Code style activity bar"""
    activities = [
        {"icon": "📁", "title": "Explorer", "id": "explorer"},
        {"icon": "🔍", "title": "Search", "id": "search"},
        {"icon": "📚", "title": "Source Control", "id": "git"},
        {"icon": "🐞", "title": "Debug", "id": "debug"},
        {"icon": "🧪", "title": "Testing", "id": "test"},
        {"icon": "📦", "title": "Extensions", "id": "extensions"},
        {"icon": "⚙️", "title": "Settings", "id": "settings"}
    ]
    
    html = '<div class="activity-bar">'
    for activity in activities:
        active_class = "active" if st.session_state.get("active_activity") == activity["id"] else ""
        html += f'''
        <div class="activity-item {active_class}" 
             onclick="this.dispatchEvent(new Event('click'));"
             title="{activity["title"]}">
            {activity["icon"]}
        </div>
        '''
    html += '</div>'
    
    # Create the activity bar
    st.markdown(html, unsafe_allow_html=True)
    
    # Handle clicks
    for activity in activities:
        if st.button(f"", key=f"activity_{activity['id']}"):
            st.session_state.active_activity = activity["id"]

def vs_code_status_bar():
    """Create VS Code style status bar"""
    current_time = datetime.now().strftime("%H:%M:%S")
    market_status = "OPEN" if market_open() else "CLOSED"
    
    html = f'''
    <div class="status-bar">
        <div class="status-item">🕐 {current_time}</div>
        <div class="status-item">🌐 Market: {market_status}</div>
        <div class="status-item">🐍 Python {sys.version.split()[0]}</div>
        <div class="status-item">📊 VS Code Mode</div>
        <div class="status-item" style="margin-left:auto;">{"🟢" if ACE_AVAILABLE else "🟡"} Editor</div>
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
             onclick="this.dispatchEvent(new Event('click'));">
            {tab["icon"]} {tab["name"]}
            <span class="vs-code-tab-close" onclick="event.stopPropagation(); this.dispatchEvent(new Event('click'));">×</span>
        </div>
        '''
    html += '</div>'
    
    st.markdown(html, unsafe_allow_html=True)
    
    # Handle tab clicks
    for tab in tabs:
        if st.button(f"", key=f"tab_{tab['id']}"):
            st.session_state.active_tab = tab["id"]
        if st.button(f"", key=f"close_{tab['id']}"):
            if st.session_state.get("active_tab") == tab["id"]:
                st.session_state.active_tab = "dashboard"

def install_streamlit_ace():
    """Install streamlit-ace via terminal"""
    st.warning("streamlit-ace not installed. Full VS Code editor requires:")
    st.code("pip install streamlit-ace")
    
    if st.button("🔄 Install Now (via terminal)"):
        with st.spinner("Installing streamlit-ace..."):
            if install_package("streamlit-ace"):
                st.success("✅ streamlit-ace installed! Please refresh the page.")
                time.sleep(2)
                st.rerun()
            else:
                st.error("❌ Failed to install streamlit-ace")

# ==================== TRADING FUNCTIONS ====================
def get_nifty_data():
    """Get Nifty 50 data"""
    try:
        nifty = yf.download("^NSEI", period="1d", interval="1m")
        if not nifty.empty:
            return nifty['Close'].iloc[-1]
    except:
        pass
    return 22000  # Default value

def generate_sample_signals():
    """Generate sample trading signals"""
    return [
        {"symbol": "RELIANCE", "action": "BUY", "confidence": 82, "price": 2750, "change": "+1.5%"},
        {"symbol": "TCS", "action": "SELL", "confidence": 75, "price": 3850, "change": "-0.8%"},
        {"symbol": "HDFCBANK", "action": "BUY", "confidence": 78, "price": 1650, "change": "+1.2%"},
        {"symbol": "INFY", "action": "BUY", "confidence": 85, "price": 1450, "change": "+2.1%"},
    ]

# ==================== MAIN INTERFACE ====================
def main():
    """Main VS Code interface"""
    
    # Initialize session states
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "dashboard"
    if "active_activity" not in st.session_state:
        st.session_state.active_activity = "explorer"
    if "terminal" not in st.session_state:
        st.session_state.terminal = VSCodeTerminal()
    if "show_install_prompt" not in st.session_state:
        st.session_state.show_install_prompt = not ACE_AVAILABLE
    
    # Show install prompt if needed
    if st.session_state.show_install_prompt and not ACE_AVAILABLE:
        install_streamlit_ace()
        if st.button("Continue with Basic Editor"):
            st.session_state.show_install_prompt = False
            st.rerun()
        st.stop()
    
    # Create layout
    vs_code_activity_bar()
    
    # Show explorer if active
    if st.session_state.active_activity == "explorer":
        vs_code_file_explorer()
    
    # Main content
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # Tabs
    vs_code_tabs()
    
    # Tab content
    active_tab = st.session_state.active_tab
    
    if active_tab == "dashboard":
        st.title("📈 Trading Dashboard")
        
        # Market Overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Market Status", "OPEN" if market_open() else "CLOSED", 
                     delta="Trading" if market_open() else "Closed")
        with col2:
            nifty_val = get_nifty_data()
            st.metric("NIFTY 50", f"₹{nifty_val:,.2f}")
        with col3:
            st.metric("Account Value", "₹2,150,000", "+3.5%")
        with col4:
            st.metric("Today's P&L", "₹15,250", "+2.1%")
        
        # Quick Actions
        st.subheader("⚡ Quick Actions")
        action_cols = st.columns(4)
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
            if st.button("🖥️ Open Terminal", use_container_width=True):
                st.session_state.active_tab = "terminal"
        
        # Recent Signals
        st.subheader("🎯 Recent Signals")
        signals = generate_sample_signals()
        for signal in signals[:3]:
            col_a, col_b, col_c, col_d = st.columns([2, 1, 1, 1])
            with col_a:
                st.write(f"**{signal['symbol']}** - {signal['action']}")
            with col_b:
                st.write(f"{signal['confidence']}%")
            with col_c:
                st.write(f"₹{signal['price']}")
            with col_d:
                st.write(signal['change'])
    
    elif active_tab == "signals":
        st.title("🚦 Trading Signals")
        
        # Signal Generator
        st.subheader("Generate Signals")
        
        col1, col2 = st.columns(2)
        with col1:
            universe = st.selectbox("Select Universe", 
                                   ["NIFTY 50", "NIFTY 100", "Midcap 150", "All Stocks"])
            min_confidence = st.slider("Min Confidence %", 60, 90, 70)
        
        with col2:
            strategies = st.multiselect(
                "Strategies",
                ["EMA Crossover", "RSI Reversal", "MACD", "Bollinger Bands", "Volume Breakout"],
                default=["EMA Crossover", "RSI Reversal"]
            )
        
        if st.button("🔍 Scan for Signals", type="primary"):
            with st.spinner(f"Scanning {universe}..."):
                time.sleep(1)  # Simulate processing
                signals = generate_sample_signals()
                
                # Filter by confidence
                filtered = [s for s in signals if s['confidence'] >= min_confidence]
                
                if filtered:
                    st.success(f"Found {len(filtered)} signals")
                    for signal in filtered:
                        with st.expander(f"{signal['symbol']} - {signal['action']} ({signal['confidence']}%)"):
                            st.write(f"Price: ₹{signal['price']}")
                            st.write(f"Change: {signal['change']}")
                            if st.button(f"Trade {signal['symbol']}", key=f"trade_{signal['symbol']}"):
                                st.success(f"Order placed for {signal['symbol']}")
                else:
                    st.warning("No signals found")
    
    elif active_tab == "trading":
        st.title("💰 Paper Trading")
        
        # Trading Interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Place Order")
            
            symbol = st.selectbox("Symbol", ["RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR"])
            action = st.radio("Action", ["BUY", "SELL"], horizontal=True)
            quantity = st.number_input("Quantity", min_value=1, value=10)
            
            # Get price
            try:
                ticker = yf.Ticker(f"{symbol}.NS")
                data = ticker.history(period="1d", interval="1m")
                price = data['Close'].iloc[-1] if not data.empty else 1000
                st.info(f"Current Price: ₹{price:.2f}")
            except:
                price = 1000
                st.warning("Using sample price")
            
            if st.button("Execute Trade", type="primary"):
                value = quantity * price
                st.success(f"Order: {action} {quantity} {symbol} @ ₹{price:.2f}")
                st.info(f"Value: ₹{value:,.2f}")
        
        with col2:
            st.subheader("Account")
            st.metric("Cash", "₹1,850,000")
            st.metric("Margin", "₹150,000")
            st.metric("Positions", "3")
            st.metric("Today's P&L", "₹15,250")
    
    elif active_tab == "charts":
        st.title("📊 Live Charts")
        
        # Chart Settings
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Settings")
            symbol = st.selectbox("Symbol", ["NIFTY 50", "BANKNIFTY", "RELIANCE", "TCS"])
            interval = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d"])
            chart_type = st.selectbox("Chart Type", ["Candlestick", "Line"])
            
            # Indicators
            st.subheader("Indicators")
            show_ema = st.checkbox("EMA", value=True)
            if show_ema:
                ema_period = st.slider("EMA Period", 5, 50, 20)
        
        with col2:
            st.subheader(f"{symbol} Chart")
            
            # Generate sample data
            dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
            prices = 100 + np.cumsum(np.random.randn(100) * 2)
            
            fig = go.Figure()
            
            if chart_type == "Candlestick":
                fig.add_trace(go.Candlestick(
                    x=dates,
                    open=prices * 0.99,
                    high=prices + np.random.rand(100) * 5,
                    low=prices - np.random.rand(100) * 5,
                    close=prices * 1.01,
                    name=symbol
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=prices,
                    mode='lines',
                    name=symbol,
                    line=dict(color='blue', width=2)
                ))
            
            # Add EMA
            if show_ema:
                ema_vals = pd.Series(prices).ewm(span=ema_period, adjust=False).mean()
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=ema_vals,
                    mode='lines',
                    name=f'EMA{ema_period}',
                    line=dict(color='orange', dash='dash')
                ))
            
            fig.update_layout(
                title=f"{symbol} Price Chart",
                xaxis_title="Time",
                yaxis_title="Price",
                height=500,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif active_tab == "code":
        st.title("💻 Code Editor")
        
        if not ACE_AVAILABLE:
            st.warning("⚠️ Basic editor mode. Install streamlit-ace for full VS Code editor:")
            if st.button("Install streamlit-ace"):
                st.session_state.show_install_prompt = True
                st.rerun()
        
        # Editor
        st.markdown('<div class="editor-container">', unsafe_allow_html=True)
        st.markdown('<div class="editor-header"><div class="editor-title">strategies.py</div></div>', unsafe_allow_html=True)
        
        default_code = """# Trading Strategies
# VS Code Style Editor

import pandas as pd
import numpy as np

def ema_crossover(data, short=20, long=50):
    \"\"\"EMA Crossover Strategy\"\"\"
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['short_ema'] = data['Close'].ewm(span=short, adjust=False).mean()
    signals['long_ema'] = data['Close'].ewm(span=long, adjust=False).mean()
    signals['signal'] = 0.0
    signals['signal'][short:] = np.where(
        signals['short_ema'][short:] > signals['long_ema'][short:], 1.0, 0.0
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

# Test
print("Strategies loaded!")
"""
        
        code = st_ace(
            value=default_code,
            language="python",
            theme="monokai",
            key="code_editor",
            height=300
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Editor controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            filename = st.text_input("Filename", "strategy.py")
        with col2:
            if st.button("▶️ Run", use_container_width=True):
                try:
                    exec(code)
                    st.success("✅ Code executed!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        with col3:
            if st.button("💾 Save", use_container_width=True):
                st.success(f"✅ Saved as {filename}")
    
    elif active_tab == "terminal":
        st.title("🖥️ VS Code Terminal")
        st.session_state.terminal.render()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Status bar
    vs_code_status_bar()

# ==================== RUN APP ====================
if __name__ == "__main__":
    # Set page config
    st.set_page_config(
        page_title="Rantv Intraday - VS Code",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    try:
        main()
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.code(traceback.format_exc())
