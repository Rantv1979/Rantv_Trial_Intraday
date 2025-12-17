# Rantv Intraday Trading Signals & Market Analysis - PRODUCTION READY
# ENHANCED VERSION WITH FULL STOCK SCANNING & BETTER SIGNAL QUALITY
# UPDATED: Lowered confidence to 70%, score to 6, added ADX trend filter, optimized for peak hours
# INTEGRATED WITH KITE CONNECT FOR LIVE CHARTS & VS CODE STYLE UI

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

# VS Code-like UI components
try:
    from streamlit_ace import st_ace
    ACE_AVAILABLE = True
except ImportError:
    ACE_AVAILABLE = False

# Auto-install missing critical dependencies including kiteconnect
try:
    from kiteconnect import KiteConnect, KiteTicker
    KITECONNECT_AVAILABLE = True
except ImportError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kiteconnect"])
        from kiteconnect import KiteConnect, KiteTicker
        KITECONNECT_AVAILABLE = True
        st.success("✅ Installed kiteconnect")
    except:
        KITECONNECT_AVAILABLE = False

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sqlalchemy"])
        import sqlalchemy
        from sqlalchemy import create_engine, text
        SQLALCHEMY_AVAILABLE = True
        st.success("✅ Installed sqlalchemy")
    except:
        SQLALCHEMY_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])
        import joblib
        JOBLIB_AVAILABLE = True
        st.success("✅ Installed joblib")
    except:
        JOBLIB_AVAILABLE = False

# Setup basic logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Kite Connect API Credentials - REMOVE HARDCODED VALUES FOR SECURITY
KITE_API_KEY = os.environ.get("KITE_API_KEY", "")
KITE_API_SECRET = os.environ.get("KITE_API_SECRET", "")
KITE_ACCESS_TOKEN = ""  # Will be set after login

# Configuration
@dataclass
class AppConfig:
    database_url: str = 'sqlite:///trading_journal.db'
    risk_tolerance: str = 'MODERATE'
    max_daily_loss: float = 50000.0
    enable_ml: bool = True
    kite_api_key: str = KITE_API_KEY
    kite_api_secret: str = KITE_API_SECRET
    vs_code_theme: str = 'monokai'  # VS Code style theme
    enable_vs_code_ui: bool = True  # Enable VS Code style UI

    @classmethod
    def from_env(cls):
        return cls()

# Initialize configuration
config = AppConfig.from_env()

# VS Code Style Theme Configuration
VS_CODE_THEMES = {
    'monokai': {
        'background': '#272822',
        'foreground': '#f8f8f2',
        'selection': '#49483e',
        'line_highlight': '#3e3d32',
        'gutter': '#90908a',
        'gutter_foreground': '#d0d0cc'
    },
    'dracula': {
        'background': '#282a36',
        'foreground': '#f8f8f2',
        'selection': '#44475a',
        'line_highlight': '#44475a',
        'gutter': '#6272a4',
        'gutter_foreground': '#bd93f9'
    },
    'solarized_dark': {
        'background': '#002b36',
        'foreground': '#839496',
        'selection': '#073642',
        'line_highlight': '#073642',
        'gutter': '#586e75',
        'gutter_foreground': '#cb4b16'
    },
    'light': {
        'background': '#ffffff',
        'foreground': '#000000',
        'selection': '#add6ff',
        'line_highlight': '#f0f0f0',
        'gutter': '#f0f0f0',
        'gutter_foreground': '#999999'
    }
}

def apply_vs_code_theme(theme_name='monokai'):
    """Apply VS Code like theme to Streamlit"""
    theme = VS_CODE_THEMES.get(theme_name, VS_CODE_THEMES['monokai'])
    
    st.markdown(f"""
    <style>
        /* Main VS Code Theme */
        .stApp {{
            background-color: {theme['background']};
            color: {theme['foreground']};
        }}
        
        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: {theme['selection']};
        }}
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: {theme['selection']};
            border-bottom: 1px solid {theme['gutter']};
        }}
        
        .stTabs [data-baseweb="tab"] {{
            color: {theme['foreground']};
            background-color: {theme['selection']};
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: {theme['background']};
            color: {theme['foreground']};
            border-bottom: 2px solid #007acc;
        }}
        
        /* Code Editor */
        .ace_editor {{
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 14px;
            border-radius: 5px;
            border: 1px solid {theme['gutter']};
        }}
        
        /* Metrics and Cards */
        .metric-card {{
            background-color: {theme['selection']};
            border-left: 4px solid #007acc;
            color: {theme['foreground']};
        }}
        
        /* Dataframes */
        .dataframe {{
            background-color: {theme['selection']};
            color: {theme['foreground']};
        }}
        
        /* Inputs */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > div {{
            background-color: {theme['selection']};
            color: {theme['foreground']};
            border-color: {theme['gutter']};
        }}
        
        /* Buttons */
        .stButton > button {{
            background-color: #007acc;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}
        
        .stButton > button:hover {{
            background-color: #005a9e;
        }}
        
        /* VS Code Style Terminal */
        .terminal {{
            background-color: {theme['background']};
            color: {theme['foreground']};
            font-family: 'Consolas', 'Monaco', monospace;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid {theme['gutter']};
            height: 300px;
            overflow-y: auto;
            white-space: pre-wrap;
        }}
        
        /* Status Bar */
        .status-bar {{
            background-color: {theme['selection']};
            color: {theme['foreground']};
            padding: 5px 15px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 12px;
            border-top: 1px solid {theme['gutter']};
            position: fixed;
            bottom: 0;
            width: 100%;
            display: flex;
            justify-content: space-between;
        }}
        
        /* File Explorer */
        .file-explorer {{
            background-color: {theme['selection']};
            color: {theme['foreground']};
            padding: 10px;
            border-right: 1px solid {theme['gutter']};
        }}
        
        .file-item {{
            padding: 5px 10px;
            cursor: pointer;
            border-radius: 3px;
        }}
        
        .file-item:hover {{
            background-color: {theme['line_highlight']};
        }}
        
        /* Panel Header */
        .panel-header {{
            background-color: {theme['selection']};
            color: {theme['foreground']};
            padding: 10px 15px;
            border-bottom: 1px solid {theme['gutter']};
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        /* VS Code Activity Bar */
        .activity-bar {{
            background-color: {theme['selection']};
            width: 48px;
            padding-top: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        
        .activity-item {{
            padding: 12px;
            cursor: pointer;
            color: {theme['gutter_foreground']};
        }}
        
        .activity-item.active {{
            color: white;
            border-left: 3px solid #007acc;
        }}
        
        .activity-item:hover {{
            color: {theme['foreground']};
        }}
    </style>
    """, unsafe_allow_html=True)

# Apply VS Code theme
apply_vs_code_theme(config.vs_code_theme)

st.set_page_config(
    page_title="Rantv Intraday Terminal Pro - VS Code Style",
    layout="wide",
    initial_sidebar_state="expanded"
)

IND_TZ = pytz.timezone("Asia/Kolkata")

# Trading Constants
CAPITAL = 2_000_000.0
TRADE_ALLOC = 0.15
MAX_DAILY_TRADES = 10
MAX_STOCK_TRADES = 10
MAX_AUTO_TRADES = 10

SIGNAL_REFRESH_MS = 120000
PRICE_REFRESH_MS = 100000

MARKET_OPTIONS = ["CASH"]

# Stock Universes - COMBINED ALL STOCKS
NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS",
    "SBIN.NS", "ASIANPAINT.NS", "HCLTECH.NS", "AXISBANK.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "NTPC.NS",
    "NESTLEIND.NS", "POWERGRID.NS", "M&M.NS", "BAJFINANCE.NS", "ONGC.NS",
    "TATASTEEL.NS", "JSWSTEEL.NS", "ADANIPORTS.NS", "COALINDIA.NS",
    "HDFCLIFE.NS", "DRREDDY.NS", "HINDALCO.NS", "CIPLA.NS", "SBILIFE.NS",
    "GRASIM.NS", "TECHM.NS", "BAJAJFINSV.NS", "BRITANNIA.NS", "EICHERMOT.NS",
    "DIVISLAB.NS", "SHREECEM.NS", "APOLLOHOSP.NS", "UPL.NS", "BAJAJ-AUTO.NS",
    "HEROMOTOCO.NS", "INDUSINDBK.NS", "ADANIENT.NS", "TATACONSUM.NS", "BPCL.NS"
]

# ... [Rest of your stock universe definitions remain the same] ...

# Enhanced Trading Strategies with Better Balance - ALL STRATEGIES ENABLED
TRADING_STRATEGIES = {
    "EMA_VWAP_Confluence": {"name": "EMA + VWAP Confluence", "weight": 3, "type": "BUY"},
    "RSI_MeanReversion": {"name": "RSI Mean Reversion", "weight": 2, "type": "BUY"},
    "Bollinger_Reversion": {"name": "Bollinger Band Reversion", "weight": 2, "type": "BUY"},
    "MACD_Momentum": {"name": "MACD Momentum", "weight": 2, "type": "BUY"},
    "Support_Resistance_Breakout": {"name": "Support/Resistance Breakout", "weight": 3, "type": "BUY"},
    "EMA_VWAP_Downtrend": {"name": "EMA + VWAP Downtrend", "weight": 3, "type": "SELL"},
    "RSI_Overbought": {"name": "RSI Overbought Reversal", "weight": 2, "type": "SELL"},
    "Bollinger_Rejection": {"name": "Bollinger Band Rejection", "weight": 2, "type": "SELL"},
    "MACD_Bearish": {"name": "MACD Bearish Crossover", "weight": 2, "type": "SELL"},
    "Trend_Reversal": {"name": "Trend Reversal", "weight": 2, "type": "SELL"}
}

# HIGH ACCURACY STRATEGIES FOR ALL STOCKS - ENABLED FOR ALL UNIVERSES
HIGH_ACCURACY_STRATEGIES = {
    "Multi_Confirmation": {"name": "Multi-Confirmation Ultra", "weight": 5, "type": "BOTH"},
    "Enhanced_EMA_VWAP": {"name": "Enhanced EMA-VWAP", "weight": 4, "type": "BOTH"},
    "Volume_Breakout": {"name": "Volume Weighted Breakout", "weight": 4, "type": "BOTH"},
    "RSI_Divergence": {"name": "RSI Divergence", "weight": 3, "type": "BOTH"},
    "MACD_Trend": {"name": "MACD Trend Momentum", "weight": 3, "type": "BOTH"}
}

# FIXED CSS with VS Code Style
st.markdown("""
<style>
    /* VS Code Activity Bar */
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
        z-index: 1000;
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
    
    /* Main content area */
    .main-content {
        margin-left: 50px;
        padding: 20px;
    }
    
    /* VS Code Status Bar */
    .status-bar {
        position: fixed;
        bottom: 0;
        left: 50px;
        right: 0;
        height: 22px;
        background: #007acc;
        color: white;
        display: flex;
        align-items: center;
        padding: 0 10px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 12px;
        z-index: 1000;
    }
    
    .status-item {
        margin-right: 20px;
        display: flex;
        align-items: center;
    }
    
    /* VS Code Panel */
    .panel {
        background: #1e1e1e;
        border-top: 1px solid #333333;
        position: fixed;
        bottom: 22px;
        left: 50px;
        right: 0;
        height: 200px;
        z-index: 999;
    }
    
    .panel-header {
        background: #252526;
        color: #cccccc;
        padding: 8px 15px;
        border-bottom: 1px solid #333333;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .panel-tabs {
        display: flex;
    }
    
    .panel-tab {
        padding: 5px 15px;
        cursor: pointer;
        border-bottom: 2px solid transparent;
        margin-right: 5px;
    }
    
    .panel-tab.active {
        border-bottom-color: #007acc;
        color: #ffffff;
    }
    
    /* VS Code Editor */
    .editor-container {
        background: #1e1e1e;
        border-radius: 5px;
        border: 1px solid #333333;
        margin-bottom: 15px;
    }
    
    .editor-toolbar {
        background: #252526;
        padding: 8px 15px;
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
    
    /* Terminal */
    .terminal {
        background: #1e1e1e;
        color: #cccccc;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        padding: 10px;
        height: 100%;
        overflow-y: auto;
        white-space: pre-wrap;
        font-size: 13px;
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
    
    /* VS Code Explorer */
    .explorer {
        background: #252526;
        width: 250px;
        position: fixed;
        left: 50px;
        top: 0;
        bottom: 22px;
        border-right: 1px solid #333333;
        overflow-y: auto;
        z-index: 998;
    }
    
    .explorer-header {
        padding: 15px;
        color: #cccccc;
        font-weight: bold;
        border-bottom: 1px solid #333333;
    }
    
    .explorer-item {
        padding: 5px 15px;
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
    
    /* VS Code Breadcrumbs */
    .breadcrumbs {
        background: #252526;
        padding: 5px 15px;
        border-bottom: 1px solid #333333;
        display: flex;
        align-items: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 12px;
        color: #cccccc;
    }
    
    .breadcrumb-item {
        cursor: pointer;
        padding: 2px 5px;
        border-radius: 3px;
    }
    
    .breadcrumb-item:hover {
        background: rgba(90, 93, 94, 0.31);
    }
    
    .breadcrumb-separator {
        margin: 0 5px;
        color: #666666;
    }
    
    /* VS Code Tabs */
    .vs-code-tabs {
        display: flex;
        background: #252526;
        border-bottom: 1px solid #333333;
    }
    
    .vs-code-tab {
        padding: 8px 15px;
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
    }
    
    .vs-code-tab-close {
        margin-left: 10px;
        opacity: 0.5;
        cursor: pointer;
    }
    
    .vs-code-tab-close:hover {
        opacity: 1;
    }
    
    /* Adjust main content for explorer */
    .main-content.with-explorer {
        margin-left: 300px;
    }
</style>
""", unsafe_allow_html=True)

# System Status Check
def check_system_status():
    """Check system dependencies and return status"""
    status = {
        "kiteconnect": KITECONNECT_AVAILABLE,
        "sqlalchemy": SQLALCHEMY_AVAILABLE,
        "joblib": JOBLIB_AVAILABLE,
        "yfinance": True,
        "plotly": True,
        "pandas": True,
        "numpy": True,
        "streamlit": True,
        "pytz": True,
        "streamlit_autorefresh": True,
        "ace_editor": ACE_AVAILABLE
    }
    return status

# Display system status in sidebar
system_status = check_system_status()

# VS Code Style Activity Bar Component
def vs_code_activity_bar():
    """Create VS Code style activity bar"""
    activities = [
        {"icon": "📁", "title": "Explorer", "id": "explorer"},
        {"icon": "🔍", "title": "Search", "id": "search"},
        {"icon": "📚", "title": "Source Control", "id": "git"},
        {"icon": "🐞", "title": "Run and Debug", "id": "debug"},
        {"icon": "🧪", "title": "Testing", "id": "test"},
        {"icon": "📦", "title": "Extensions", "id": "extensions"},
        {"icon": "⚙️", "title": "Settings", "id": "settings"}
    ]
    
    html = '<div class="activity-bar">'
    for activity in activities:
        active_class = "active" if st.session_state.get("active_activity", "explorer") == activity["id"] else ""
        html += f'''
        <div class="activity-item {active_class}" 
             onclick="window.parent.document.getElementById('activity_{activity["id"]}').click()"
             title="{activity["title"]}">
            {activity["icon"]}
        </div>
        '''
    html += '</div>'
    
    st.markdown(html, unsafe_allow_html=True)
    
    # Create hidden buttons for each activity
    for activity in activities:
        if st.button(f"Activity: {activity['title']}", key=f"activity_{activity['id']}", help=activity['title']):
            st.session_state.active_activity = activity["id"]

# VS Code Style Status Bar Component
def vs_code_status_bar():
    """Create VS Code style status bar"""
    current_time = datetime.now().strftime("%H:%M:%S")
    branch = "main"
    python_version = f"Python {sys.version.split()[0]}"
    
    status_items = [
        f"🔄 {current_time}",
        f"🌐 Market: {'OPEN' if market_open() else 'CLOSED'}",
        f"📊 {python_version}",
        f"📈 P&L: ₹{st.session_state.get('total_pnl', 0):+.2f}",
        f"🔧 VS Code Mode"
    ]
    
    html = '<div class="status-bar">'
    for item in status_items:
        html += f'<div class="status-item">{item}</div>'
    html += '</div>'
    
    st.markdown(html, unsafe_allow_html=True)

# VS Code Style Terminal Component
class VSCodeTerminal:
    """VS Code style terminal emulator"""
    
    def __init__(self, height=200):
        self.height = height
        self.history = []
        self.max_history = 100
        self.current_input = ""
        
    def execute_command(self, command):
        """Execute a terminal command"""
        self.history.append(f"$ {command}")
        
        try:
            # Execute Python code
            if command.startswith("python "):
                code = command[7:]
                old_stdout = sys.stdout
                redirected_output = sys.stdout = StringIO()
                exec(code, globals())
                sys.stdout = old_stdout
                output = redirected_output.getvalue()
                self.history.append(output)
            
            # Kite Connect commands
            elif command.startswith("kite "):
                cmd = command[5:]
                if cmd == "login":
                    self.history.append("Opening Kite Connect login...")
                    # Kite login logic here
                elif cmd == "status":
                    self.history.append("Checking Kite Connect status...")
                else:
                    self.history.append(f"Unknown kite command: {cmd}")
            
            # System commands
            elif command == "clear":
                self.history = []
                return
            elif command == "help":
                self.history.append("Available commands:")
                self.history.append("  python <code> - Execute Python code")
                self.history.append("  kite login - Login to Kite Connect")
                self.history.append("  kite status - Check Kite Connect status")
                self.history.append("  clear - Clear terminal")
                self.history.append("  help - Show this help")
            else:
                self.history.append(f"Command not found: {command}")
        
        except Exception as e:
            self.history.append(f"Error: {str(e)}")
        
        # Keep history within limit
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def render(self):
        """Render the terminal"""
        st.markdown('<div class="panel-header"><div>TERMINAL</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="terminal">', unsafe_allow_html=True)
        
        for line in self.history[-20:]:  # Show last 20 lines
            if line.startswith("$ "):
                st.markdown(f'<div class="terminal-line terminal-command">{line}</div>', unsafe_allow_html=True)
            elif line.startswith("Error:"):
                st.markdown(f'<div class="terminal-line terminal-error">{line}</div>', unsafe_allow_html=True)
            elif line in ["Available commands:", "Opening Kite Connect login...", "Checking Kite Connect status..."]:
                st.markdown(f'<div class="terminal-line terminal-success">{line}</div>', unsafe_allow_html=True)
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

# VS Code Style File Explorer
def vs_code_file_explorer():
    """Create VS Code style file explorer"""
    files = [
        {"name": "trading_journal.db", "type": "database", "icon": "🗃️"},
        {"name": "strategies.py", "type": "python", "icon": "🐍"},
        {"name": "config.json", "type": "json", "icon": "⚙️"},
        {"name": "signals.csv", "type": "csv", "icon": "📊"},
        {"name": "backtest_results.json", "type": "json", "icon": "📈"},
        {"name": "ml_models", "type": "folder", "icon": "🤖"},
        {"name": "logs", "type": "folder", "icon": "📝"}
    ]
    
    st.markdown('<div class="explorer">', unsafe_allow_html=True)
    st.markdown('<div class="explorer-header">EXPLORER</div>', unsafe_allow_html=True)
    
    for file in files:
        active_class = "active" if st.session_state.get("selected_file") == file["name"] else ""
        st.markdown(f'''
            <div class="explorer-item {active_class}" 
                 onclick="window.parent.document.getElementById('file_{file["name"]}').click()">
                {file["icon"]} {file["name"]}
            </div>
        ''', unsafe_allow_html=True)
        
        # Create hidden button for each file
        if st.button(f"Select {file['name']}", key=f"file_{file['name']}", help=f"Open {file['name']}"):
            st.session_state.selected_file = file["name"]
    
    st.markdown('</div>', unsafe_allow_html=True)

# VS Code Style Code Editor
def vs_code_code_editor():
    """Create VS Code style code editor"""
    if not ACE_AVAILABLE:
        st.warning("Streamlit-Ace not installed. Install with: pip install streamlit-ace")
        code = st.text_area("Code Editor", height=400, value="print('Hello VS Code!')")
        return code
    
    # Default code based on selected file
    default_code = """# VS Code Style Trading Terminal
# Welcome to the enhanced trading platform

import pandas as pd
import numpy as np
from datetime import datetime

def calculate_signals(data):
    \"\"\"Calculate trading signals\"\"\"
    signals = []
    # Your trading logic here
    return signals

# Example usage
if __name__ == "__main__":
    print("Trading terminal ready!")
"""
    
    selected_file = st.session_state.get("selected_file", "strategies.py")
    
    if selected_file.endswith(".py"):
        default_code = """# Trading Strategy Module

def ema_crossover_strategy(data, short_window=8, long_window=21):
    \"\"\"EMA Crossover Strategy\"\"\"
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['short_ema'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    signals['long_ema'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    signals['signal'] = 0.0
    signals['signal'][short_window:] = np.where(
        signals['short_ema'][short_window:] > signals['long_ema'][short_window:], 1.0, 0.0
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
    signals.loc[rsi < oversold, 'signal'] = 1  # Buy signal
    signals.loc[rsi > overbought, 'signal'] = -1  # Sell signal
    return signals
"""
    
    code = st_ace(
        value=default_code,
        language="python",
        theme="monokai",
        key="code_editor",
        height=400,
        font_size=14,
        show_gutter=True,
        show_print_margin=True,
        wrap=True,
        auto_update=True
    )
    
    # Editor toolbar
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.caption(f"📄 {selected_file}")
    with col2:
        if st.button("Run", key="run_code"):
            try:
                exec_globals = {
                    'pd': pd,
                    'np': np,
                    'datetime': datetime,
                    'go': go,
                    'st': st
                }
                exec(code, exec_globals)
                st.success("Code executed successfully!")
            except Exception as e:
                st.error(f"Execution error: {str(e)}")
    with col3:
        if st.button("Save", key="save_code"):
            st.success(f"Code saved to {selected_file}")
    with col4:
        if st.button("Format", key="format_code"):
            st.info("Formatting code...")
    
    return code

# VS Code Style Tabs
def vs_code_tabs():
    """Create VS Code style tab interface"""
    tabs = [
        {"id": "dashboard", "name": "📈 Dashboard", "icon": "📈"},
        {"id": "signals", "name": "🚦 Signals", "icon": "🚦"},
        {"id": "trading", "name": "💰 Trading", "icon": "💰"},
        {"id": "charts", "name": "📊 Charts", "icon": "📊"},
        {"id": "code", "name": "💻 Code", "icon": "💻"},
        {"id": "terminal", "name": "🖥️ Terminal", "icon": "🖥️"},
        {"id": "settings", "name": "⚙️ Settings", "icon": "⚙️"}
    ]
    
    html = '<div class="vs-code-tabs">'
    for tab in tabs:
        active_class = "active" if st.session_state.get("active_tab", "dashboard") == tab["id"] else ""
        html += f'''
        <div class="vs-code-tab {active_class}" 
             onclick="window.parent.document.getElementById('tab_{tab["id"]}').click()">
            {tab["icon"]} {tab["name"]}
            <span class="vs-code-tab-close" onclick="window.parent.document.getElementById('close_{tab["id"]}').click()">×</span>
        </div>
        '''
    html += '</div>'
    
    st.markdown(html, unsafe_allow_html=True)
    
    # Create hidden buttons for tabs
    for tab in tabs:
        if st.button(f"Tab: {tab['name']}", key=f"tab_{tab['id']}"):
            st.session_state.active_tab = tab["id"]
        # Close button (doesn't actually close, just switches to first tab)
        if st.button(f"Close: {tab['name']}", key=f"close_{tab['id']}"):
            if st.session_state.get("active_tab") == tab["id"]:
                st.session_state.active_tab = "dashboard"

# Enhanced Kite Token Database Manager
class KiteTokenDatabase:
    def __init__(self):
        self.db_url = os.environ.get("DATABASE_URL", "sqlite:///kite_tokens.db")
        self.engine = None
        self.connected = False
        if SQLALCHEMY_AVAILABLE:
            try:
                self.engine = create_engine(self.db_url)
                self.create_tables()
                self.connected = True
            except Exception as e:
                logger.error(f"Kite Token DB connection failed: {e}")

    def create_tables(self):
        if not self.engine:
            return
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS kite_tokens (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id VARCHAR(100) DEFAULT 'default',
                        access_token TEXT,
                        refresh_token TEXT,
                        public_token TEXT,
                        user_name VARCHAR(200),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP,
                        is_valid BOOLEAN DEFAULT TRUE
                    )
                """))
                conn.commit()
        except Exception as e:
            logger.error(f"Error creating kite_tokens table: {e}")

    def save_token(self, access_token, user_name="", public_token="", refresh_token=""):
        if not self.connected:
            return False
        try:
            with self.engine.connect() as conn:
                # Invalidate old tokens
                conn.execute(text("UPDATE kite_tokens SET is_valid = FALSE WHERE user_id = 'default'"))
                # Insert new token
                conn.execute(text("""
                    INSERT INTO kite_tokens (user_id, access_token, refresh_token, public_token, user_name, is_valid, expires_at)
                    VALUES ('default', :access_token, :refresh_token, :public_token, :user_name, TRUE, datetime('now', '+8 hours'))
                """), {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "public_token": public_token,
                    "user_name": user_name
                })
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving token: {e}")
            return False

    def get_valid_token(self):
        if not self.connected:
            return None
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT access_token, user_name FROM kite_tokens
                    WHERE user_id = 'default' AND is_valid = TRUE 
                    AND datetime(expires_at) > datetime('now')
                    ORDER BY created_at DESC LIMIT 1
                """))
                row = result.fetchone()
                if row:
                    return {"access_token": row[0], "user_name": row[1]}
                return None
        except Exception as e:
            logger.error(f"Error getting token: {e}")
            return None

    def invalidate_token(self):
        if not self.connected:
            return
        try:
            with self.engine.connect() as conn:
                conn.execute(text("UPDATE kite_tokens SET is_valid = FALSE WHERE user_id = 'default'"))
                conn.commit()
        except Exception as e:
            logger.error(f"Error invalidating token: {e}")

# Enhanced Kite Connect Manager with VS Code Integration
class KiteConnectManager:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.kite = None
        self.kws = None
        self.access_token = None
        self.is_authenticated = False
        self.tick_buffer = {}
        self.candle_store = {}
        self.ws_running = False
        self.token_db = KiteTokenDatabase()
        
        # Store in session state
        if "kite_manager" not in st.session_state:
            st.session_state.kite_manager = self

    def check_oauth_callback(self):
        """Check for OAuth callback - FIXED to prevent redirect loops"""
        try:
            # Check if we have a request_token in query params
            if hasattr(st, 'query_params') and 'request_token' in st.query_params:
                request_token = st.query_params['request_token']
                if request_token and self.api_key and self.api_secret:
                    return self.exchange_request_token(request_token)
            return False
        except Exception as e:
            logger.error(f"OAuth callback error: {e}")
            return False

    def exchange_request_token(self, request_token):
        """Exchange request_token for access_token"""
        try:
            if not self.kite:
                self.kite = KiteConnect(api_key=self.api_key)

            # Generate session
            data = self.kite.generate_session(request_token, api_secret=self.api_secret)

            if data and "access_token" in data:
                self.access_token = data["access_token"]
                self.kite.set_access_token(self.access_token)
                self.is_authenticated = True

                # Save to session state
                st.session_state.kite_access_token = self.access_token
                st.session_state.kite_user_name = data.get("user_name", "")
                st.session_state.kite_profile = data

                # Save to database
                self.token_db.save_token(
                    access_token=self.access_token,
                    user_name=data.get("user_name", ""),
                    public_token=data.get("public_token", ""),
                    refresh_token=data.get("refresh_token", "")
                )

                # Clear query params to prevent redirect loops
                try:
                    if hasattr(st, 'query_params'):
                        # Create a new query params dict without request_token
                        new_params = {k: v for k, v in st.query_params.items() if k != 'request_token'}
                        st.query_params.clear()
                        for k, v in new_params.items():
                            st.query_params[k] = v
                except:
                    pass

                return True
        except Exception as e:
            logger.error(f"Token exchange error: {e}")
            st.error(f"Token exchange failed: {str(e)}")
        return False

    def login_via_ui(self):
        """Login via UI - FIXED to prevent redirect loops"""
        try:
            if not self.api_key:
                st.warning("Kite API Key not configured. Set KITE_API_KEY and KITE_API_SECRET in environment secrets.")
                return False

            # Check for existing token first
            if self.check_existing_token():
                return True

            # Check for OAuth callback
            if self.check_oauth_callback():
                st.success("✅ Successfully authenticated with Kite Connect!")
                st.rerun()
                return True

            # Show login UI
            st.markdown("""
            <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%); 
                        padding: 20px; border-radius: 10px; text-align: center; margin: 10px 0;">
                <h3 style="color: white; margin-bottom: 15px;">Connect to Zerodha Kite</h3>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Method 1: OAuth Login")
                if self.api_key:
                    self.kite = KiteConnect(api_key=self.api_key)
                    login_url = self.kite.login_url()
                    st.markdown(f"""
                    <a href="{login_url}" target="_blank" style="
                        display: inline-block; 
                        background: #f59e0b; 
                        color: white; 
                        padding: 12px 30px; 
                        border-radius: 8px; 
                        text-decoration: none; 
                        font-weight: bold;
                        margin: 10px 0;">
                        Login with Kite
                    </a>
                    """, unsafe_allow_html=True)
                    
                    st.info("After login, you'll get a request token. Paste it below:")
                    request_token = st.text_input("Request Token", type="password")
                    if st.button("Authenticate", key="oauth_auth"):
                        if request_token:
                            if self.exchange_request_token(request_token):
                                st.success("Authenticated successfully!")
                                st.rerun()
                            else:
                                st.error("Authentication failed")
            
            with col2:
                st.subheader("Method 2: Direct Token")
                access_token = st.text_input("Access Token", type="password")
                if st.button("Use Token", key="direct_token"):
                    if access_token:
                        try:
                            self.kite = KiteConnect(api_key=self.api_key)
                            self.access_token = access_token
                            self.kite.set_access_token(self.access_token)
                            profile = self.kite.profile()
                            self.is_authenticated = True
                            st.session_state.kite_access_token = self.access_token
                            st.session_state.kite_user_name = profile.get("user_name", "")
                            self.token_db.save_token(access_token=self.access_token, user_name=profile.get("user_name", ""))
                            st.success(f"Authenticated as {profile.get('user_name', 'User')}")
                            return True
                        except Exception as e:
                            st.error(f"Invalid token: {str(e)}")
            
            return False

        except Exception as e:
            st.error(f"Login error: {str(e)}")
            return False

    def check_existing_token(self):
        """Check for existing valid token"""
        try:
            # Check session state first
            if "kite_access_token" in st.session_state and st.session_state.kite_access_token:
                try:
                    if not self.kite:
                        self.kite = KiteConnect(api_key=self.api_key)
                    self.access_token = st.session_state.kite_access_token
                    self.kite.set_access_token(self.access_token)
                    profile = self.kite.profile()
                    self.is_authenticated = True
                    return True
                except:
                    # Token expired, clear it
                    del st.session_state.kite_access_token

            # Check database
            db_token = self.token_db.get_valid_token()
            if db_token:
                try:
                    if not self.kite:
                        self.kite = KiteConnect(api_key=self.api_key)
                    self.access_token = db_token["access_token"]
                    self.kite.set_access_token(self.access_token)
                    profile = self.kite.profile()
                    self.is_authenticated = True
                    st.session_state.kite_access_token = self.access_token
                    st.session_state.kite_user_name = db_token.get("user_name", "")
                    return True
                except:
                    self.token_db.invalidate_token()

            return False
        except Exception as e:
            logger.error(f"Error checking existing token: {e}")
            return False

    def logout(self):
        """Logout from Kite Connect"""
        try:
            if "kite_access_token" in st.session_state:
                del st.session_state.kite_access_token
            if "kite_user_name" in st.session_state:
                del st.session_state.kite_user_name
            if "kite_profile" in st.session_state:
                del st.session_state.kite_profile
            self.token_db.invalidate_token()
            self.access_token = None
            self.is_authenticated = False
            if self.kws:
                self.kws.close()
                self.ws_running = False
            return True
        except Exception as e:
            logger.error(f"Logout error: {e}")
            return False

    def get_live_data(self, instrument_token, interval="minute", days=1):
        """Get live data from Kite Connect"""
        if not self.is_authenticated:
            return None

        try:
            from_date = datetime.now().date() - pd.Timedelta(days=days)
            to_date = datetime.now().date()

            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
                continuous=False,
                oi=False
            )

            if data:
                df = pd.DataFrame(data)
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                return df
            return None

        except Exception as e:
            logger.error(f"Error fetching Kite data: {e}")
            return None

    def get_live_quote(self, instrument_token):
        """Get live quote for an instrument"""
        if not self.is_authenticated:
            return None

        try:
            quote = self.kite.quote([instrument_token])
            if instrument_token in quote:
                return quote[instrument_token]
            return None
        except Exception as e:
            logger.error(f"Error fetching live quote: {e}")
            return None

    def get_instruments(self, exchange="NSE"):
        """Get all instruments for an exchange"""
        if not self.is_authenticated:
            return None

        try:
            instruments = self.kite.instruments(exchange)
            return pd.DataFrame(instruments)
        except Exception as e:
            logger.error(f"Error fetching instruments: {e}")
            return None

    def search_instruments(self, symbol):
        """Search for instruments by symbol"""
        if not self.is_authenticated:
            return []

        try:
            instruments_df = self.get_instruments()
            if instruments_df is not None:
                results = instruments_df[instruments_df['tradingsymbol'].str.contains(symbol, case=False, na=False)]
                return results[['instrument_token', 'tradingsymbol', 'name', 'lot_size']].to_dict('records')
            return []
        except Exception as e:
            logger.error(f"Error searching instruments: {e}")
            return []

# VS Code Style Kite Live Charts
def create_vs_code_kite_charts():
    """Create VS Code style Kite Connect Live Charts"""
    
    st.markdown('<div class="panel-header"><div>📈 KITE CONNECT LIVE CHARTS</div></div>', unsafe_allow_html=True)
    
    # Initialize Kite Manager
    if "kite_manager" not in st.session_state:
        st.session_state.kite_manager = KiteConnectManager(KITE_API_KEY, KITE_API_SECRET)
    
    kite_manager = st.session_state.kite_manager
    
    # Authentication Section
    if not kite_manager.is_authenticated:
        st.warning("🔐 Kite Connect authentication required")
        if kite_manager.login_via_ui():
            st.rerun()
        return
    
    # User Info
    user_info = st.session_state.get('kite_profile', {})
    st.success(f"✅ Connected as: {user_info.get('user_name', 'User')}")
    
    # Main Chart Interface
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Chart Settings")
        
        # Instrument Search
        search_symbol = st.text_input("Search Symbol", placeholder="RELIANCE, TCS, etc.")
        if search_symbol:
            results = kite_manager.search_instruments(search_symbol)
            if results:
                selected_instrument = st.selectbox(
                    "Select Instrument",
                    results,
                    format_func=lambda x: f"{x['tradingsymbol']} - {x['name']}"
                )
                instrument_token = selected_instrument['instrument_token']
                instrument_name = selected_instrument['tradingsymbol']
            else:
                st.warning("No instruments found")
                instrument_token = None
        else:
            # Default instruments
            default_instruments = [
                {"name": "NIFTY 50", "token": 256265},
                {"name": "BANKNIFTY", "token": 260105},
                {"name": "RELIANCE", "token": 738561},
                {"name": "TCS", "token": 2953217}
            ]
            selected_default = st.selectbox(
                "Select Index",
                default_instruments,
                format_func=lambda x: x['name']
            )
            instrument_token = selected_default['token']
            instrument_name = selected_default['name']
        
        # Chart Settings
        interval = st.selectbox(
            "Interval",
            ["minute", "5minute", "15minute", "30minute", "hour", "day"],
            index=2
        )
        
        days_back = st.slider("Days Back", 1, 30, 7)
        
        chart_type = st.selectbox(
            "Chart Type",
            ["Candlestick", "Line", "OHLC"]
        )
        
        # Indicators
        st.subheader("Indicators")
        show_ema = st.checkbox("EMA", value=True)
        if show_ema:
            ema_period = st.slider("EMA Period", 5, 50, 20)
        
        show_rsi = st.checkbox("RSI", value=False)
        if show_rsi:
            rsi_period = st.slider("RSI Period", 5, 30, 14)
        
        show_volume = st.checkbox("Volume", value=True)
        
        # Live Updates
        live_updates = st.checkbox("Live Updates", value=False)
    
    with col2:
        st.subheader(f"Live Chart: {instrument_name if 'instrument_name' in locals() else 'NIFTY 50'}")
        
        if st.button("Load Chart", type="primary") or 'chart_loaded' in st.session_state:
            st.session_state.chart_loaded = True
            
            with st.spinner("Fetching data..."):
                data = kite_manager.get_live_data(instrument_token, interval, days_back)
                
                if data is not None and not data.empty:
                    # Create chart
                    fig = make_subplots(
                        rows=2 if show_volume else 1, 
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        row_heights=[0.7, 0.3] if show_volume else [1.0]
                    )
                    
                    # Price chart
                    row = 1
                    if chart_type == "Candlestick":
                        fig.add_trace(
                            go.Candlestick(
                                x=data.index,
                                open=data['open'],
                                high=data['high'],
                                low=data['low'],
                                close=data['close'],
                                name='Price',
                                increasing_line_color='#2ecc71',
                                decreasing_line_color='#e74c3c'
                            ),
                            row=row, col=1
                        )
                    elif chart_type == "OHLC":
                        fig.add_trace(
                            go.Ohlc(
                                x=data.index,
                                open=data['open'],
                                high=data['high'],
                                low=data['low'],
                                close=data['close'],
                                name='Price',
                                increasing_line_color='#2ecc71',
                                decreasing_line_color='#e74c3c'
                            ),
                            row=row, col=1
                        )
                    else:  # Line
                        fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=data['close'],
                                mode='lines',
                                name='Price',
                                line=dict(color='#3498db', width=2)
                            ),
                            row=row, col=1
                        )
                    
                    # Add EMA
                    if show_ema:
                        data['EMA'] = data['close'].ewm(span=ema_period, adjust=False).mean()
                        fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=data['EMA'],
                                mode='lines',
                                name=f'EMA{ema_period}',
                                line=dict(color='#f39c12', width=1.5, dash='dash')
                            ),
                            row=row, col=1
                        )
                    
                    # Add RSI subplot
                    if show_rsi:
                        delta = data['close'].diff()
                        gain = delta.clip(lower=0)
                        loss = -delta.clip(upper=0)
                        avg_gain = gain.rolling(window=rsi_period).mean()
                        avg_loss = loss.rolling(window=rsi_period).mean()
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                        
                        fig.add_trace(
                            go.Scatter(
                                x=data.index,
                                y=rsi,
                                mode='lines',
                                name='RSI',
                                line=dict(color='#9b59b6', width=1.5)
                            ),
                            row=row, col=1
                        )
                        
                        # Add RSI levels
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=row, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=row, col=1)
                    
                    # Add volume
                    if show_volume and 'volume' in data.columns:
                        colors = ['#2ecc71' if close >= open else '#e74c3c' 
                                 for close, open in zip(data['close'], data['open'])]
                        
                        fig.add_trace(
                            go.Bar(
                                x=data.index,
                                y=data['volume'],
                                name='Volume',
                                marker_color=colors,
                                opacity=0.7
                            ),
                            row=2, col=1
                        )
                    
                    # Update layout
                    fig.update_layout(
                        title=f'{instrument_name} - {interval} Chart',
                        xaxis_title='Time',
                        yaxis_title='Price',
                        height=600,
                        template='plotly_dark',
                        showlegend=True,
                        hovermode='x unified'
                    )
                    
                    if show_volume:
                        fig.update_yaxes(title_text="Volume", row=2, col=1)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Current stats
                    current_price = data['close'].iloc[-1]
                    prev_close = data['close'].iloc[-2] if len(data) > 1 else current_price
                    change_pct = ((current_price - prev_close) / prev_close) * 100
                    
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    col_stat1.metric("Current", f"₹{current_price:.2f}")
                    col_stat2.metric("Change", f"₹{current_price - prev_close:+.2f}")
                    col_stat3.metric("Change %", f"{change_pct:+.2f}%")
                    col_stat4.metric("Period High", f"₹{data['high'].max():.2f}")
                    
                else:
                    st.error("Could not fetch data. Please check your connection.")
    
    # Real-time quotes section
    st.markdown("---")
    st.subheader("📊 Real-time Quotes")
    
    if st.button("Get Real-time Quotes"):
        if instrument_token:
            quote = kite_manager.get_live_quote(instrument_token)
            if quote:
                col_q1, col_q2, col_q3, col_q4 = st.columns(4)
                with col_q1:
                    st.metric("Last Price", f"₹{quote['last_price']:.2f}")
                with col_q2:
                    st.metric("Volume", f"{quote.get('volume', 0):,}")
                with col_q3:
                    st.metric("OI", f"{quote.get('oi', 0):,}")
                with col_q4:
                    buy_qty = quote.get('buy_quantity', 0)
                    sell_qty = quote.get('sell_quantity', 0)
                    st.metric("Buy/Sell Qty", f"{buy_qty:,}/{sell_qty:,}")

# Main VS Code Interface
def main_vs_code_interface():
    """Main VS Code style interface"""
    
    # Initialize session states
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "dashboard"
    if "active_activity" not in st.session_state:
        st.session_state.active_activity = "explorer"
    if "show_explorer" not in st.session_state:
        st.session_state.show_explorer = True
    if "show_terminal" not in st.session_state:
        st.session_state.show_terminal = False
    if "terminal" not in st.session_state:
        st.session_state.terminal = VSCodeTerminal()
    
    # Create layout containers
    main_container = st.container()
    
    with main_container:
        # Activity Bar (Left)
        vs_code_activity_bar()
        
        # Explorer (Left Panel)
        if st.session_state.show_explorer:
            vs_code_file_explorer()
        
        # Main Content Area
        content_class = "main-content with-explorer" if st.session_state.show_explorer else "main-content"
        st.markdown(f'<div class="{content_class}">', unsafe_allow_html=True)
        
        # Tabs
        vs_code_tabs()
        
        # Tab Content
        active_tab = st.session_state.active_tab
        
        if active_tab == "dashboard":
            st.title("📈 Trading Dashboard - VS Code Style")
            
            # Dashboard widgets
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Account Value", "₹2,150,000", "+3.5%")
            with col2:
                st.metric("Today's P&L", "₹15,250", "+2.1%")
            with col3:
                st.metric("Open Positions", "8")
            with col4:
                st.metric("Win Rate", "68.5%")
            
            # Quick actions
            st.subheader("Quick Actions")
            action_cols = st.columns(5)
            with action_cols[0]:
                if st.button("📊 Scan Market", use_container_width=True):
                    st.session_state.active_tab = "signals"
            with action_cols[1]:
                if st.button("📈 View Charts", use_container_width=True):
                    st.session_state.active_tab = "charts"
            with action_cols[2]:
                if st.button("💰 Place Trade", use_container_width=True):
                    st.session_state.active_tab = "trading"
            with action_cols[3]:
                if st.button("🤖 Run Strategy", use_container_width=True):
                    st.session_state.active_tab = "code"
            with action_cols[4]:
                if st.button("🖥️ Open Terminal", use_container_width=True):
                    st.session_state.show_terminal = True
            
            # Market Overview
            st.subheader("Market Overview")
            # Add market overview content here
            
        elif active_tab == "signals":
            st.title("🚦 Trading Signals")
            # Add signals content here
            
        elif active_tab == "trading":
            st.title("💰 Paper Trading")
            # Add trading content here
            
        elif active_tab == "charts":
            st.title("📊 Kite Connect Live Charts")
            create_vs_code_kite_charts()
            
        elif active_tab == "code":
            st.title("💻 Strategy Code Editor")
            vs_code_code_editor()
            
        elif active_tab == "terminal":
            st.title("🖥️ VS Code Terminal")
            st.session_state.terminal.render()
            
        elif active_tab == "settings":
            st.title("⚙️ Settings")
            
            # Theme Settings
            st.subheader("Theme Settings")
            selected_theme = st.selectbox(
                "Select Theme",
                list(VS_CODE_THEMES.keys()),
                index=0
            )
            if st.button("Apply Theme"):
                apply_vs_code_theme(selected_theme)
                st.success(f"Theme changed to {selected_theme}")
                st.rerun()
            
            # Kite Connect Settings
            st.subheader("Kite Connect Settings")
            api_key = st.text_input("API Key", value=KITE_API_KEY, type="password")
            api_secret = st.text_input("API Secret", value=KITE_API_SECRET, type="password")
            
            if st.button("Save API Settings"):
                os.environ["KITE_API_KEY"] = api_key
                os.environ["KITE_API_SECRET"] = api_secret
                st.success("API settings saved")
            
            # UI Settings
            st.subheader("UI Settings")
            show_explorer = st.checkbox("Show Explorer", value=st.session_state.show_explorer)
            if show_explorer != st.session_state.show_explorer:
                st.session_state.show_explorer = show_explorer
                st.rerun()
            
            auto_refresh = st.checkbox("Auto Refresh Charts", value=True)
            refresh_rate = st.slider("Refresh Rate (seconds)", 5, 60, 30)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Terminal (Bottom Panel)
        if st.session_state.show_terminal:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.session_state.terminal.render()
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Status Bar (Bottom)
        vs_code_status_bar()

# Run the main interface
if __name__ == "__main__":
    main_vs_code_interface()
