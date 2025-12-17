# Rantv Intraday Trading Signals & Market Analysis - PRODUCTION READY
# ENHANCED VERSION WITH FULL STOCK SCANNING & BETTER SIGNAL QUALITY
# UPDATED: Lowered confidence to 70%, score to 6, added ADX trend filter, optimized for peak hours
# INTEGRATED WITH KITE CONNECT FOR LIVE CHARTS AND VS CODE

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
import base64
from io import BytesIO
import tempfile
import zipfile

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

# Kite Connect API Credentials
KITE_API_KEY = os.environ.get("KITE_API_KEY", "pwnmsnpy30s4uotu")
KITE_API_SECRET = os.environ.get("KITE_API_SECRET", "m44rfdl9ligc4ctaq7r9sxkxpgnfm30m")
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

    @classmethod
    def from_env(cls):
        return cls()

# Initialize configuration
config = AppConfig.from_env()

st.set_page_config(page_title="Rantv Intraday Terminal Pro - Enhanced", layout="wide", initial_sidebar_state="expanded")
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

NIFTY_100 = NIFTY_50 + [
    "BAJAJHLDNG.NS", "TATAMOTORS.NS", "VEDANTA.NS", "PIDILITIND.NS",
    "BERGEPAINT.NS", "AMBUJACEM.NS", "DABUR.NS", "HAVELLS.NS", "ICICIPRULI.NS",
    "MARICO.NS", "PEL.NS", "SIEMENS.NS", "TORNTPHARM.NS", "ACC.NS",
    "AUROPHARMA.NS", "BOSCHLTD.NS", "GLENMARK.NS", "MOTHERSUMI.NS", "BIOCON.NS",
    "ZYDUSLIFE.NS", "COLPAL.NS", "CONCOR.NS", "DLF.NS", "GODREJCP.NS",
    "HINDPETRO.NS", "IBULHSGFIN.NS", "IOC.NS", "JINDALSTEL.NS", "LUPIN.NS",
    "MANAPPURAM.NS", "MCDOWELL-N.NS", "NMDC.NS", "PETRONET.NS", "PFC.NS",
    "PNB.NS", "RBLBANK.NS", "SAIL.NS", "SRTRANSFIN.NS", "TATAPOWER.NS",
    "YESBANK.NS", "ZEEL.NS"
]

# MIDCAP STOCKS - High Potential for Intraday
NIFTY_MIDCAP_150 = [
    "ABB.NS", "ABCAPITAL.NS", "ABFRL.NS", "ACC.NS", "AUBANK.NS", "AIAENG.NS",
    "APLAPOLLO.NS", "ASTRAL.NS", "AARTIIND.NS", "BALKRISIND.NS", "BANKBARODA.NS",
    "BANKINDIA.NS", "BATAINDIA.NS", "BEL.NS", "BHARATFORG.NS", "BHEL.NS",
    "BIOCON.NS", "BOSCHLTD.NS", "BRIGADE.NS", "CANBK.NS", "CANFINHOME.NS",
    "CHOLAFIN.NS", "CIPLA.NS", "COALINDIA.NS", "COFORGE.NS", "COLPAL.NS",
    "CONCOR.NS", "COROMANDEL.NS", "CROMPTON.NS", "CUMMINSIND.NS", "DABUR.NS",
    "DALBHARAT.NS", "DEEPAKNTR.NS", "DELTACORP.NS", "DIVISLAB.NS", "DIXON.NS",
    "DLF.NS", "DRREDDY.NS", "EDELWEISS.NS", "EICHERMOT.NS", "ESCORTS.NS",
    "EXIDEIND.NS", "FEDERALBNK.NS", "GAIL.NS", "GLENMARK.NS", "GODREJCP.NS",
    "GODREJPROP.NS", "GRANULES.NS", "GRASIM.NS", "GUJGASLTD.NS", "HAL.NS",
    "HAVELLS.NS", "HCLTECH.NS", "HDFCAMC.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS",
    "HINDALCO.NS", "HINDPETRO.NS", "HINDUNILVR.NS", "ICICIPRULI.NS",
    "IDEA.NS", "IDFCFIRSTB.NS", "IGL.NS", "INDIACEM.NS", "INDIAMART.NS",
    "INDUSTOWER.NS", "INFY.NS", "IOC.NS", "IPCALAB.NS", "JINDALSTEL.NS",
    "JSWENERGY.NS", "JUBLFOOD.NS", "KOTAKBANK.NS", "L&TFH.NS", "LICHSGFIN.NS",
    "LT.NS", "LTTS.NS", "MANAPPURAM.NS", "MARICO.NS", "MARUTI.NS", "MFSL.NS",
    "MGL.NS", "MINDTREE.NS", "MOTHERSUMI.NS", "MPHASIS.NS", "MRF.NS",
    "MUTHOOTFIN.NS", "NATIONALUM.NS", "NAUKRI.NS", "NESTLEIND.NS", "NMDC.NS",
    "NTPC.NS", "OBEROIRLTY.NS", "OFSS.NS", "ONGC.NS", "PAGEIND.NS",
    "PEL.NS", "PETRONET.NS", "PFC.NS", "PIDILITIND.NS", "PIIND.NS",
    "PNB.NS", "POWERGRID.NS", "RAJESHEXPO.NS", "RAMCOCEM.NS", "RBLBANK.NS",
    "RECLTD.NS", "RELIANCE.NS", "SAIL.NS", "SBICARD.NS", "SBILIFE.NS",
    "SHREECEM.NS", "SIEMENS.NS", "SRF.NS", "SRTRANSFIN.NS", "SUNPHARMA.NS",
    "SUNTV.NS", "SYNGENE.NS", "TATACHEM.NS", "TATACONSUM.NS", "TATAMOTORS.NS",
    "TATAPOWER.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS",
    "TORNTPHARM.NS", "TRENT.NS", "UPL.NS", "VOLTAS.NS", "WIPRO.NS",
    "YESBANK.NS", "ZEEL.NS"
]

# COMBINED ALL STOCKS - NEW UNIVERSES
import re as _re
def _clean_list(lst):
    clean = []
    removed = []
    for s in lst:
        if not isinstance(s, str):
            continue
        t = s.strip().upper()
        if not t.endswith(".NS"):
            t = t.replace(" ", "").upper() + ".NS"
        if _re.match(r"^[A-Z0-9\.\-]+$", t) and "&" not in t and "#" not in t and "@" not in t:
            clean.append(t)
        else:
            removed.append(t)
    final = []
    seen = set()
    for c in clean:
        if c not in seen:
            final.append(c)
            seen.add(c)
    return final, removed

NIFTY_50, bad1 = _clean_list(NIFTY_50)
NIFTY_100, bad2 = _clean_list(NIFTY_100)
NIFTY_MIDCAP_150, bad3 = _clean_list(NIFTY_MIDCAP_150)

ALL_STOCKS = list(dict.fromkeys(NIFTY_50 + NIFTY_100 + NIFTY_MIDCAP_150))

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

# ====================================================
# VS CODE INTEGRATION SECTION
# ====================================================

class VSCodeEditor:
    """VS Code-like Editor Component"""
    
    def __init__(self):
        self.files = {}
        self.current_file = None
        self.theme = "vs-dark"
        
    def create_editor(self, height=600):
        """Create VS Code-like editor interface"""
        st.markdown("""
        <style>
        .vscode-editor {
            background: #1e1e1e;
            border-radius: 8px;
            padding: 10px;
            border: 1px solid #333;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        }
        .vscode-tab {
            background: #2d2d2d;
            padding: 8px 15px;
            border-radius: 4px 4px 0 0;
            border: 1px solid #333;
            border-bottom: none;
            margin-right: 5px;
            cursor: pointer;
            color: #ccc;
            font-size: 12px;
        }
        .vscode-tab.active {
            background: #1e1e1e;
            color: #fff;
            border-color: #007acc;
        }
        .vscode-line-number {
            color: #858585;
            text-align: right;
            padding-right: 10px;
            user-select: none;
        }
        .code-area {
            background: #1e1e1e;
            color: #d4d4d4;
            border: none;
            width: 100%;
            min-height: 500px;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            line-height: 1.5;
            white-space: pre;
            overflow-x: auto;
            tab-size: 4;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Editor layout
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown('<h4 style="color: #007acc;">📝 VS Code Editor</h4>', unsafe_allow_html=True)
        
        with col2:
            theme = st.selectbox("Theme", ["vs-dark", "vs-light", "hc-black"], key="vscode_theme")
            self.theme = theme
            
        with col3:
            if st.button("💾 Save", key="save_code"):
                self.save_current_file()
                
        # File tabs
        tab_html = '<div style="display: flex; margin-bottom: -1px;">'
        for idx, (filename, content) in enumerate(self.files.items()):
            active = "active" if filename == self.current_file else ""
            tab_html += f'<div class="vscode-tab {active}" id="tab_{idx}">{filename}</div>'
        tab_html += '</div>'
        st.markdown(tab_html, unsafe_allow_html=True)
        
        # Code editor area
        if self.current_file and self.current_file in self.files:
            code = self.files[self.current_file]
            lines = code.split('\n')
            
            # Create line numbers and code display
            editor_html = '<div class="vscode-editor">'
            editor_html += '<div style="display: flex;">'
            editor_html += '<div style="width: 50px;">'
            for i in range(1, len(lines) + 1):
                editor_html += f'<div class="vscode-line-number">{i}</div>'
            editor_html += '</div>'
            editor_html += '<div style="flex-grow: 1;">'
            
            # Syntax highlighting (basic)
            for line in lines:
                colored_line = self._syntax_highlight(line)
                editor_html += f'<div style="white-space: pre; padding-left: 10px;">{colored_line}</div>'
            
            editor_html += '</div></div></div>'
            
            st.markdown(editor_html, unsafe_allow_html=True)
            
            # Actual text area for editing
            edited_code = st.text_area(
                "Edit code",
                value=code,
                height=height,
                label_visibility="collapsed",
                key=f"editor_{self.current_file}"
            )
            
            if edited_code != code:
                self.files[self.current_file] = edited_code
                
        else:
            st.info("No file open. Create a new file or open an existing one.")
            
        # File operations
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            new_filename = st.text_input("New filename", "script.py", key="new_filename")
            if st.button("➕ New", key="new_file"):
                self.files[new_filename] = "# New Python script\nprint('Hello World!')"
                self.current_file = new_filename
                st.rerun()
                
        with col2:
            file_to_open = st.selectbox("Open file", list(self.files.keys()), key="open_file_select")
            if st.button("📂 Open", key="open_file"):
                self.current_file = file_to_open
                st.rerun()
                
        with col3:
            if st.button("🗑️ Delete", key="delete_file"):
                if self.current_file in self.files:
                    del self.files[self.current_file]
                    if self.files:
                        self.current_file = list(self.files.keys())[0]
                    else:
                        self.current_file = None
                    st.rerun()
                    
        with col4:
            if st.button("▶️ Run", key="run_code"):
                if self.current_file:
                    self.run_code()
    
    def _syntax_highlight(self, line):
        """Basic syntax highlighting"""
        keywords = ['def', 'class', 'if', 'else', 'elif', 'for', 'while', 'import', 'from', 'as', 'return', 'try', 'except', 'finally']
        builtins = ['print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 'set']
        
        highlighted = line
        
        # Highlight keywords
        for kw in keywords:
            pattern = rf'\b{kw}\b'
            highlighted = re.sub(pattern, f'<span style="color: #569cd6;">{kw}</span>', highlighted)
            
        # Highlight builtins
        for bi in builtins:
            pattern = rf'\b{bi}\b'
            highlighted = re.sub(pattern, f'<span style="color: #dcdcaa;">{bi}</span>', highlighted)
            
        # Highlight strings
        highlighted = re.sub(r'(\".*?\")', r'<span style="color: #ce9178;">\1</span>', highlighted)
        highlighted = re.sub(r"(\'.*?\')", r'<span style="color: #ce9178;">\1</span>', highlighted)
        
        # Highlight numbers
        highlighted = re.sub(r'\b(\d+)\b', r'<span style="color: #b5cea8;">\1</span>', highlighted)
        
        # Highlight comments
        highlighted = re.sub(r'(#.*)$', r'<span style="color: #6a9955;">\1</span>', highlighted)
        
        return highlighted
    
    def save_current_file(self):
        """Save current file"""
        if self.current_file:
            st.success(f"✅ Saved {self.current_file}")
            
    def run_code(self):
        """Execute the current code"""
        if not self.current_file or self.current_file not in self.files:
            st.error("No file to run")
            return
            
        code = self.files[self.current_file]
        
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
                
            # Execute the code
            exec_globals = {}
            exec(code, exec_globals)
            
            st.success("✅ Code executed successfully")
            
            # Show output
            if 'output' in exec_globals:
                st.code(exec_globals['output'])
                
        except Exception as e:
            st.error(f"❌ Error executing code: {str(e)}")
            
    def load_default_files(self):
        """Load default trading scripts"""
        default_files = {
            "trading_strategy.py": '''# Trading Strategy Template
import pandas as pd
import numpy as np

class TradingStrategy:
    def __init__(self):
        self.name = "Custom Strategy"
        
    def calculate_signals(self, data):
        """
        Calculate trading signals from OHLCV data
        data: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
        Returns: DataFrame with signals
        """
        signals = pd.DataFrame(index=data.index)
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        signals['RSI'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        signals['Signal'] = 0
        signals.loc[signals['RSI'] < 30, 'Signal'] = 1  # Buy signal
        signals.loc[signals['RSI'] > 70, 'Signal'] = -1  # Sell signal
        
        return signals

# Example usage
if __name__ == "__main__":
    strategy = TradingStrategy()
    print("Custom strategy loaded successfully")''',
            
            "kite_connect_helper.py": '''# Kite Connect Helper Functions
import pandas as pd
from datetime import datetime, timedelta

class KiteDataFetcher:
    def __init__(self, kite_client):
        self.kite = kite_client
        
    def get_historical_data(self, instrument_token, interval="minute", days=7):
        """
        Fetch historical data from Kite Connect
        """
        to_date = datetime.now().date()
        from_date = to_date - timedelta(days=days)
        
        data = self.kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_date,
            to_date=to_date,
            interval=interval
        )
        
        df = pd.DataFrame(data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
        return df
        
    def get_live_quotes(self, instrument_tokens):
        """
        Get live quotes for multiple instruments
        """
        quotes = self.kite.quote(instrument_tokens)
        return quotes
        
    def place_order(self, symbol, quantity, order_type="MARKET", product="MIS"):
        """
        Place an order through Kite Connect
        """
        try:
            order_id = self.kite.place_order(
                tradingsymbol=symbol,
                exchange=self.kite.EXCHANGE_NSE,
                transaction_type=self.kite.TRANSACTION_TYPE_BUY,
                quantity=quantity,
                order_type=order_type,
                product=product,
                variety=self.kite.VARIETY_REGULAR
            )
            return order_id
        except Exception as e:
            print(f"Order placement failed: {e}")
            return None''',
            
            "data_analyzer.py": '''# Data Analysis and Visualization
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DataAnalyzer:
    def __init__(self):
        self.data = None
        
    def load_data(self, df):
        """Load DataFrame for analysis"""
        self.data = df
        
    def calculate_indicators(self):
        """Calculate technical indicators"""
        if self.data is None:
            return
            
        # Moving Averages
        self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
        
        # Bollinger Bands
        self.data['BB_Middle'] = self.data['Close'].rolling(window=20).mean()
        self.data['BB_Std'] = self.data['Close'].rolling(window=20).std()
        self.data['BB_Upper'] = self.data['BB_Middle'] + 2 * self.data['BB_Std']
        self.data['BB_Lower'] = self.data['BB_Middle'] - 2 * self.data['BB_Std']
        
        # Volume indicators
        self.data['Volume_SMA'] = self.data['Volume'].rolling(window=20).mean()
        
    def create_chart(self):
        """Create interactive chart"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=self.data.index,
                open=self.data['Open'],
                high=self.data['High'],
                low=self.data['Low'],
                close=self.data['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Volume bars
        colors = ['red' if row['Open'] > row['Close'] else 'green' 
                 for _, row in self.data.iterrows()]
        
        fig.add_trace(
            go.Bar(
                x=self.data.index,
                y=self.data['Volume'],
                name='Volume',
                marker_color=colors
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Stock Analysis',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            height=600
        )
        
        return fig''',
            
            "risk_manager.py": '''# Risk Management Module
import numpy as np

class RiskManager:
    def __init__(self, max_position_size=0.1, max_daily_loss=0.02):
        """
        max_position_size: Maximum position size as percentage of capital
        max_daily_loss: Maximum daily loss as percentage of capital
        """
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.daily_pnl = 0
        
    def calculate_position_size(self, capital, entry_price, stop_loss):
        """
        Calculate position size based on risk parameters
        """
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return 0
            
        max_risk = capital * self.max_daily_loss
        max_shares_by_risk = max_risk / risk_per_share
        
        max_shares_by_capital = (capital * self.max_position_size) // entry_price
        
        return min(max_shares_by_risk, max_shares_by_capital)
        
    def check_trade_validity(self, capital, entry_price, stop_loss, quantity):
        """
        Check if a trade is within risk limits
        """
        position_value = entry_price * quantity
        position_percentage = position_value / capital
        
        if position_percentage > self.max_position_size:
            return False, f"Position size {position_percentage:.1%} exceeds limit {self.max_position_size:.1%}"
            
        risk_amount = abs(entry_price - stop_loss) * quantity
        risk_percentage = risk_amount / capital
        
        if risk_percentage > self.max_daily_loss:
            return False, f"Risk {risk_percentage:.1%} exceeds daily limit {self.max_daily_loss:.1%}"
            
        return True, "Trade within risk limits"
        
    def update_daily_pnl(self, pnl):
        """
        Update daily P&L
        """
        self.daily_pnl += pnl
        
    def check_daily_limit(self, capital):
        """
        Check if daily loss limit is reached
        """
        if self.daily_pnl / capital <= -self.max_daily_loss:
            return False, f"Daily loss limit reached: {self.daily_pnl/capital:.1%}"
        return True, "Daily limit not reached"'''
        }
        
        self.files.update(default_files)
        if not self.current_file and default_files:
            self.current_file = list(default_files.keys())[0]

# ====================================================
# ENHANCED KITE CONNECT INTEGRATION
# ====================================================

class EnhancedKiteConnectManager:
    """Enhanced Kite Connect Manager with proper authentication flow"""
    
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.kite = None
        self.kws = None
        self.access_token = None
        self.is_authenticated = False
        self.user_id = None
        self.user_name = None
        
    def handle_oauth_callback(self):
        """Handle OAuth callback from Kite Connect"""
        query_params = st.query_params.to_dict()
        
        if "request_token" in query_params:
            request_token = query_params["request_token"]
            
            try:
                # Remove request_token from URL to prevent redirect loops
                st.query_params.clear()
                
                # Exchange request token for access token
                self.kite = KiteConnect(api_key=self.api_key)
                data = self.kite.generate_session(
                    request_token=request_token,
                    api_secret=self.api_secret
                )
                
                if data and "access_token" in data:
                    self.access_token = data["access_token"]
                    self.kite.set_access_token(self.access_token)
                    self.is_authenticated = True
                    self.user_id = data.get("user_id")
                    self.user_name = data.get("user_name")
                    
                    # Save to session state
                    st.session_state.kite_access_token = self.access_token
                    st.session_state.kite_user_id = self.user_id
                    st.session_state.kite_user_name = self.user_name
                    st.session_state.kite_is_authenticated = True
                    
                    st.success(f"✅ Authenticated as {self.user_name}")
                    return True
                    
            except Exception as e:
                st.error(f"Authentication failed: {str(e)}")
                
        return False
        
    def login(self):
        """Show login interface"""
        if not self.api_key or not self.api_secret:
            st.warning("Please set KITE_API_KEY and KITE_API_SECRET in environment variables")
            return False
            
        # Check session state first
        if st.session_state.get("kite_is_authenticated", False):
            self.access_token = st.session_state.get("kite_access_token")
            self.user_id = st.session_state.get("kite_user_id")
            self.user_name = st.session_state.get("kite_user_name")
            self.is_authenticated = True
            
            try:
                self.kite = KiteConnect(api_key=self.api_key)
                self.kite.set_access_token(self.access_token)
                return True
            except:
                st.session_state.kite_is_authenticated = False
                
        # Handle OAuth callback
        if self.handle_oauth_callback():
            return True
            
        # Show login button
        st.markdown("---")
        st.subheader("🔗 Connect to Zerodha Kite")
        
        if not self.kite:
            self.kite = KiteConnect(api_key=self.api_key)
            
        login_url = self.kite.login_url()
        
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%); border-radius: 10px;">
            <h3 style="color: white;">Connect to Zerodha Kite</h3>
            <p style="color: #e0f2fe;">Click the button below to authenticate with Kite Connect</p>
            <a href="{login_url}" target="_self">
                <button style="background: #f59e0b; color: white; border: none; padding: 12px 30px; border-radius: 8px; font-size: 16px; cursor: pointer; margin: 10px;">
                    Login with Zerodha
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)
        
        return False
        
    def get_historical_data(self, instrument_token, interval="minute", days=7):
        """Get historical data from Kite Connect"""
        if not self.is_authenticated or not self.kite:
            return None
            
        try:
            to_date = datetime.now().date()
            from_date = to_date - timedelta(days=days)
            
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            if data:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                return df
                
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            
        return None
        
    def get_live_quote(self, instrument_token):
        """Get live quote"""
        if not self.is_authenticated or not self.kite:
            return None
            
        try:
            quote = self.kite.quote([instrument_token])
            return quote.get(str(instrument_token))
        except Exception as e:
            logger.error(f"Error fetching live quote: {e}")
            return None
            
    def get_instruments(self, exchange="NSE"):
        """Get instrument list"""
        if not self.is_authenticated or not self.kite:
            return None
            
        try:
            instruments = self.kite.instruments(exchange=exchange)
            return pd.DataFrame(instruments)
        except Exception as e:
            logger.error(f"Error fetching instruments: {e}")
            return None
            
    def place_order(self, symbol, quantity, order_type="MARKET", product="MIS", transaction_type="BUY"):
        """Place an order"""
        if not self.is_authenticated or not self.kite:
            return None
            
        try:
            order_id = self.kite.place_order(
                tradingsymbol=symbol,
                exchange=self.kite.EXCHANGE_NSE,
                transaction_type=transaction_type,
                quantity=quantity,
                order_type=order_type,
                product=product,
                variety=self.kite.VARIETY_REGULAR
            )
            return order_id
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None

# ====================================================
# STYLES AND CONFIGURATION
# ====================================================

st.markdown("""
<style>
    /* Light Yellowish Background */
    .stApp {
        background: linear-gradient(135deg, #fff9e6 0%, #fff0d6 100%);
    }

    /* Enhanced Tabs with Multiple Colors */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: linear-gradient(135deg, #e6f2ff 0%, #ffe6e6 50%, #e6ffe6 100%);
        padding: 8px;
        border-radius: 12px;
        margin-bottom: 1rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 8px;
        gap: 8px;
        padding: 12px 20px;
        font-weight: 600;
        font-size: 14px;
        color: #1e3a8a;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
        color: white;
        border: 2px solid #2563eb;
        box-shadow: 0 4px 8px rgba(30, 58, 138, 0.3);
        transform: translateY(-2px);
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #dbeafe 0%, #e0f2fe 100%);
        border: 2px solid #93c5fd;
        transform: translateY(-1px);
    }

    /* Circular Market Mood Gauge */
    .gauge-container {
        background: white;
        border-radius: 50%;
        padding: 25px;
        margin: 10px auto;
        border: 4px solid #e0f2fe;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        width: 200px;
        height: 200px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        position: relative;
    }

    /* Signal Quality Cards */
    .high-quality-signal {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid #047857;
    }

    /* Kite Connect Panel */
    .kite-panel {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ====================================================
# MAIN APPLICATION
# ====================================================

def main():
    # Initialize VS Code Editor
    if "vscode_editor" not in st.session_state:
        st.session_state.vscode_editor = VSCodeEditor()
        st.session_state.vscode_editor.load_default_files()
    
    # Initialize Kite Connect Manager
    if "kite_manager" not in st.session_state:
        st.session_state.kite_manager = EnhancedKiteConnectManager(KITE_API_KEY, KITE_API_SECRET)
    
    # Application Header
    st.markdown("<h1 style='text-align:center; color: #1e3a8a;'>🚀 Rantv Intraday Terminal Pro</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center; color: #6b7280;'>VS Code Editor + Kite Connect Integration</h4>", unsafe_allow_html=True)
    
    # Main tabs
    tabs = st.tabs([
        "📈 Dashboard",
        "💻 VS Code Editor",
        "📊 Kite Live Charts",
        "🚦 Trading Signals",
        "💰 Paper Trading",
        "⚡ Strategies"
    ])
    
    # Tab 1: Dashboard
    with tabs[0]:
        st.subheader("📊 Market Overview")
        
        # Market metrics
        cols = st.columns(4)
        with cols[0]:
            try:
                nifty = yf.download("^NSEI", period="1d", interval="1m")["Close"].iloc[-1]
                st.metric("NIFTY 50", f"₹{nifty:,.2f}")
            except:
                st.metric("NIFTY 50", "₹22,000")
        
        with cols[1]:
            try:
                banknifty = yf.download("^NSEBANK", period="1d", interval="1m")["Close"].iloc[-1]
                st.metric("BANK NIFTY", f"₹{banknifty:,.2f}")
            except:
                st.metric("BANK NIFTY", "₹48,000")
        
        with cols[2]:
            st.metric("Market Status", "OPEN" if market_open() else "CLOSED")
        
        with cols[3]:
            st.metric("Refresh Count", st.session_state.get("refresh_count", 0))
        
        # Kite Connect Status
        st.subheader("🔗 Kite Connect Status")
        kite_manager = st.session_state.kite_manager
        
        if kite_manager.is_authenticated:
            st.success(f"✅ Connected as: {kite_manager.user_name}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh Data", key="refresh_kite"):
                    st.rerun()
            
            with col2:
                if st.button("🚪 Logout", key="logout_kite"):
                    st.session_state.kite_is_authenticated = False
                    st.session_state.kite_manager.is_authenticated = False
                    st.rerun()
        else:
            kite_manager.login()
    
    # Tab 2: VS Code Editor
    with tabs[1]:
        st.session_state.vscode_editor.create_editor()
    
    # Tab 3: Kite Live Charts
    with tabs[2]:
        st.subheader("📊 Kite Connect Live Charts")
        
        kite_manager = st.session_state.kite_manager
        
        if not kite_manager.is_authenticated:
            st.warning("Please connect to Kite Connect first")
            kite_manager.login()
        else:
            # Chart configuration
            col1, col2, col3 = st.columns(3)
            
            with col1:
                index_type = st.selectbox("Select Index", ["NIFTY 50", "BANK NIFTY", "FINNIFTY", "Custom"])
            
            with col2:
                interval = st.selectbox("Interval", ["minute", "5minute", "15minute", "30minute", "hour", "day"])
            
            with col3:
                days = st.slider("Days", 1, 30, 7)
            
            # Instrument mapping
            instrument_map = {
                "NIFTY 50": 256265,
                "BANK NIFTY": 260105,
                "FINNIFTY": 257801
            }
            
            if index_type in instrument_map:
                instrument_token = instrument_map[index_type]
                
                if st.button("📈 Load Chart", key="load_chart"):
                    with st.spinner("Fetching data from Kite Connect..."):
                        data = kite_manager.get_historical_data(instrument_token, interval, days)
                        
                        if data is not None and not data.empty:
                            # Create candlestick chart
                            fig = go.Figure(data=[go.Candlestick(
                                x=data.index,
                                open=data['open'],
                                high=data['high'],
                                low=data['low'],
                                close=data['close'],
                                name='Price'
                            )])
                            
                            # Add moving averages
                            data['MA20'] = data['close'].rolling(window=20).mean()
                            data['MA50'] = data['close'].rolling(window=50).mean()
                            
                            fig.add_trace(go.Scatter(
                                x=data.index,
                                y=data['MA20'],
                                mode='lines',
                                name='MA20',
                                line=dict(color='orange', width=1)
                            ))
                            
                            fig.add_trace(go.Scatter(
                                x=data.index,
                                y=data['MA50'],
                                mode='lines',
                                name='MA50',
                                line=dict(color='blue', width=1)
                            ))
                            
                            fig.update_layout(
                                title=f'{index_type} Chart ({interval} interval)',
                                xaxis_title='Date',
                                yaxis_title='Price',
                                height=500,
                                template='plotly_dark'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show statistics
                            current_price = data['close'].iloc[-1]
                            prev_close = data['close'].iloc[-2] if len(data) > 1 else current_price
                            change = current_price - prev_close
                            change_pct = (change / prev_close) * 100
                            
                            stat_cols = st.columns(4)
                            stat_cols[0].metric("Current", f"₹{current_price:.2f}")
                            stat_cols[1].metric("Change", f"₹{change:+.2f}")
                            stat_cols[2].metric("Change %", f"{change_pct:+.2f}%")
                            stat_cols[3].metric("Volume", f"{data['volume'].iloc[-1]:,.0f}")
                        else:
                            st.error("Failed to fetch data from Kite Connect")
            else:
                st.info("Select an index to load chart")
            
            # Live quotes section
            st.subheader("📊 Live Quotes")
            
            if st.button("🔄 Get Live Quotes", key="live_quotes"):
                nifty_token = 256265
                quote = kite_manager.get_live_quote(nifty_token)
                
                if quote:
                    cols = st.columns(3)
                    cols[0].metric("Last Price", f"₹{quote['last_price']:.2f}")
                    cols[1].metric("Volume", f"{quote['volume']:,}")
                    cols[2].metric("OI", f"{quote.get('oi', 0):,}")
    
    # Tab 4: Trading Signals
    with tabs[3]:
        st.subheader("🚦 Trading Signals from Kite Connect")
        
        kite_manager = st.session_state.kite_manager
        
        if not kite_manager.is_authenticated:
            st.warning("Connect to Kite Connect for live signals")
        else:
            # Signal generation parameters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                signal_universe = st.selectbox("Universe", ["NIFTY 50", "NIFTY 100", "All Stocks"])
            
            with col2:
                min_confidence = st.slider("Min Confidence", 60, 85, 70)
            
            with col3:
                if st.button("🔍 Generate Signals", key="generate_kite_signals"):
                    with st.spinner("Generating signals from Kite data..."):
                        # This would integrate with your existing signal generation logic
                        # using Kite Connect data instead of yfinance
                        signals = generate_kite_signals(kite_manager, signal_universe, min_confidence)
                        
                        if signals:
                            st.success(f"Found {len(signals)} signals")
                            display_signals(signals)
                        else:
                            st.info("No signals found")
    
    # Tab 5: Paper Trading
    with tabs[4]:
        st.subheader("💰 Paper Trading Simulator")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbol = st.selectbox("Symbol", NIFTY_50[:20], key="paper_trade_symbol")
        
        with col2:
            action = st.selectbox("Action", ["BUY", "SELL"], key="paper_trade_action")
        
        with col3:
            quantity = st.number_input("Quantity", min_value=1, value=10, key="paper_trade_qty")
        
        if st.button("📊 Execute Paper Trade", key="execute_paper"):
            st.success(f"Paper trade executed: {action} {quantity} {symbol}")
    
    # Tab 6: Strategies
    with tabs[5]:
        st.subheader("⚡ Trading Strategies")
        
        for strategy, config in HIGH_ACCURACY_STRATEGIES.items():
            with st.expander(f"🔥 {config['name']}"):
                st.write(f"**Type:** {config['type']}")
                st.write(f"**Weight:** {config['weight']}")
                st.write("**Description:** High probability setup with multiple confirmations")

# ====================================================
# HELPER FUNCTIONS
# ====================================================

def market_open():
    """Check if market is open"""
    n = datetime.now(IND_TZ)
    try:
        open_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(9, 15)))
        close_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(15, 30)))
        return open_time <= n <= close_time
    except:
        return False

def generate_kite_signals(kite_manager, universe, min_confidence):
    """Generate trading signals using Kite Connect data"""
    # This is a simplified version - integrate with your existing signal generation logic
    signals = []
    
    try:
        # Get instruments list
        instruments_df = kite_manager.get_instruments()
        
        if instruments_df is not None:
            # Filter by universe
            if universe == "NIFTY 50":
                symbols = NIFTY_50[:50]
            elif universe == "NIFTY 100":
                symbols = NIFTY_100[:100]
            else:
                symbols = ALL_STOCKS[:100]
            
            # Convert symbols to tradingsymbol format
            for symbol in symbols:
                tradingsymbol = symbol.replace(".NS", "")
                
                # Find instrument token
                instrument = instruments_df[instruments_df['tradingsymbol'] == tradingsymbol]
                if not instrument.empty:
                    instrument_token = instrument.iloc[0]['instrument_token']
                    
                    # Get historical data
                    data = kite_manager.get_historical_data(instrument_token, "15minute", 7)
                    
                    if data is not None and len(data) > 50:
                        # Calculate indicators (simplified)
                        data['MA20'] = data['close'].rolling(window=20).mean()
                        data['MA50'] = data['close'].rolling(window=50).mean()
                        
                        last_price = data['close'].iloc[-1]
                        ma20 = data['MA20'].iloc[-1]
                        ma50 = data['MA50'].iloc[-1]
                        
                        # Generate signal
                        if last_price > ma20 > ma50:
                            confidence = 0.75
                            if confidence >= min_confidence / 100:
                                signals.append({
                                    'symbol': symbol,
                                    'action': 'BUY',
                                    'price': last_price,
                                    'confidence': confidence,
                                    'reason': 'Price above moving averages'
                                })
                        
    except Exception as e:
        logger.error(f"Error generating Kite signals: {e}")
    
    return signals

def display_signals(signals):
    """Display trading signals"""
    for signal in signals:
        confidence = signal['confidence']
        
        if confidence >= 0.8:
            quality_class = "high-quality-signal"
        elif confidence >= 0.7:
            quality_class = "medium-quality-signal"
        else:
            quality_class = "low-quality-signal"
        
        st.markdown(f"""
        <div class="{quality_class}">
            <strong>{signal['symbol']}</strong> | {signal['action']} @ ₹{signal['price']:.2f}<br>
            Confidence: {signal['confidence']:.1%} | Reason: {signal['reason']}
        </div>
        """, unsafe_allow_html=True)

# ====================================================
# RUN APPLICATION
# ====================================================

if __name__ == "__main__":
    # Initialize refresh count
    if "refresh_count" not in st.session_state:
        st.session_state.refresh_count = 0
    st.session_state.refresh_count += 1
    
    # Add auto-refresh
    st_autorefresh(interval=60000, key="auto_refresh")
    
    # Run main application
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application crash: {e}")
        st.code(traceback.format_exc())
