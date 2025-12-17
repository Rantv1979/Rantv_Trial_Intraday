# Rantv Intraday Trading Signals & Market Analysis - FIXED VERSION
# NO REDIRECT ERRORS - FULLY WORKING

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
import hashlib
import urllib.parse

# Auto-install missing dependencies
def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except:
        return False

# Check and install kiteconnect if needed
try:
    from kiteconnect import KiteConnect, KiteTicker
    KITECONNECT_AVAILABLE = True
except ImportError:
    if install_package("kiteconnect"):
        from kiteconnect import KiteConnect, KiteTicker
        KITECONNECT_AVAILABLE = True
    else:
        KITECONNECT_AVAILABLE = False

# Setup basic logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Kite Connect API Credentials - Use environment variables
KITE_API_KEY = os.environ.get("KITE_API_KEY", "")
KITE_API_SECRET = os.environ.get("KITE_API_SECRET", "")

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

# Set page config FIRST
st.set_page_config(
    page_title="Rantv Intraday Terminal Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
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

# Stock Universes
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

# Clean CSS - NO redirect issues
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #fff9e6 0%, #fff0d6 100%);
    }
    
    .main .block-container {
        background-color: transparent;
        padding-top: 1rem;
    }
    
    /* VS Code Editor Styling */
    .vscode-container {
        background: #1e1e1e;
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
        border: 1px solid #333;
        font-family: 'Consolas', monospace;
    }
    
    .vscode-header {
        background: #252526;
        padding: 8px 12px;
        border-radius: 5px 5px 0 0;
        color: #cccccc;
        font-family: 'Consolas', monospace;
        font-size: 14px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .vscode-editor {
        background: #1e1e1e;
        color: #d4d4d4;
        font-family: 'Consolas', monospace;
        font-size: 14px;
        padding: 15px;
        border-radius: 0 0 5px 5px;
        min-height: 400px;
        white-space: pre-wrap;
        tab-size: 4;
        overflow-x: auto;
    }
    
    /* Kite Connect Status */
    .kite-connected {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #059669;
        margin: 10px 0;
    }
    
    .kite-disconnected {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #dc2626;
        margin: 10px 0;
    }
    
    /* Signal Cards */
    .signal-buy {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #059669;
        margin: 10px 0;
    }
    
    .signal-sell {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #dc2626;
        margin: 10px 0;
    }
    
    /* Fix for Streamlit elements */
    .stButton > button {
        width: 100%;
    }
    
    div[data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# FIXED: Kite Connect Manager - NO REDIRECT ISSUES
# =============================================
class KiteConnectManager:
    def __init__(self, api_key=None, api_secret=None):
        self.api_key = api_key or KITE_API_KEY
        self.api_secret = api_secret or KITE_API_SECRET
        self.kite = None
        self.access_token = None
        self.is_authenticated = False
        self.user_name = ""
        
    def initialize(self):
        """Initialize Kite Connect - called once"""
        if not self.api_key or not self.api_secret:
            st.warning("Kite API credentials not configured")
            return False
        
        if self.kite is None:
            self.kite = KiteConnect(api_key=self.api_key)
        
        # Check if we have access token in session state
        if 'kite_access_token' in st.session_state:
            self.access_token = st.session_state.kite_access_token
            self.kite.set_access_token(self.access_token)
            
            # Verify token is valid
            try:
                profile = self.kite.profile()
                self.user_name = profile.get('user_name', 'User')
                self.is_authenticated = True
                return True
            except Exception as e:
                logger.error(f"Token invalid: {e}")
                # Clear invalid token
                del st.session_state.kite_access_token
        
        return False
    
    def login(self):
        """Show login interface"""
        if not self.api_key or not self.api_secret:
            st.error("Kite API credentials not configured")
            return False
        
        if self.kite is None:
            self.kite = KiteConnect(api_key=self.api_key)
        
        # Check for request token in URL
        query_params = st.query_params
        if "request_token" in query_params:
            request_token = query_params["request_token"]
            
            try:
                # Generate session
                data = self.kite.generate_session(
                    request_token=request_token,
                    api_secret=self.api_secret
                )
                
                if data and "access_token" in data:
                    self.access_token = data["access_token"]
                    self.kite.set_access_token(self.access_token)
                    self.is_authenticated = True
                    self.user_name = data.get("user_name", "User")
                    
                    # Store in session state
                    st.session_state.kite_access_token = self.access_token
                    st.session_state.kite_user_name = self.user_name
                    
                    # Clear the request token from URL without redirect
                    st.query_params.clear()
                    
                    st.success(f"✅ Authenticated as {self.user_name}")
                    st.rerun()
                    return True
                    
            except Exception as e:
                st.error(f"Authentication failed: {str(e)}")
                return False
        
        # Show login button
        st.info("Kite Connect authentication required for live features")
        
        # Generate login URL
        login_url = self.kite.login_url()
        
        # Display login instructions
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; margin: 10px 0;">
            <h3 style="color: white; margin-bottom: 15px;">Connect to Zerodha Kite</h3>
            <a href="{login_url}" target="_blank"
               style="display: inline-block; background: #f59e0b; color: white; 
                      padding: 12px 30px; border-radius: 8px; text-decoration: none; 
                      font-weight: bold; margin-bottom: 10px;">
                🔗 Login with Kite
            </a>
            <p style="color: #e0f2fe; margin-top: 15px; font-size: 12px;">
                Click the link above to login. After login, paste the request token below:
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Manual token input
        with st.form("manual_kite_login"):
            request_token = st.text_input(
                "Request Token",
                help="Paste the request token from Kite after login",
                type="password"
            )
            
            if st.form_submit_button("Authenticate", type="primary"):
                if request_token:
                    try:
                        data = self.kite.generate_session(
                            request_token=request_token,
                            api_secret=self.api_secret
                        )
                        
                        if data and "access_token" in data:
                            self.access_token = data["access_token"]
                            self.kite.set_access_token(self.access_token)
                            self.is_authenticated = True
                            self.user_name = data.get("user_name", "User")
                            
                            # Store in session state
                            st.session_state.kite_access_token = self.access_token
                            st.session_state.kite_user_name = self.user_name
                            
                            st.success(f"✅ Authenticated as {self.user_name}")
                            st.rerun()
                            return True
                    except Exception as e:
                        st.error(f"Authentication failed: {str(e)}")
        
        return False
    
    def logout(self):
        """Logout from Kite Connect"""
        try:
            # Clear session state
            if 'kite_access_token' in st.session_state:
                del st.session_state.kite_access_token
            if 'kite_user_name' in st.session_state:
                del st.session_state.kite_user_name
            
            # Reset instance variables
            self.access_token = None
            self.is_authenticated = False
            self.user_name = ""
            
            # Clear query params if any
            try:
                st.query_params.clear()
            except:
                pass
            
            return True
        except Exception as e:
            logger.error(f"Logout error: {e}")
            return False
    
    def get_historical_data(self, instrument_token, interval="minute", days=1):
        """Get historical data from Kite Connect"""
        if not self.is_authenticated:
            return None
        
        try:
            from_date = datetime.now().date() - timedelta(days=days)
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
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                return df
            return None
        except Exception as e:
            logger.error(f"Error fetching Kite data: {e}")
            return None
    
    def get_instruments(self, exchange="NSE"):
        """Get instrument list from Kite"""
        if not self.is_authenticated:
            return []
        
        try:
            instruments = self.kite.instruments(exchange)
            return instruments
        except Exception as e:
            logger.error(f"Error fetching instruments: {e}")
            return []
    
    def get_live_quote(self, instrument_token):
        """Get live quote for an instrument"""
        if not self.is_authenticated:
            return None
        
        try:
            quote = self.kite.quote([instrument_token])
            return quote.get(str(instrument_token))
        except Exception as e:
            logger.error(f"Error fetching live quote: {e}")
            return None

# =============================================
# VS Code Strategy Editor - SIMPLIFIED
# =============================================
class StrategyEditor:
    def __init__(self):
        self.templates = {
            "EMA Crossover": """
def generate_signals(data):
    '''EMA Crossover Strategy'''
    import pandas as pd
    import numpy as np
    
    signals = []
    
    # Calculate EMAs
    data['EMA_9'] = data['close'].ewm(span=9, adjust=False).mean()
    data['EMA_21'] = data['close'].ewm(span=21, adjust=False).mean()
    
    current_price = data['close'].iloc[-1]
    ema_9 = data['EMA_9'].iloc[-1]
    ema_21 = data['EMA_21'].iloc[-1]
    
    # BUY when EMA_9 crosses above EMA_21
    if ema_9 > ema_21 and data['EMA_9'].iloc[-2] <= data['EMA_21'].iloc[-2]:
        signals.append({
            'action': 'BUY',
            'price': current_price,
            'confidence': 0.75,
            'reason': 'EMA 9 crossed above EMA 21'
        })
    
    # SELL when EMA_9 crosses below EMA_21
    elif ema_9 < ema_21 and data['EMA_9'].iloc[-2] >= data['EMA_21'].iloc[-2]:
        signals.append({
            'action': 'SELL',
            'price': current_price,
            'confidence': 0.75,
            'reason': 'EMA 9 crossed below EMA 21'
        })
    
    return signals
""",
            
            "RSI Strategy": """
def generate_signals(data):
    '''RSI Strategy'''
    import pandas as pd
    import numpy as np
    
    signals = []
    
    # Calculate RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    current_price = data['close'].iloc[-1]
    current_rsi = rsi.iloc[-1]
    
    # BUY when RSI < 30 (oversold)
    if current_rsi < 30:
        signals.append({
            'action': 'BUY',
            'price': current_price,
            'confidence': 0.70,
            'reason': f'RSI oversold ({current_rsi:.1f})'
        })
    
    # SELL when RSI > 70 (overbought)
    elif current_rsi > 70:
        signals.append({
            'action': 'SELL',
            'price': current_price,
            'confidence': 0.70,
            'reason': f'RSI overbought ({current_rsi:.1f})'
        })
    
    return signals
"""
        }
    
    def render(self):
        """Render the strategy editor"""
        st.subheader("💻 Strategy Editor")
        
        # Template selection
        col1, col2 = st.columns([3, 1])
        with col1:
            template = st.selectbox("Select Template", list(self.templates.keys()))
        with col2:
            if st.button("Load Template"):
                st.session_state.strategy_code = self.templates[template]
        
        # Initialize code in session state
        if 'strategy_code' not in st.session_state:
            st.session_state.strategy_code = self.templates["EMA Crossover"]
        
        # Code editor
        st.markdown('<div class="vscode-container">', unsafe_allow_html=True)
        st.markdown('<div class="vscode-header">📁 strategy.py</div>', unsafe_allow_html=True)
        
        edited_code = st.text_area(
            "Edit your strategy:",
            value=st.session_state.strategy_code,
            height=300,
            label_visibility="collapsed",
            key="code_editor"
        )
        
        st.session_state.strategy_code = edited_code
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Control buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("▶️ Test Strategy", type="primary"):
                self.test_strategy(edited_code)
        with col2:
            if st.button("💾 Save Strategy"):
                self.save_strategy(edited_code)
        with col3:
            if st.button("📊 Run Backtest"):
                self.run_backtest(edited_code)
    
    def test_strategy(self, code):
        """Test the strategy with sample data"""
        try:
            # Create execution namespace
            namespace = {}
            
            # Execute the code
            exec(code, namespace)
            
            # Check if generate_signals function exists
            if 'generate_signals' in namespace:
                # Create sample data
                np.random.seed(42)
                dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
                data = pd.DataFrame({
                    'open': np.random.normal(100, 5, 100),
                    'high': np.random.normal(102, 5, 100),
                    'low': np.random.normal(98, 5, 100),
                    'close': np.random.normal(100, 5, 100),
                    'volume': np.random.randint(1000, 10000, 100)
                }, index=dates)
                
                # Generate signals
                signals = namespace['generate_signals'](data)
                
                if signals:
                    st.success(f"✅ Generated {len(signals)} signals:")
                    for signal in signals:
                        st.json(signal)
                else:
                    st.info("No signals generated")
            else:
                st.error("❌ Function 'generate_signals(data)' not found in code")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    def save_strategy(self, code):
        """Save strategy to file"""
        try:
            with open("custom_strategy.py", "w") as f:
                f.write(code)
            st.success("✅ Strategy saved to custom_strategy.py")
        except Exception as e:
            st.error(f"Error saving: {str(e)}")
    
    def run_backtest(self, code):
        """Run backtest with historical data"""
        try:
            symbol = st.selectbox("Select Symbol", NIFTY_50[:10])
            
            if st.button("Run Backtest"):
                with st.spinner(f"Fetching data for {symbol}..."):
                    # Get historical data
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="1mo", interval="1h")
                    
                    if not data.empty:
                        # Prepare data
                        data = data.rename(columns={
                            'Open': 'open',
                            'High': 'high',
                            'Low': 'low',
                            'Close': 'close',
                            'Volume': 'volume'
                        })
                        
                        # Execute strategy
                        namespace = {}
                        exec(st.session_state.strategy_code, namespace)
                        
                        if 'generate_signals' in namespace:
                            signals = namespace['generate_signals'](data)
                            
                            if signals:
                                st.success(f"✅ Generated {len(signals)} signals")
                                
                                # Display results
                                signal_df = pd.DataFrame(signals)
                                st.dataframe(signal_df)
                                
                                # Simple chart
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=data.index,
                                    y=data['close'],
                                    mode='lines',
                                    name='Price',
                                    line=dict(color='blue', width=1)
                                ))
                                
                                # Add signal markers
                                buy_signals = [s for s in signals if s['action'] == 'BUY']
                                sell_signals = [s for s in signals if s['action'] == 'SELL']
                                
                                if buy_signals:
                                    buy_times = [data.index[-1]] * len(buy_signals)
                                    buy_prices = [s['price'] for s in buy_signals]
                                    fig.add_trace(go.Scatter(
                                        x=buy_times,
                                        y=buy_prices,
                                        mode='markers',
                                        name='BUY',
                                        marker=dict(color='green', size=10, symbol='triangle-up')
                                    ))
                                
                                if sell_signals:
                                    sell_times = [data.index[-1]] * len(sell_signals)
                                    sell_prices = [s['price'] for s in sell_signals]
                                    fig.add_trace(go.Scatter(
                                        x=sell_times,
                                        y=sell_prices,
                                        mode='markers',
                                        name='SELL',
                                        marker=dict(color='red', size=10, symbol='triangle-down')
                                    ))
                                
                                fig.update_layout(
                                    title=f"{symbol} - Strategy Signals",
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No signals generated")
                    else:
                        st.error("Failed to fetch data")
        except Exception as e:
            st.error(f"Backtest error: {str(e)}")

# =============================================
# Utility Functions
# =============================================
def now_indian():
    return datetime.now(IND_TZ)

def market_open():
    n = now_indian()
    try:
        open_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(9, 15)))
        close_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(15, 30)))
        return open_time <= n <= close_time
    except Exception:
        return False

def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rs = rs.fillna(0)
    return 100 - (100 / (1 + rs))

# =============================================
# Main Application
# =============================================
def main():
    # Initialize session state
    if 'page_loaded' not in st.session_state:
        st.session_state.page_loaded = True
    if 'kite_manager' not in st.session_state:
        st.session_state.kite_manager = KiteConnectManager()
    if 'strategy_editor' not in st.session_state:
        st.session_state.strategy_editor = StrategyEditor()
    
    # Initialize managers
    kite_manager = st.session_state.kite_manager
    strategy_editor = st.session_state.strategy_editor
    
    # Try to initialize Kite
    kite_initialized = kite_manager.initialize()
    
    # Title
    st.markdown("""
    <h1 style='text-align:center; color: #1e3a8a;'>
        📊 Rantv Intraday Terminal Pro
    </h1>
    <p style='text-align:center; color: #6b7280;'>
        VS Code Strategy Editor & Kite Connect Integration
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar - Kite Connect Status
    st.sidebar.header("🔗 Kite Connect Status")
    
    if kite_initialized and kite_manager.is_authenticated:
        st.sidebar.markdown(f"""
        <div class="kite-connected">
            <strong>✅ Connected</strong><br>
            User: {kite_manager.user_name}<br>
            Status: Live
        </div>
        """, unsafe_allow_html=True)
        
        if st.sidebar.button("Disconnect Kite", type="secondary"):
            kite_manager.logout()
            st.rerun()
    else:
        st.sidebar.markdown("""
        <div class="kite-disconnected">
            <strong>⚠️ Disconnected</strong><br>
            Connect for live trading
        </div>
        """, unsafe_allow_html=True)
        
        if st.sidebar.button("Connect Kite", type="primary"):
            kite_manager.login()
    
    # Sidebar - Quick Stats
    st.sidebar.header("📈 Market Status")
    st.sidebar.write(f"**Time:** {now_indian().strftime('%H:%M:%S')}")
    st.sidebar.write(f"**Market:** {'🟢 OPEN' if market_open() else '🔴 CLOSED'}")
    
    # Auto-refresh
    st_autorefresh(interval=15000, key="auto_refresh")
    
    # Main Tabs
    tabs = st.tabs([
        "📊 Dashboard",
        "🚦 Signals",
        "💻 Strategy Editor",
        "📈 Live Charts",
        "💰 Paper Trading"
    ])
    
    # Tab 1: Dashboard
    with tabs[0]:
        st.header("Trading Dashboard")
        
        # Quick Stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Available Cash", f"₹{CAPITAL:,.0f}")
        with col2:
            st.metric("Market Status", "OPEN" if market_open() else "CLOSED")
        with col3:
            st.metric("Kite Connect", "Connected" if kite_initialized else "Disconnected")
        with col4:
            st.metric("Auto Refresh", "15s")
        
        # Kite Connect Status Card
        if kite_initialized and kite_manager.is_authenticated:
            st.markdown(f"""
            <div class="kite-connected">
                <h3>✅ Kite Connect Active</h3>
                <p><strong>User:</strong> {kite_manager.user_name}</p>
                <p><strong>Features Available:</strong> Live Charts, Real-time Quotes, Historical Data</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="kite-disconnected">
                <h3>🔌 Connect Kite Connect</h3>
                <p>Connect to Kite for live trading features:</p>
                <ul>
                    <li>Real-time charts</li>
                    <li>Live quotes</li>
                    <li>Historical data</li>
                    <li>Paper trading</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick Actions
        st.subheader("Quick Actions")
        action_cols = st.columns(3)
        with action_cols[0]:
            if st.button("Generate Signals", type="primary", use_container_width=True):
                st.session_state.generate_signals = True
        with action_cols[1]:
            if st.button("View Charts", type="secondary", use_container_width=True):
                st.session_state.view_charts = True
        with action_cols[2]:
            if st.button("Open Editor", type="secondary", use_container_width=True):
                st.session_state.open_editor = True
    
    # Tab 2: Signals
    with tabs[1]:
        st.header("🚦 Trading Signals")
        
        # Signal Generation Options
        col1, col2, col3 = st.columns(3)
        with col1:
            signal_source = st.selectbox(
                "Signal Source",
                ["Yahoo Finance", "Kite Connect"],
                disabled=not kite_initialized
            )
        with col2:
            universe = st.selectbox(
                "Universe",
                ["Nifty 50", "Nifty 100", "All Stocks"]
            )
        with col3:
            if st.button("Generate Signals", type="primary"):
                st.session_state.generate_now = True
        
        if st.session_state.get('generate_now', False):
            with st.spinner("Generating signals..."):
                # Sample signal generation
                signals = []
                
                # Get selected universe
                if universe == "Nifty 50":
                    symbols = NIFTY_50[:5]  # Limit for demo
                elif universe == "Nifty 100":
                    symbols = NIFTY_50[:10]
                else:
                    symbols = NIFTY_50[:3]
                
                for symbol in symbols:
                    try:
                        # Get data
                        ticker = yf.Ticker(symbol)
                        data = ticker.history(period="1d", interval="15m")
                        
                        if not data.empty:
                            # Calculate indicators
                            current_price = data['Close'].iloc[-1]
                            ema_9 = calculate_ema(data['Close'], 9).iloc[-1]
                            ema_21 = calculate_ema(data['Close'], 21).iloc[-1]
                            rsi = calculate_rsi(data['Close']).iloc[-1]
                            
                            # Generate signals
                            if ema_9 > ema_21 and rsi < 70:
                                signals.append({
                                    'symbol': symbol,
                                    'action': 'BUY',
                                    'price': current_price,
                                    'confidence': 0.75,
                                    'reason': 'EMA bullish, RSI not overbought'
                                })
                            elif ema_9 < ema_21 and rsi > 30:
                                signals.append({
                                    'symbol': symbol,
                                    'action': 'SELL',
                                    'price': current_price,
                                    'confidence': 0.70,
                                    'reason': 'EMA bearish, RSI not oversold'
                                })
                    except:
                        continue
                
                # Display signals
                if signals:
                    st.success(f"✅ Generated {len(signals)} signals")
                    
                    for signal in signals:
                        action_class = "signal-buy" if signal['action'] == 'BUY' else "signal-sell"
                        action_emoji = "🟢" if signal['action'] == 'BUY' else "🔴"
                        
                        st.markdown(f"""
                        <div class="{action_class}">
                            <div style="display: flex; justify-content: space-between;">
                                <div>
                                    <strong>{action_emoji} {signal['symbol'].replace('.NS', '')}</strong><br>
                                    <small>{signal['reason']}</small>
                                </div>
                                <div style="text-align: right;">
                                    <strong>{signal['action']} @ ₹{signal['price']:.2f}</strong><br>
                                    <small>Confidence: {signal['confidence']:.0%}</small>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No signals generated")
                
                # Reset flag
                st.session_state.generate_now = False
    
    # Tab 3: Strategy Editor
    with tabs[2]:
        strategy_editor.render()
    
    # Tab 4: Live Charts
    with tabs[3]:
        st.header("📈 Live Charts")
        
        # Chart Options
        col1, col2, col3 = st.columns(3)
        with col1:
            chart_symbol = st.selectbox("Symbol", NIFTY_50[:15], key="chart_symbol")
        with col2:
            chart_interval = st.selectbox(
                "Interval",
                ["1m", "5m", "15m", "1h", "1d"],
                index=2
            )
        with col3:
            chart_type = st.selectbox("Chart Type", ["Candlestick", "Line"])
        
        if st.button("Load Chart", type="primary"):
            with st.spinner("Loading chart..."):
                try:
                    # Get data
                    ticker = yf.Ticker(chart_symbol)
                    
                    # Map interval for yfinance
                    interval_map = {
                        "1m": "1m", "5m": "5m", "15m": "15m",
                        "1h": "60m", "1d": "1d"
                    }
                    
                    period = "1d" if chart_interval in ["1m", "5m", "15m"] else "5d"
                    
                    data = ticker.history(
                        period=period,
                        interval=interval_map.get(chart_interval, "15m")
                    )
                    
                    if not data.empty:
                        # Create chart
                        fig = go.Figure()
                        
                        if chart_type == "Candlestick":
                            fig.add_trace(go.Candlestick(
                                x=data.index,
                                open=data['Open'],
                                high=data['High'],
                                low=data['Low'],
                                close=data['Close'],
                                name='Price'
                            ))
                        else:
                            fig.add_trace(go.Scatter(
                                x=data.index,
                                y=data['Close'],
                                mode='lines',
                                name='Price',
                                line=dict(color='#1e3a8a', width=2)
                            ))
                        
                        # Add EMA indicators
                        data['EMA_9'] = calculate_ema(data['Close'], 9)
                        data['EMA_21'] = calculate_ema(data['Close'], 21)
                        
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['EMA_9'],
                            mode='lines',
                            name='EMA 9',
                            line=dict(color='orange', width=1)
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['EMA_21'],
                            mode='lines',
                            name='EMA 21',
                            line=dict(color='blue', width=1)
                        ))
                        
                        fig.update_layout(
                            title=f"{chart_symbol} - Live Chart",
                            xaxis_title="Time",
                            yaxis_title="Price (₹)",
                            height=500,
                            template="plotly_white",
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show current stats
                        current_price = data['Close'].iloc[-1]
                        prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
                        change_pct = ((current_price - prev_close) / prev_close) * 100
                        
                        st.metric(
                            f"Current Price",
                            f"₹{current_price:.2f}",
                            f"{change_pct:+.2f}%"
                        )
                    else:
                        st.error("No data available")
                        
                except Exception as e:
                    st.error(f"Chart error: {str(e)}")
    
    # Tab 5: Paper Trading
    with tabs[4]:
        st.header("💰 Paper Trading")
        
        # Trading Panel
        col1, col2, col3 = st.columns(3)
        with col1:
            trade_symbol = st.selectbox("Symbol", NIFTY_50[:10], key="trade_symbol")
        with col2:
            trade_action = st.selectbox("Action", ["BUY", "SELL"], key="trade_action")
        with col3:
            trade_qty = st.number_input("Quantity", min_value=1, value=10, key="trade_qty")
        
        if st.button("Execute Trade", type="primary"):
            try:
                # Get current price
                ticker = yf.Ticker(trade_symbol)
                data = ticker.history(period="1d", interval="1m")
                
                if not data.empty:
                    price = data['Close'].iloc[-1]
                    total_value = price * trade_qty
                    
                    # Store trade in session
                    if 'paper_trades' not in st.session_state:
                        st.session_state.paper_trades = []
                    
                    trade = {
                        'id': len(st.session_state.paper_trades) + 1,
                        'symbol': trade_symbol,
                        'action': trade_action,
                        'quantity': trade_qty,
                        'price': price,
                        'value': total_value,
                        'time': now_indian().strftime("%H:%M:%S"),
                        'date': now_indian().strftime("%Y-%m-%d")
                    }
                    
                    st.session_state.paper_trades.append(trade)
                    
                    st.success(f"""
                    ✅ Trade Executed:
                    - {trade_action} {trade_qty} {trade_symbol.replace('.NS', '')}
                    - Price: ₹{price:.2f}
                    - Total: ₹{total_value:.2f}
                    """)
                else:
                    st.error("Could not fetch price")
                    
            except Exception as e:
                st.error(f"Trade error: {str(e)}")
        
        # Show trade history
        if 'paper_trades' in st.session_state and st.session_state.paper_trades:
            st.subheader("Trade History")
            
            # Convert to DataFrame for display
            trades_df = pd.DataFrame(st.session_state.paper_trades)
            
            # Format columns
            trades_df['price'] = trades_df['price'].apply(lambda x: f"₹{x:.2f}")
            trades_df['value'] = trades_df['value'].apply(lambda x: f"₹{x:.2f}")
            
            st.dataframe(
                trades_df[['time', 'symbol', 'action', 'quantity', 'price', 'value']],
                use_container_width=True
            )
            
            # Summary
            total_trades = len(trades_df)
            buy_trades = len(trades_df[trades_df['action'] == 'BUY'])
            sell_trades = len(trades_df[trades_df['action'] == 'SELL'])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Trades", total_trades)
            col2.metric("BUY Trades", buy_trades)
            col3.metric("SELL Trades", sell_trades)
        
        else:
            st.info("No trades yet. Execute a trade to see history here.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color: #6b7280; font-size: 12px;'>
        <p>Rantv Intraday Terminal Pro | VS Code Strategy Editor | Kite Connect Integration</p>
        <p>⚠️ This is for educational purposes only. Trading involves risk.</p>
    </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"App error: {e}")
        st.code(traceback.format_exc())
