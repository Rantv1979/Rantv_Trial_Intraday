# Rantv Intraday Trading Signals & Market Analysis - PRODUCTION READY
# ENHANCED VERSION WITH FULL STOCK SCANNING & BETTER SIGNAL QUALITY
# UPDATED: Fixed OAuth redirects, Added VS Code Strategy Editor, Enhanced Kite Connect
# INTEGRATED WITH KITE CONNECT FOR LIVE CHARTS & SIGNALS

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

# Kite Connect API Credentials - Use environment variables
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

st.set_page_config(
    page_title="Rantv Intraday Terminal Pro - Enhanced", 
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

# Clean and combine all stocks
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

# Enhanced Trading Strategies
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

HIGH_ACCURACY_STRATEGIES = {
    "Multi_Confirmation": {"name": "Multi-Confirmation Ultra", "weight": 5, "type": "BOTH"},
    "Enhanced_EMA_VWAP": {"name": "Enhanced EMA-VWAP", "weight": 4, "type": "BOTH"},
    "Volume_Breakout": {"name": "Volume Weighted Breakout", "weight": 4, "type": "BOTH"},
    "RSI_Divergence": {"name": "RSI Divergence", "weight": 3, "type": "BOTH"},
    "MACD_Trend": {"name": "MACD Trend Momentum", "weight": 3, "type": "BOTH"}
}

# FIXED CSS with improved layout
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #fff9e6 0%, #fff0d6 100%);
    }
    
    .main .block-container {
        background-color: transparent;
        padding-top: 2rem;
    }
    
    /* VS Code Editor Styling */
    .vscode-container {
        background: #1e1e1e;
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
        border: 1px solid #333;
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
        max-height: 600px;
        overflow-y: auto;
        white-space: pre-wrap;
        tab-size: 4;
    }
    
    .code-keyword { color: #569cd6; }
    .code-string { color: #ce9178; }
    .code-comment { color: #6a9955; }
    .code-function { color: #dcdcaa; }
    .code-number { color: #b5cea8; }
    
    /* Kite Connect Specific */
    .kite-connected { background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); }
    .kite-disconnected { background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); }
    
    /* Fix for redirects */
    .stAlert { margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# System Status Check
def check_system_status():
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
        "streamlit_autorefresh": True
    }
    return status

system_status = check_system_status()

# FIXED: Improved Kite Connect Manager without redirect loops
class KiteConnectManager:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.kite = None
        self.kws = None
        self.access_token = None
        self.is_authenticated = False
        self.user_name = ""
        self.tick_buffer = {}
        self.candle_store = {}
        self.ws_running = False
        
        # Initialize session state for auth
        if 'kite_auth_complete' not in st.session_state:
            st.session_state.kite_auth_complete = False
        if 'kite_access_token' not in st.session_state:
            st.session_state.kite_access_token = None
        if 'kite_user_name' not in st.session_state:
            st.session_state.kite_user_name = ""
    
    def handle_oauth_callback(self):
        """Handle OAuth callback without causing redirect loops"""
        try:
            query_params = st.query_params.to_dict()
            if "request_token" in query_params and not st.session_state.kite_auth_complete:
                request_token = query_params["request_token"]
                
                if not self.api_key or not self.api_secret:
                    st.error("Kite API credentials not configured")
                    return False
                
                # Create KiteConnect instance
                self.kite = KiteConnect(api_key=self.api_key)
                
                try:
                    # Generate session
                    data = self.kite.generate_session(request_token, api_secret=self.api_secret)
                    
                    if data and "access_token" in data:
                        self.access_token = data["access_token"]
                        self.kite.set_access_token(self.access_token)
                        self.is_authenticated = True
                        self.user_name = data.get("user_name", "")
                        
                        # Store in session state
                        st.session_state.kite_access_token = self.access_token
                        st.session_state.kite_user_name = self.user_name
                        st.session_state.kite_auth_complete = True
                        
                        # Clear query params safely
                        try:
                            # Create new params dict without request_token
                            new_params = {k: v for k, v in query_params.items() if k != "request_token"}
                            st.query_params.clear()
                            # Don't set empty params - this avoids redirect loops
                        except:
                            pass
                        
                        st.success(f"✅ Authenticated as {self.user_name}")
                        return True
                        
                except Exception as e:
                    st.error(f"Authentication failed: {str(e)}")
                    return False
                    
        except Exception as e:
            logger.error(f"OAuth callback error: {e}")
            
        return False
    
    def login(self):
        """Login to Kite Connect - Fixed version without redirect loops"""
        # First check if we're handling a callback
        if self.handle_oauth_callback():
            st.rerun()
            return True
        
        # Check if already authenticated in session
        if st.session_state.kite_auth_complete and st.session_state.kite_access_token:
            self.access_token = st.session_state.kite_access_token
            self.user_name = st.session_state.kite_user_name
            
            if not self.kite:
                self.kite = KiteConnect(api_key=self.api_key)
            self.kite.set_access_token(self.access_token)
            
            # Verify token is still valid
            try:
                profile = self.kite.profile()
                self.is_authenticated = True
                return True
            except:
                # Token expired or invalid
                st.session_state.kite_auth_complete = False
                st.session_state.kite_access_token = None
        
        # Not authenticated - show login options
        if not self.api_key or not self.api_secret:
            st.warning("Please configure KITE_API_KEY and KITE_API_SECRET in environment variables")
            return False
        
        self.kite = KiteConnect(api_key=self.api_key)
        
        # Show login UI
        st.info("Kite Connect authentication required for live trading")
        
        # Generate login URL
        login_url = self.kite.login_url()
        
        # Use markdown for login button to avoid automatic redirects
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; margin: 10px 0;">
            <h3 style="color: white; margin-bottom: 15px;">Connect to Zerodha Kite</h3>
            <a href="{login_url}" target="_blank" 
               style="display: inline-block; background: #f59e0b; color: white; 
                      padding: 12px 30px; border-radius: 8px; text-decoration: none; 
                      font-weight: bold; margin-bottom: 10px;">
                Login with Kite
            </a>
            <p style="color: #e0f2fe; margin-top: 15px; font-size: 12px;">
                After login, you'll get a request token. Paste it below:
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Manual token input
        with st.form("manual_kite_login"):
            request_token = st.text_input("Request Token", 
                                         help="Paste the request token from Kite after login",
                                         type="password")
            
            if st.form_submit_button("Authenticate", type="primary"):
                if request_token:
                    try:
                        data = self.kite.generate_session(request_token, api_secret=self.api_secret)
                        
                        if data and "access_token" in data:
                            self.access_token = data["access_token"]
                            self.kite.set_access_token(self.access_token)
                            self.is_authenticated = True
                            self.user_name = data.get("user_name", "")
                            
                            st.session_state.kite_access_token = self.access_token
                            st.session_state.kite_user_name = self.user_name
                            st.session_state.kite_auth_complete = True
                            
                            st.success(f"✅ Authenticated as {self.user_name}")
                            st.rerun()
                            return True
                    except Exception as e:
                        st.error(f"Authentication failed: {str(e)}")
        
        return False
    
    def logout(self):
        """Logout from Kite Connect"""
        try:
            st.session_state.kite_auth_complete = False
            st.session_state.kite_access_token = None
            st.session_state.kite_user_name = ""
            
            self.access_token = None
            self.is_authenticated = False
            self.user_name = ""
            
            if self.kws:
                self.kws.close()
                self.ws_running = False
                
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

# NEW: VS Code-like Strategy Editor
class VSCodeStrategyEditor:
    """VS Code-like interface for strategy development"""
    
    def __init__(self):
        self.strategy_templates = {
            "Basic EMA Crossover": """
import pandas as pd
import numpy as np

def generate_signals(data):
    '''
    Basic EMA Crossover Strategy
    data: DataFrame with OHLCV data
    Returns: List of signal dictionaries
    '''
    signals = []
    
    # Calculate EMAs
    data['EMA_9'] = data['close'].ewm(span=9, adjust=False).mean()
    data['EMA_21'] = data['close'].ewm(span=21, adjust=False).mean()
    
    # Generate signals
    current_price = data['close'].iloc[-1]
    ema_9 = data['EMA_9'].iloc[-1]
    ema_21 = data['EMA_21'].iloc[-1]
    
    if ema_9 > ema_21 and data['EMA_9'].iloc[-2] <= data['EMA_21'].iloc[-2]:
        # Golden Cross - BUY Signal
        signals.append({
            'action': 'BUY',
            'price': current_price,
            'confidence': 0.75,
            'reason': 'EMA 9 crossed above EMA 21'
        })
    
    elif ema_9 < ema_21 and data['EMA_9'].iloc[-2] >= data['EMA_21'].iloc[-2]:
        # Death Cross - SELL Signal
        signals.append({
            'action': 'SELL',
            'price': current_price,
            'confidence': 0.75,
            'reason': 'EMA 9 crossed below EMA 21'
        })
    
    return signals

# Test the strategy
if __name__ == "__main__":
    # Sample test data
    test_data = pd.DataFrame({
        'close': np.random.normal(100, 5, 100)
    })
    test_signals = generate_signals(test_data)
    print(f"Generated {len(test_signals)} signals")
""",
            
            "RSI Divergence Strategy": """
import pandas as pd
import numpy as np

def calculate_rsi(prices, period=14):
    '''Calculate RSI indicator'''
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(data):
    '''
    RSI Divergence Strategy
    '''
    signals = []
    
    # Calculate RSI
    data['RSI'] = calculate_rsi(data['close'])
    
    current_rsi = data['RSI'].iloc[-1]
    current_price = data['close'].iloc[-1]
    
    # Check for oversold/overbought conditions
    if current_rsi < 30:
        # Oversold - Potential BUY
        signals.append({
            'action': 'BUY',
            'price': current_price,
            'confidence': 0.70,
            'reason': f'RSI oversold ({current_rsi:.1f})'
        })
    
    elif current_rsi > 70:
        # Overbought - Potential SELL
        signals.append({
            'action': 'SELL',
            'price': current_price,
            'confidence': 0.70,
            'reason': f'RSI overbought ({current_rsi:.1f})'
        })
    
    return signals

# Test the strategy
if __name__ == "__main__":
    test_data = pd.DataFrame({
        'close': np.cumsum(np.random.randn(100)) + 100
    })
    test_signals = generate_signals(test_data)
    print(f"Generated {len(test_signals)} RSI signals")
"""
        }
    
    def render_editor(self):
        """Render the VS Code-like editor interface"""
        st.subheader("💻 VS Code Strategy Editor")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            strategy_name = st.selectbox(
                "Strategy Template",
                list(self.strategy_templates.keys()),
                key="strategy_template"
            )
        
        with col2:
            if st.button("Load Template", key="load_template"):
                st.session_state.current_code = self.strategy_templates[strategy_name]
                st.rerun()
        
        # Initialize code in session state
        if 'current_code' not in st.session_state:
            st.session_state.current_code = self.strategy_templates["Basic EMA Crossover"]
        
        # Editor container
        st.markdown('<div class="vscode-container">', unsafe_allow_html=True)
        st.markdown('<div class="vscode-header">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("📁 strategy.py")
        with col2:
            if st.button("Save", key="save_code"):
                # Save code to file
                with open("custom_strategy.py", "w") as f:
                    f.write(st.session_state.current_code)
                st.success("Strategy saved to custom_strategy.py")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Code editor (using text_area as a simple editor)
        edited_code = st.text_area(
            "Edit your strategy code:",
            value=st.session_state.current_code,
            height=400,
            label_visibility="collapsed",
            key="code_editor"
        )
        
        st.session_state.current_code = edited_code
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Execution controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("▶️ Run Strategy", type="primary", key="run_strategy"):
                self.execute_strategy(edited_code)
        
        with col2:
            if st.button("📊 Test with Sample Data", key="test_strategy"):
                self.test_with_sample_data(edited_code)
        
        with col3:
            if st.button("🔄 Reset", key="reset_code"):
                st.session_state.current_code = self.strategy_templates["Basic EMA Crossover"]
                st.rerun()
        
        # Show saved strategies
        self.show_saved_strategies()
    
    def execute_strategy(self, code):
        """Execute the strategy code"""
        try:
            # Create a temporary module
            import types
            module = types.ModuleType("custom_strategy")
            
            # Execute the code in the module context
            exec(code, module.__dict__)
            
            # Check if generate_signals function exists
            if hasattr(module, 'generate_signals'):
                st.success("✅ Strategy loaded successfully!")
                
                # Test with sample data
                import pandas as pd
                import numpy as np
                
                # Generate sample data
                sample_data = pd.DataFrame({
                    'open': np.random.normal(100, 2, 50),
                    'high': np.random.normal(102, 2, 50),
                    'low': np.random.normal(98, 2, 50),
                    'close': np.random.normal(100, 2, 50),
                    'volume': np.random.randint(1000, 10000, 50)
                })
                
                # Generate signals
                signals = module.generate_signals(sample_data)
                
                # Display results
                if signals:
                    st.success(f"✅ Generated {len(signals)} signals:")
                    for signal in signals:
                        st.json(signal)
                else:
                    st.info("No signals generated with sample data")
            else:
                st.error("❌ Strategy must define 'generate_signals(data)' function")
                
        except Exception as e:
            st.error(f"❌ Error executing strategy: {str(e)}")
            st.code(traceback.format_exc())
    
    def test_with_sample_data(self, code):
        """Test strategy with historical data"""
        try:
            # Import yfinance for real data
            symbol = st.selectbox("Test Symbol", NIFTY_50[:10], key="test_symbol")
            
            if st.button("Fetch & Test", key="fetch_test"):
                with st.spinner(f"Fetching data for {symbol}..."):
                    # Get real data
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="7d", interval="15m")
                    
                    if not data.empty:
                        # Prepare data for strategy
                        data = data.rename(columns={
                            'Open': 'open',
                            'High': 'high',
                            'Low': 'low',
                            'Close': 'close',
                            'Volume': 'volume'
                        })
                        
                        # Execute strategy
                        import types
                        module = types.ModuleType("test_strategy")
                        exec(code, module.__dict__)
                        
                        if hasattr(module, 'generate_signals'):
                            signals = module.generate_signals(data)
                            
                            if signals:
                                st.success(f"✅ Generated {len(signals)} signals for {symbol}:")
                                
                                # Create a table
                                signal_df = pd.DataFrame(signals)
                                st.dataframe(signal_df)
                                
                                # Plot
                                fig = go.Figure()
                                fig.add_trace(go.Candlestick(
                                    x=data.index,
                                    open=data['open'],
                                    high=data['high'],
                                    low=data['low'],
                                    close=data['close'],
                                    name='Price'
                                ))
                                
                                # Add buy/sell markers
                                buy_signals = [s for s in signals if s['action'] == 'BUY']
                                sell_signals = [s for s in signals if s['action'] == 'SELL']
                                
                                if buy_signals:
                                    buy_prices = [s['price'] for s in buy_signals]
                                    buy_times = [data.index[-1] for _ in buy_signals]  # Last timestamp
                                    fig.add_trace(go.Scatter(
                                        x=buy_times,
                                        y=buy_prices,
                                        mode='markers',
                                        name='BUY',
                                        marker=dict(color='green', size=10, symbol='triangle-up')
                                    ))
                                
                                if sell_signals:
                                    sell_prices = [s['price'] for s in sell_signals]
                                    sell_times = [data.index[-1] for _ in sell_signals]
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
                            st.error("Strategy must define 'generate_signals' function")
                    else:
                        st.error("Failed to fetch data")
                        
        except Exception as e:
            st.error(f"Test error: {str(e)}")
    
    def show_saved_strategies(self):
        """Show list of saved strategies"""
        if os.path.exists("custom_strategy.py"):
            with st.expander("📁 Saved Strategies", expanded=False):
                st.code(open("custom_strategy.py", "r").read(), language="python")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Load Saved", key="load_saved"):
                        st.session_state.current_code = open("custom_strategy.py", "r").read()
                        st.rerun()
                with col2:
                    if st.button("Delete", key="delete_strategy"):
                        os.remove("custom_strategy.py")
                        st.success("Strategy deleted")
                        st.rerun()

# NEW: Kite Connect Signal Generator
class KiteSignalGenerator:
    """Generate signals from Kite Connect data"""
    
    def __init__(self, kite_manager):
        self.kite = kite_manager
        self.instruments_cache = {}
        self.last_fetch_time = {}
        self.cache_duration = 60  # seconds
    
    def get_instrument_token(self, symbol):
        """Get instrument token for a symbol"""
        cache_key = f"token_{symbol}"
        
        # Check cache
        if cache_key in self.instruments_cache:
            cached_time, token = self.instruments_cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                return token
        
        # Fetch from Kite
        if self.kite.is_authenticated:
            try:
                # Search in instruments
                instruments = self.kite.get_instruments("NSE")
                for inst in instruments:
                    if inst['tradingsymbol'] == symbol.replace('.NS', ''):
                        token = inst['instrument_token']
                        self.instruments_cache[cache_key] = (time.time(), token)
                        return token
            except Exception as e:
                logger.error(f"Error fetching instrument token: {e}")
        
        return None
    
    def generate_signals_from_kite(self, symbols, interval="15minute", days=1):
        """Generate signals from Kite Connect data"""
        signals = []
        
        if not self.kite.is_authenticated:
            st.warning("Kite Connect not authenticated")
            return signals
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, symbol in enumerate(symbols[:20]):  # Limit to 20 symbols for performance
            status_text.text(f"Analyzing {symbol} ({idx+1}/{min(20, len(symbols))})")
            progress_bar.progress((idx + 1) / min(20, len(symbols)))
            
            try:
                # Get instrument token
                token = self.get_instrument_token(symbol)
                if not token:
                    continue
                
                # Get historical data from Kite
                data = self.kite.get_historical_data(token, interval, days)
                if data is None or len(data) < 20:
                    continue
                
                # Get live quote
                quote = self.kite.get_live_quote(token)
                if not quote:
                    continue
                
                # Prepare data
                df = pd.DataFrame({
                    'open': data['open'],
                    'high': data['high'],
                    'low': data['low'],
                    'close': data['close'],
                    'volume': data['volume']
                })
                
                # Calculate indicators
                df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
                df['EMA_21'] = df['close'].ewm(span=21, adjust=False).mean()
                
                # Calculate RSI
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                
                current_price = quote['last_price']
                ema_9 = df['EMA_9'].iloc[-1]
                ema_21 = df['EMA_21'].iloc[-1]
                rsi = df['RSI'].iloc[-1]
                
                # Generate signals
                if ema_9 > ema_21 and df['EMA_9'].iloc[-2] <= df['EMA_21'].iloc[-2]:
                    # Golden Cross
                    signals.append({
                        'symbol': symbol,
                        'action': 'BUY',
                        'price': current_price,
                        'confidence': 0.75,
                        'strategy': 'Kite_EMA_Crossover',
                        'reason': 'EMA 9 crossed above EMA 21',
                        'rsi': rsi,
                        'volume': quote['volume']
                    })
                
                elif rsi < 30:
                    # Oversold
                    signals.append({
                        'symbol': symbol,
                        'action': 'BUY',
                        'price': current_price,
                        'confidence': 0.70,
                        'strategy': 'Kite_RSI_Oversold',
                        'reason': f'RSI oversold ({rsi:.1f})',
                        'rsi': rsi,
                        'volume': quote['volume']
                    })
                
                elif ema_9 < ema_21 and df['EMA_9'].iloc[-2] >= df['EMA_21'].iloc[-2]:
                    # Death Cross
                    signals.append({
                        'symbol': symbol,
                        'action': 'SELL',
                        'price': current_price,
                        'confidence': 0.75,
                        'strategy': 'Kite_EMA_Crossover',
                        'reason': 'EMA 9 crossed below EMA 21',
                        'rsi': rsi,
                        'volume': quote['volume']
                    })
                
                elif rsi > 70:
                    # Overbought
                    signals.append({
                        'symbol': symbol,
                        'action': 'SELL',
                        'price': current_price,
                        'confidence': 0.70,
                        'strategy': 'Kite_RSI_Overbought',
                        'reason': f'RSI overbought ({rsi:.1f})',
                        'rsi': rsi,
                        'volume': quote['volume']
                    })
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        return signals

# Utility functions
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

def should_auto_close():
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

# Initialize Kite Manager
kite_manager = KiteConnectManager(KITE_API_KEY, KITE_API_SECRET)

# Initialize VS Code Editor
vscode_editor = VSCodeStrategyEditor()

# Initialize Kite Signal Generator
kite_signal_generator = KiteSignalGenerator(kite_manager)

# MAIN APPLICATION
def main():
    # Check system status
    with st.sidebar.expander("🛠️ System Status"):
        for package, status in system_status.items():
            if status:
                st.write(f"✅ {package}")
            else:
                st.write(f"❌ {package} - Missing")
    
    # Title and header
    st.markdown("<h1 style='text-align:center; color: #1e3a8a;'>Rantv Intraday Terminal Pro - ENHANCED</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center; color: #6b7280;'>VS Code Integration & Kite Connect Live Signals</h4>", unsafe_allow_html=True)
    
    # Kite Connect Status
    st.sidebar.header("🔗 Kite Connect")
    if kite_manager.is_authenticated or st.session_state.get('kite_auth_complete', False):
        st.sidebar.success(f"✅ Connected: {st.session_state.get('kite_user_name', 'User')}")
        if st.sidebar.button("Disconnect Kite", type="secondary"):
            kite_manager.logout()
            st.rerun()
    else:
        if st.sidebar.button("Connect Kite", type="primary"):
            kite_manager.login()
    
    # Auto-refresh
    st_autorefresh(interval=PRICE_REFRESH_MS, key="price_refresh")
    
    # Market status
    cols = st.columns(4)
    cols[0].metric("Market", "OPEN" if market_open() else "CLOSED")
    cols[1].metric("Time", now_indian().strftime("%H:%M:%S"))
    cols[2].metric("Date", now_indian().strftime("%d %b %Y"))
    cols[3].metric("Auto Refresh", f"{PRICE_REFRESH_MS//1000}s")
    
    # Main tabs
    tabs = st.tabs([
        "📈 Dashboard",
        "🚦 Kite Signals",
        "💻 VS Code Editor",
        "📊 Kite Live Charts",
        "💰 Paper Trading",
        "📋 Trade History"
    ])
    
    # Tab 1: Dashboard
    with tabs[0]:
        st.subheader("Trading Dashboard")
        
        # Kite Connect Status Card
        if kite_manager.is_authenticated:
            st.markdown("""
            <div class="kite-connected" style="padding: 20px; border-radius: 10px; margin: 10px 0;">
                <h3 style="color: #065f46;">✅ Kite Connect Connected</h3>
                <p><strong>User:</strong> {}</p>
                <p><strong>Status:</strong> Live trading enabled</p>
            </div>
            """.format(st.session_state.get('kite_user_name', 'User')), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="kite-disconnected" style="padding: 20px; border-radius: 10px; margin: 10px 0;">
                <h3 style="color: #7f1d1d;">⚠️ Kite Connect Disconnected</h3>
                <p>Connect Kite for live trading signals and charts</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick actions
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📊 Generate Kite Signals", type="primary", use_container_width=True):
                st.session_state.generate_kite_signals = True
        with col2:
            if st.button("📈 View Live Charts", type="secondary", use_container_width=True):
                st.session_state.view_charts = True
        with col3:
            if st.button("💻 Open Strategy Editor", type="secondary", use_container_width=True):
                st.session_state.open_editor = True
    
    # Tab 2: Kite Signals
    with tabs[1]:
        st.subheader("🚦 Kite Connect Live Signals")
        
        if not kite_manager.is_authenticated:
            st.warning("Please connect Kite Connect first to generate signals")
            if st.button("Connect Kite Connect", type="primary"):
                kite_manager.login()
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                universe = st.selectbox("Select Universe", ["Nifty 50", "Nifty 100", "All Stocks"], key="kite_universe")
            with col2:
                interval = st.selectbox("Interval", ["minute", "5minute", "15minute", "30minute"], key="kite_interval")
            with col3:
                days = st.slider("Days History", 1, 7, 2, key="kite_days")
            
            if st.button("Generate Signals from Kite", type="primary", key="gen_kite_signals"):
                with st.spinner("Fetching data from Kite Connect..."):
                    # Select symbols based on universe
                    if universe == "Nifty 50":
                        symbols = NIFTY_50[:20]  # Limit for performance
                    elif universe == "Nifty 100":
                        symbols = NIFTY_100[:30]
                    else:
                        symbols = ALL_STOCKS[:40]
                    
                    signals = kite_signal_generator.generate_signals_from_kite(symbols, interval, days)
                    
                    if signals:
                        st.success(f"✅ Generated {len(signals)} signals from Kite Connect")
                        
                        # Display signals
                        for signal in signals:
                            action_color = "🟢" if signal['action'] == 'BUY' else "🔴"
                            st.markdown(f"""
                            <div style="background: {'#d1fae5' if signal['action'] == 'BUY' else '#fee2e2'}; 
                                        padding: 15px; border-radius: 8px; margin: 10px 0; 
                                        border-left: 4px solid {'#059669' if signal['action'] == 'BUY' else '#dc2626'};">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <strong>{action_color} {signal['symbol'].replace('.NS', '')}</strong><br>
                                        <span style="font-size: 12px; color: #6b7280;">{signal['strategy']}</span>
                                    </div>
                                    <div style="text-align: right;">
                                        <strong>{signal['action']} @ ₹{signal['price']:.2f}</strong><br>
                                        <span style="font-size: 12px;">Confidence: {signal['confidence']:.0%}</span>
                                    </div>
                                </div>
                                <div style="margin-top: 8px; font-size: 13px;">
                                    {signal['reason']}<br>
                                    RSI: {signal.get('rsi', 0):.1f} | Volume: {signal.get('volume', 0):,}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Summary
                        buy_signals = [s for s in signals if s['action'] == 'BUY']
                        sell_signals = [s for s in signals if s['action'] == 'SELL']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("BUY Signals", len(buy_signals))
                        with col2:
                            st.metric("SELL Signals", len(sell_signals))
                    else:
                        st.info("No signals generated. Try different parameters.")
    
    # Tab 3: VS Code Editor
    with tabs[2]:
        vscode_editor.render_editor()
    
    # Tab 4: Kite Live Charts
    with tabs[3]:
        st.subheader("📊 Kite Connect Live Charts")
        
        if not kite_manager.is_authenticated:
            st.warning("Connect Kite Connect to view live charts")
            if st.button("Connect Kite", type="primary"):
                kite_manager.login()
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                symbol = st.selectbox("Select Symbol", NIFTY_50[:20], key="chart_symbol")
            
            with col2:
                chart_type = st.selectbox("Chart Type", ["Candlestick", "Line"], key="chart_type")
                interval = st.selectbox("Interval", ["minute", "5minute", "15minute", "30minute", "60minute"], key="chart_interval")
            
            if st.button("Load Chart", type="primary"):
                token = kite_signal_generator.get_instrument_token(symbol)
                if token:
                    with st.spinner("Fetching chart data..."):
                        data = kite_manager.get_historical_data(token, interval, days=2)
                        
                        if data is not None and not data.empty:
                            # Create chart
                            fig = go.Figure()
                            
                            if chart_type == "Candlestick":
                                fig.add_trace(go.Candlestick(
                                    x=data.index,
                                    open=data['open'],
                                    high=data['high'],
                                    low=data['low'],
                                    close=data['close'],
                                    name='Price'
                                ))
                            else:
                                fig.add_trace(go.Scatter(
                                    x=data.index,
                                    y=data['close'],
                                    mode='lines',
                                    name='Price',
                                    line=dict(color='#1e3a8a', width=2)
                                ))
                            
                            # Get live quote for current price
                            quote = kite_manager.get_live_quote(token)
                            if quote:
                                current_price = quote['last_price']
                                fig.add_hline(y=current_price, line_dash="dash", line_color="green",
                                            annotation_text=f"Current: ₹{current_price:.2f}")
                            
                            fig.update_layout(
                                title=f"{symbol} - Kite Connect Live Chart",
                                xaxis_title="Time",
                                yaxis_title="Price (₹)",
                                height=500,
                                template="plotly_white"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show quote details
                            if quote:
                                st.subheader("Live Quote")
                                q_cols = st.columns(4)
                                q_cols[0].metric("Last Price", f"₹{quote['last_price']:.2f}")
                                q_cols[1].metric("Change", f"₹{quote.get('change', 0):.2f}")
                                q_cols[2].metric("Volume", f"{quote.get('volume', 0):,}")
                                q_cols[3].metric("OI", f"{quote.get('oi', 0):,}")
                        else:
                            st.error("Could not fetch chart data")
                else:
                    st.error("Could not find instrument token")
    
    # Tab 5: Paper Trading (simplified)
    with tabs[4]:
        st.subheader("💰 Paper Trading Simulator")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.selectbox("Symbol", NIFTY_50[:15], key="paper_trade_symbol")
        with col2:
            action = st.selectbox("Action", ["BUY", "SELL"], key="paper_action")
        with col3:
            quantity = st.number_input("Quantity", min_value=1, value=10, key="paper_qty")
        
        if st.button("Execute Paper Trade", type="primary"):
            try:
                # Get price
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1d", interval="1m")
                price = float(data["Close"].iloc[-1]) if not data.empty else 1000
                
                st.success(f"Executed {action} {quantity} {symbol} @ ₹{price:.2f}")
                st.info(f"Trade Value: ₹{quantity * price:.2f}")
                
                # Store in session
                if 'paper_trades' not in st.session_state:
                    st.session_state.paper_trades = []
                
                st.session_state.paper_trades.append({
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'price': price,
                    'time': now_indian().strftime("%H:%M:%S"),
                    'value': quantity * price
                })
                
            except Exception as e:
                st.error(f"Trade execution failed: {str(e)}")
        
        # Show paper trades
        if 'paper_trades' in st.session_state and st.session_state.paper_trades:
            st.subheader("Recent Paper Trades")
            for trade in st.session_state.paper_trades[-5:]:  # Last 5 trades
                st.write(f"{trade['time']} - {trade['action']} {trade['quantity']} {trade['symbol']} @ ₹{trade['price']:.2f}")
    
    # Tab 6: Trade History
    with tabs[5]:
        st.subheader("📋 Trade History")
        
        # Placeholder for trade history
        st.info("Trade history will be stored here when you execute trades")
        
        if kite_manager.is_authenticated:
            st.write("**Kite Connect Status:** Active")
            st.write(f"**User:** {st.session_state.get('kite_user_name', 'N/A')}")
            st.write("**Features Enabled:** Live Charts, Real-time Signals")
        else:
            st.write("**Kite Connect Status:** Not Connected")
            st.write("Connect Kite for trade history synchronization")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color: #6b7280; font-size: 12px;'>
        <p>Rantv Intraday Terminal Pro | VS Code Strategy Editor | Kite Connect Integration</p>
        <p>⚠️ Paper trading only. Real trading involves risk.</p>
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
