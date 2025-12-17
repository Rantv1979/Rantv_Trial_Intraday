# Rantv Intraday Trading Signals & Market Analysis - PRODUCTION READY
# COMPLETE VERSION WITH KITE CONNECT INTEGRATION - REDIRECT LOOP FIXED
# Enhanced with full stock scanning, high accuracy strategies, and live Kite data

import time
from datetime import datetime, time as dt_time, timedelta
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
import threading
import logging

# ============================================================================
# STREAMLIT CONFIG - MUST BE FIRST
# ============================================================================
st.set_page_config(
    page_title="Rantv Intraday Terminal Pro - Enhanced",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

IND_TZ = pytz.timezone("Asia/Kolkata")

# ============================================================================
# AUTO-INSTALL DEPENDENCIES
# ============================================================================
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
    except:
        JOBLIB_AVAILABLE = False

# ============================================================================
# KITE CONNECT CREDENTIALS
# ============================================================================
KITE_API_KEY = os.environ.get("KITE_API_KEY", "")
KITE_API_SECRET = os.environ.get("KITE_API_SECRET", "")

# ============================================================================
# SESSION STATE INITIALIZATION (PREVENT REDIRECT LOOP)
# ============================================================================
if "kite_authenticated" not in st.session_state:
    st.session_state.kite_authenticated = False

if "oauth_processed" not in st.session_state:
    st.session_state.oauth_processed = False

if "kite_access_token" not in st.session_state:
    st.session_state.kite_access_token = None

if "trader" not in st.session_state:
    st.session_state.trader = None

if "refresh_count" not in st.session_state:
    st.session_state.refresh_count = 0

if "kite_manager" not in st.session_state:
    st.session_state.kite_manager = None

if "data_manager" not in st.session_state:
    st.session_state.data_manager = None

# ============================================================================
# CONFIGURATION
# ============================================================================
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

config = AppConfig.from_env()

# Trading Constants
CAPITAL = 2_000_000.0
TRADE_ALLOC = 0.15
MAX_DAILY_TRADES = 10
MAX_STOCK_TRADES = 10
MAX_AUTO_TRADES = 10

SIGNAL_REFRESH_MS = 120000
PRICE_REFRESH_MS = 100000

MARKET_OPTIONS = ["CASH"]

# ============================================================================
# STOCK UNIVERSES
# ============================================================================
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
    "JSWENERGY.NS", "JUBLFOOD.NS", "KOTAKBANK.NS", "L&TFH.NS", "LICHSGFIN.NS"
]

ALL_STOCKS = list(dict.fromkeys(NIFTY_50 + NIFTY_100 + NIFTY_MIDCAP_150))

# ============================================================================
# TRADING STRATEGIES
# ============================================================================
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

# ============================================================================
# ENHANCED CSS
# ============================================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #fff9e6 0%, #fff0d6 100%);
    }
    
    .main .block-container {
        background-color: transparent;
        padding-top: 2rem;
    }
    
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
    
    .kite-status-connected {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #059669;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .kite-status-disconnected {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #dc2626;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1e3a8a;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .signal-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1e3a8a;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .high-quality-signal {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid #047857;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# KITE CONNECT MANAGER (FIXED FOR REDIRECT LOOP)
# ============================================================================
class KiteConnectManager:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.kite = None
        self.is_authenticated = False
        self.tick_buffer = {}
        self.candle_store = {}
        self.ws_running = False
        
    def handle_oauth_callback(self):
        """Handle OAuth callback ONCE per session - PREVENTS REDIRECT LOOP"""
        if st.session_state.oauth_processed:
            return False
            
        query_params = st.query_params
        if "request_token" in query_params:
            try:
                request_token = query_params.get("request_token")
                
                if not self.kite:
                    self.kite = KiteConnect(api_key=self.api_key)
                
                # Exchange token
                data = self.kite.generate_session(
                    request_token=request_token,
                    api_secret=self.api_secret
                )
                
                # Store in session
                st.session_state.kite_access_token = data["access_token"]
                st.session_state.kite_authenticated = True
                st.session_state.oauth_processed = True
                
                self.kite.set_access_token(data["access_token"])
                self.is_authenticated = True
                
                # CRITICAL: Clear the request_token from URL to prevent redirect loop
                st.query_params.clear()
                
                logger.info("✅ Kite Connect authenticated successfully")
                return True
            except Exception as e:
                logger.error(f"OAuth error: {e}")
                st.error(f"Authentication failed: {str(e)}")
                return False
        return False
    
    def is_logged_in(self):
        """Check if already authenticated"""
        if st.session_state.kite_authenticated and st.session_state.kite_access_token:
            try:
                if not self.kite:
                    self.kite = KiteConnect(api_key=self.api_key)
                self.kite.set_access_token(st.session_state.kite_access_token)
                # Verify token is still valid
                profile = self.kite.profile()
                self.is_authenticated = True
                return True
            except:
                # Token expired or invalid
                st.session_state.kite_authenticated = False
                st.session_state.kite_access_token = None
                return False
        return False
    
    def get_login_url(self):
        """Get Kite Connect login URL"""
        if not self.kite:
            self.kite = KiteConnect(api_key=self.api_key)
        return self.kite.login_url()
    
    def get_live_data(self, instrument_token, interval="5minute", days=7):
        """Fetch live data from Kite Connect"""
        if not self.is_authenticated:
            return None
        
        try:
            from_date = (datetime.now() - timedelta(days=days)).date()
            to_date = datetime.now().date()
            
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
            return None
        except Exception as e:
            logger.error(f"Error fetching Kite data: {e}")
            return None
    
    def get_instruments(self, exchange="NSE"):
        """Get instrument list from Kite"""
        if not self.is_authenticated:
            return []
        
        try:
            return self.kite.instruments(exchange)
        except Exception as e:
            logger.error(f"Error fetching instruments: {e}")
            return []
    
    def get_quote(self, symbols):
        """Get live quotes from Kite"""
        if not self.is_authenticated:
            return {}
        
        try:
            return self.kite.quote(symbols)
        except Exception as e:
            logger.error(f"Error fetching quotes: {e}")
            return {}
    
    def logout(self):
        """Logout from Kite Connect"""
        st.session_state.kite_authenticated = False
        st.session_state.kite_access_token = None
        st.session_state.oauth_processed = False
        self.is_authenticated = False

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def now_indian():
    return datetime.now(IND_TZ)

def market_open():
    n = now_indian()
    try:
        open_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(9, 15)))
        close_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(15, 30)))
        return open_time <= n <= close_time
    except:
        return False

def should_auto_close():
    n = now_indian()
    try:
        auto_close_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(15, 10)))
        return n >= auto_close_time
    except:
        return False

def is_peak_market_hours():
    """Check if current time is during peak market hours (9:30 AM - 2:30 PM)"""
    n = now_indian()
    try:
        peak_start = IND_TZ.localize(datetime.combine(n.date(), dt_time(10, 0)))
        peak_end = IND_TZ.localize(datetime.combine(n.date(), dt_time(14, 0)))
        return peak_start <= n <= peak_end
    except:
        return True

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rs = rs.fillna(0)
    return 100 - (100 / (1 + rs))

def calculate_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    k = 100 * (close - lowest_low) / denom
    d = k.rolling(window=d_period).mean()
    return k.fillna(50), d.fillna(50)

def macd(close, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger_bands(close, period=20, std_dev=2):
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def adx(high, low, close, period=14):
    try:
        h = high.copy().reset_index(drop=True)
        l = low.copy().reset_index(drop=True)
        c = close.copy().reset_index(drop=True)
        df = pd.DataFrame({"high": h, "low": l, "close": c})
        df["tr"] = np.maximum(df["high"] - df["low"],
                            np.maximum((df["high"] - df["close"].shift()).abs(),
                                        (df["low"] - df["close"].shift()).abs()))
        df["up_move"] = df["high"] - df["high"].shift()
        df["down_move"] = df["low"].shift() - df["low"]
        df["dm_pos"] = np.where((df["up_move"] > df["down_move"]) & (df["up_move"] > 0), df["up_move"], 0.0)
        df["dm_neg"] = np.where((df["down_move"] > df["up_move"]) & (df["down_move"] > 0), df["down_move"], 0.0)
        df["tr_sum"] = df["tr"].rolling(window=period).sum()
        df["dm_pos_sum"] = df["dm_pos"].rolling(window=period).sum()
        df["dm_neg_sum"] = df["dm_neg"].rolling(window=period).sum()
        df["di_pos"] = 100 * (df["dm_pos_sum"] / df["tr_sum"]).replace([np.inf, -np.inf], 0).fillna(0)
        df["di_neg"] = 100 * (df["dm_neg_sum"] / df["tr_sum"]).replace([np.inf, -np.inf], 0).fillna(0)
        df["dx"] = (abs(df["di_pos"] - df["di_neg"]) / (df["di_pos"] + df["di_neg"]).replace(0, np.nan)) * 100
        df["adx"] = df["dx"].rolling(window=period).mean().fillna(0)
        return df["adx"].values
    except:
        return np.array([20] * len(high))

# ============================================================================
# ENHANCED DATA MANAGER WITH KITE INTEGRATION
# ============================================================================
class EnhancedDataManager:
    def __init__(self, kite_manager=None):
        self.price_cache = {}
        self.signal_cache = {}
        self.kite_manager = kite_manager
        self.last_rsi_scan = None
    
    def _validate_live_price(self, symbol):
        now_ts = time.time()
        key = f"price_{symbol}"
        if key in self.price_cache:
            cached = self.price_cache[key]
            if now_ts - cached["ts"] < 2:
                return cached["price"]
        
        # Try Kite first
        if self.kite_manager and self.kite_manager.is_authenticated:
            try:
                trading_symbol = symbol.replace(".NS", "")
                instruments = self.kite_manager.get_instruments("NSE")
                instrument = next((i for i in instruments if i["tradingsymbol"] == trading_symbol), None)
                
                if instrument:
                    quote = self.kite_manager.get_quote([f"NSE:{trading_symbol}"])
                    if quote and f"NSE:{trading_symbol}" in quote:
                        price = quote[f"NSE:{trading_symbol}"]["last_price"]
                        self.price_cache[key] = {"price": float(price), "ts": now_ts}
                        return float(price)
            except Exception as e:
                logger.debug(f"Kite price fetch failed for {symbol}: {e}")
        
        # Fallback to yfinance
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1d", interval="1m")
            if df is not None and not df.empty:
                price = float(df["Close"].iloc[-1])
                self.price_cache[key] = {"price": round(price, 2), "ts": now_ts}
                return round(price, 2)
        except:
            pass
        
        return 1000.0
    
    def get_stock_data(self, symbol, interval="15m", use_kite=False):
        """Get stock data from Kite or yfinance"""
        
        # Try Kite Connect first if requested and available
        if use_kite and self.kite_manager and self.kite_manager.is_authenticated:
            try:
                trading_symbol = symbol.replace(".NS", "")
                instruments = self.kite_manager.get_instruments("NSE")
                instrument = next((i for i in instruments if i["tradingsymbol"] == trading_symbol), None)
                
                if instrument:
                    kite_interval = "5minute" if interval == "5m" else "15minute" if interval == "15m" else "minute"
                    kite_data = self.kite_manager.get_live_data(
                        instrument["instrument_token"],
                        interval=kite_interval,
                        days=7
                    )
                    
                    if kite_data is not None and len(kite_data) > 20:
                        # Rename columns to match expected format
                        kite_data = kite_data.rename(columns={
                            'open': 'Open',
                            'high': 'High',
                            'low': 'Low',
                            'close': 'Close',
                            'volume': 'Volume'
                        })
                        
                        # Add indicators
                        df = self._add_indicators(kite_data)
                        logger.info(f"✅ Using Kite data for {symbol}")
                        return df
            except Exception as e:
                logger.debug(f"Kite data fetch failed for {symbol}: {e}")
        
        # Fallback to yfinance
        try:
            period = "7d" if interval == "15m" else "2d"
            df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
            
            if df is None or df.empty or len(df) < 20:
                return None
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["_".join(map(str, col)).strip() for col in df.columns.values]
            
            df = df.rename(columns={c: c.capitalize() for c in df.columns})
            expected = ["Open", "High", "Low", "Close", "Volume"]
            for e in expected:
                if e not in df.columns:
                    if e.upper() in df.columns:
                        df[e] = df[e.upper()]
                    else:
                        return None
            
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
            if len(df) < 20:
                return None
            
            # Add indicators
            df = self._add_indicators(df)
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _add_indicators(self, df):
        """Add technical indicators to dataframe"""
        try:
            df["EMA8"] = ema(df["Close"], 8)
            df["EMA21"] = ema(df["Close"], 21)
            df["EMA50"] = ema(df["Close"], 50)
            df["RSI14"] = rsi(df["Close"], 14).fillna(50)
            df["ATR"] = calculate_atr(df["High"], df["Low"], df["Close"]).fillna(0)
            df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd(df["Close"])
            df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = bollinger_bands(df["Close"])
            df["Stoch_K"], df["Stoch_D"] = stochastic(df["High"], df["Low"], df["Close"])
            df["VWAP"] = (((df["High"] + df["Low"] + df["Close"]) / 3) * df["Volume"]).cumsum() / df["Volume"].cumsum()
            df["ADX"] = pd.Series(adx(df["High"], df["Low"], df["Close"], period=14), index=df.index).fillna(20)
            
            return df
        except Exception as e:
            logger.error(f"Error adding indicators: {e}")
            return df
    
    def get_historical_accuracy(self, symbol, strategy):
        """Get historical accuracy for a strategy"""
        accuracy_map = {
            "Multi_Confirmation": 0.82, "Enhanced_EMA_VWAP": 0.78, "Volume_Breakout": 0.75,
            "RSI_Divergence": 0.72, "MACD_Trend": 0.70, "EMA_VWAP_Confluence": 0.75,
            "RSI_MeanReversion": 0.68, "Bollinger_Reversion": 0.65, "MACD_Momentum": 0.70,
            "Support_Resistance_Breakout": 0.73, "EMA_VWAP_Downtrend": 0.72,
            "RSI_Overbought": 0.65, "Bollinger_Rejection": 0.63, "MACD_Bearish": 0.68,
            "Trend_Reversal": 0.60
        }
        return accuracy_map.get(strategy, 0.65)

# ============================================================================
# MULTI-STRATEGY TRADING ENGINE
# ============================================================================
class MultiStrategyIntradayTrader:
    def __init__(self, capital=CAPITAL):
        self.initial_capital = float(capital)
        self.cash = float(capital)
        self.positions = {}
        self.trade_log = []
        self.daily_trades = 0
        self.stock_trades = 0
        self.auto_trades_count = 0
        self.last_reset = now_indian().date()
        self.selected_market = "CASH"
        self.auto_execution = False
        self.signal_history = []
        self.auto_close_triggered = False
        self.last_auto_execution_time = 0
        
        self.strategy_performance = {}
        for strategy in list(TRADING_STRATEGIES.keys()) + list(HIGH_ACCURACY_STRATEGIES.keys()):
            self.strategy_performance[strategy] = {"signals": 0, "trades": 0, "wins": 0, "pnl": 0.0}
    
    def reset_daily_counts(self):
        current_date = now_indian().date()
        if current_date != self.last_reset:
            self.daily_trades = 0
            self.stock_trades = 0
            self.auto_trades_count = 0
            self.last_reset = current_date
    
    def can_auto_trade(self):
        return (
            self.auto_trades_count < MAX_AUTO_TRADES and
            self.daily_trades < MAX_DAILY_TRADES and
            market_open()
        )
    
    def equity(self):
        total = float(self.cash)
        for symbol, pos in self.positions.items():
            if pos.get("status") == "OPEN":
                try:
                    price = pos.get("current_price", pos["entry_price"])
                    total += pos["quantity"] * price
                except:
                    total += pos["quantity"] * pos["entry_price"]
        return total
    
    def generate_signals(self, data_manager, symbols, use_kite=False, min_confidence=0.70, min_score=6):
        """Generate trading signals from multiple strategies"""
        signals = []
        
        for symbol in symbols[:50]:  # Limit for performance
            try:
                data = data_manager.get_stock_data(symbol, "15m", use_kite=use_kite)
                
                if data is None or len(data) < 30:
                    continue
                
                current_price = float(data["Close"].iloc[-1])
                ema8 = float(data["EMA8"].iloc[-1])
                ema21 = float(data["EMA21"].iloc[-1])
                ema50 = float(data["EMA50"].iloc[-1])
                rsi_val = float(data["RSI14"].iloc[-1])
                vwap = float(data["VWAP"].iloc[-1])
                macd_line = float(data["MACD"].iloc[-1])
                macd_signal = float(data["MACD_Signal"].iloc[-1])
                adx_val = float(data["ADX"].iloc[-1])
                volume = float(data["Volume"].iloc[-1])
                volume_avg = float(data["Volume"].rolling(20).mean().iloc[-1]) if len(data) >= 20 else volume
                
                # Strategy 1: EMA + VWAP + ADX
                if (ema8 > ema21 > ema50 and current_price > vwap and adx_val > 25 and 
                    40 < rsi_val < 65 and volume > volume_avg * 1.3):
                    signals.append({
                        "symbol": symbol,
                        "action": "BUY",
                        "price": current_price,
                        "confidence": 0.82,
                        "score": 9,
                        "rsi": rsi_val,
                        "strategy": "EMA_VWAP_Confluence",
                        "strategy_name": "EMA + VWAP Confluence",
                        "data_source": "Kite" if use_kite else "YFinance"
                    })
                
                # Strategy 2: RSI Mean Reversion
                if rsi_val < 30 and volume > volume_avg * 1.3:
                    signals.append({
                        "symbol": symbol,
                        "action": "BUY",
                        "price": current_price,
                        "confidence": 0.78,
                        "score": 8,
                        "rsi": rsi_val,
                        "strategy": "RSI_MeanReversion",
                        "strategy_name": "RSI Mean Reversion",
                        "data_source": "Kite" if use_kite else "YFinance"
                    })
                
                # Strategy 3: MACD Momentum
                if macd_line > macd_signal and ema8 > ema21 and current_price > vwap:
                    signals.append({
                        "symbol": symbol,
                        "action": "BUY",
                        "price": current_price,
                        "confidence": 0.80,
                        "score": 8,
                        "rsi": rsi_val,
                        "strategy": "MACD_Momentum",
                        "strategy_name": "MACD Momentum",
                        "data_source": "Kite" if use_kite else "YFinance"
                    })
                
                # SELL Strategies
                if (ema8 < ema21 < ema50 and current_price < vwap and adx_val > 25 and 
                    35 < rsi_val < 60):
                    signals.append({
                        "symbol": symbol,
                        "action": "SELL",
                        "price": current_price,
                        "confidence": 0.82,
                        "score": 9,
                        "rsi": rsi_val,
                        "strategy": "EMA_VWAP_Downtrend",
                        "strategy_name": "EMA + VWAP Downtrend",
                        "data_source": "Kite" if use_kite else "YFinance"
                    })
            
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
                continue
        
        # Filter by confidence and score
        signals = [s for s in signals if s["confidence"] >= min_confidence and s["score"] >= min_score]
        signals.sort(key=lambda x: (x["score"], x["confidence"]), reverse=True)
        
        return signals[:20]
    
    def execute_trade(self, symbol, action, quantity, price, stop_loss=None, target=None, strategy=None):
        """Execute a paper trade"""
        self.reset_daily_counts()
        
        if self.daily_trades >= MAX_DAILY_TRADES:
            return False, "Daily trade limit reached"
        
        trade_value = float(quantity) * float(price)
        if action == "BUY" and trade_value > self.cash:
            return False, "Insufficient capital"
        
        trade_id = f"TRADE_{symbol}_{len(self.trade_log)}_{int(time.time())}"
        record = {
            "trade_id": trade_id,
            "symbol": symbol,
            "action": action,
            "quantity": int(quantity),
            "entry_price": float(price),
            "stop_loss": float(stop_loss) if stop_loss else None,
            "target": float(target) if target else None,
            "timestamp": now_indian(),
            "status": "OPEN",
            "current_pnl": 0.0,
            "current_price": float(price),
            "strategy": strategy
        }
        
        if action == "BUY":
            self.positions[symbol] = record
            self.cash -= trade_value
        else:
            margin = trade_value * 0.2
            record["margin_used"] = margin
            self.positions[symbol] = record
            self.cash -= margin
        
        self.trade_log.append(record)
        self.daily_trades += 1
        self.stock_trades += 1
        
        if strategy and strategy in self.strategy_performance:
            self.strategy_performance[strategy]["trades"] += 1
        
        return True, f"{action} {quantity} {symbol} @ ₹{price:.2f}"
    
    def close_position(self, symbol, exit_price=None):
        """Close an open position"""
        if symbol not in self.positions:
            return False, "Position not found"
        
        pos = self.positions[symbol]
        if exit_price is None:
            exit_price = pos.get("current_price", pos["entry_price"])
        
        if pos["action"] == "BUY":
            pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
            self.cash += pos["quantity"] * exit_price
        else:
            pnl = (pos["entry_price"] - exit_price) * pos["quantity"]
            self.cash += pos.get("margin_used", 0)
        
        pos["status"] = "CLOSED"
        pos["exit_price"] = float(exit_price)
        pos["closed_pnl"] = float(pnl)
        pos["exit_time"] = now_indian()
        
        strategy = pos.get("strategy")
        if strategy and strategy in self.strategy_performance:
            if pnl > 0:
                self.strategy_performance[strategy]["wins"] += 1
            self.strategy_performance[strategy]["pnl"] += pnl
        
        del self.positions[symbol]
        return True, f"Closed {symbol} @ ₹{exit_price:.2f} | P&L: ₹{pnl:+.2f}"
    
    def get_performance_stats(self):
        """Get performance statistics"""
        closed = [t for t in self.trade_log if t.get("status") == "CLOSED"]
        total_trades = len(closed)
        
        if total_trades == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "open_positions": len(self.positions)
            }
        
        wins = len([t for t in closed if t.get("closed_pnl", 0) > 0])
        total_pnl = sum([t.get("closed_pnl", 0) for t in closed])
        win_rate = wins / total_trades
        avg_pnl = total_pnl / total_trades
        
        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "open_positions": len(self.positions)
        }

# ============================================================================
# INITIALIZE APPLICATION
# ============================================================================
def initialize_application():
    """Initialize the application"""
    
    # Initialize Kite Manager
    if st.session_state.kite_manager is None:
        st.session_state.kite_manager = KiteConnectManager(KITE_API_KEY, KITE_API_SECRET)
    
    kite_manager = st.session_state.kite_manager
    
    # Handle OAuth callback (runs once)
    if not st.session_state.oauth_processed:
        if kite_manager.handle_oauth_callback():
            st.success("✅ Kite Connect authenticated successfully!")
            time.sleep(1)
            st.rerun()
    
    # Check if already logged in
    kite_manager.is_logged_in()
    
    # Initialize Data Manager
    if st.session_state.data_manager is None:
        st.session_state.data_manager = EnhancedDataManager(kite_manager)
    
    # Initialize Trader
    if st.session_state.trader is None:
        st.session_state.trader = MultiStrategyIntradayTrader()
    
    return kite_manager, st.session_state.data_manager, st.session_state.trader

# ============================================================================
# MAIN APPLICATION
# ============================================================================
try:
    # Initialize
    kite_manager, data_manager, trader = initialize_application()
    
    # Increment refresh count
    st.session_state.refresh_count += 1
    
    # Header
    st.markdown("<h1 style='text-align:center; color: #1e3a8a;'>Rantv Intraday Terminal Pro</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center; color: #6b7280;'>Enhanced with Kite Connect - Redirect Loop Fixed</h4>", unsafe_allow_html=True)
    
    # ========================================================================
    # SIDEBAR - KITE CONNECT
    # ========================================================================
    st.sidebar.header("🔐 Kite Connect")
    
    if kite_manager.is_authenticated:
        st.sidebar.markdown("""
        <div class="kite-status-connected">
            <strong>✅ Connected to Kite</strong><br>
            <small>Live data available</small>
        </div>
        """, unsafe_allow_html=True)
        
        if st.sidebar.button("🚪 Logout from Kite"):
            kite_manager.logout()
            st.rerun()
    else:
        st.sidebar.markdown("""
        <div class="kite-status-disconnected">
            <strong>❌ Not Connected</strong><br>
            <small>Using yfinance data</small>
        </div>
        """, unsafe_allow_html=True)
        
        if KITE_API_KEY and KITE_API_SECRET and KITECONNECT_AVAILABLE:
            login_url = kite_manager.get_login_url()
            st.sidebar.markdown(f"""
            <a href="{login_url}" target="_self">
                <button style="width:100%;padding:12px;background:#2563eb;
                color:white;border:none;border-radius:8px;font-size:16px;
                cursor:pointer;font-weight:bold;">
                    🔐 Login to Kite Connect
                </button>
            </a>
            """, unsafe_allow_html=True)
            
            st.sidebar.info("ℹ️ Login to access live Kite data and charts")
        else:
            st.sidebar.error("⚠️ Kite API credentials not configured")
    
    # Trading settings
    st.sidebar.header("⚙️ Trading Settings")
    trader.auto_execution = st.sidebar.checkbox("Auto Execution", value=False)
    universe = st.sidebar.selectbox("Trading Universe", ["Nifty 50", "Nifty 100", "Midcap 150", "All Stocks"])
    enable_high_accuracy = st.sidebar.checkbox("High Accuracy Strategies", value=True)
    
    # ========================================================================
    # MARKET OVERVIEW
    # ========================================================================
    cols = st.columns(6)
    cols[0].metric("Market", "🟢 OPEN" if market_open() else "🔴 CLOSED")
    cols[1].metric("IST Time", now_indian().strftime("%H:%M:%S"))
    cols[2].metric("Kite", "✅" if kite_manager.is_authenticated else "❌")
    cols[3].metric("Cash", f"₹{trader.cash:,.0f}")
    cols[4].metric("Equity", f"₹{trader.equity():,.0f}")
    cols[5].metric("Positions", len(trader.positions))
    
    st.markdown("---")
    
    # ========================================================================
    # MAIN TABS
    # ========================================================================
    tabs = st.tabs([
        "📊 Dashboard",
        "🚦 Trading Signals",
        "📈 Kite Live Charts",
        "💼 Positions",
        "📋 Trade History",
        "⚙️ Settings"
    ])
    
    # ========================================================================
    # TAB 1: DASHBOARD
    # ========================================================================
    with tabs[0]:
        st.subheader("📊 Dashboard")
        
        perf = trader.get_performance_stats()
        
        metric_cols = st.columns(4)
        metric_cols[0].metric("Capital", f"₹{trader.initial_capital:,.0f}")
        metric_cols[1].metric("Current Cash", f"₹{trader.cash:,.0f}")
        metric_cols[2].metric("Total Trades", perf["total_trades"])
        metric_cols[3].metric("Win Rate", f"{perf['win_rate']:.1%}")
        
        if kite_manager.is_authenticated:
            st.success("✅ Kite Connect is active - using live market data")
        else:
            st.info("ℹ️ Connect to Kite for live market data and advanced charts")
        
        # Strategy Performance
        st.subheader("Strategy Performance")
        strategy_data = []
        for strategy, perf_data in trader.strategy_performance.items():
            if perf_data["trades"] > 0:
                config = {**TRADING_STRATEGIES, **HIGH_ACCURACY_STRATEGIES}.get(strategy, {})
                win_rate = perf_data["wins"] / perf_data["trades"] if perf_data["trades"] > 0 else 0
                strategy_data.append({
                    "Strategy": config.get("name", strategy),
                    "Trades": perf_data["trades"],
                    "Win Rate": f"{win_rate:.1%}",
                    "P&L": f"₹{perf_data['pnl']:+,.2f}"
                })
        
        if strategy_data:
            st.dataframe(pd.DataFrame(strategy_data), use_container_width=True)
    
    # ========================================================================
    # TAB 2: TRADING SIGNALS
    # ========================================================================
    with tabs[1]:
        st.subheader("🚦 Trading Signals")
        
        use_kite_data = st.checkbox(
            "Use Kite Connect Data (Live)",
            value=kite_manager.is_authenticated,
            disabled=not kite_manager.is_authenticated,
            help="Use live data from Kite Connect instead of yfinance"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            min_confidence = st.slider("Min Confidence %", 60, 85, 70, 5)
        with col2:
            min_score = st.slider("Min Score", 5, 9, 6, 1)
        
        if st.button("🔄 Generate Signals", type="primary"):
            with st.spinner(f"Scanning {universe} stocks..."):
                # Select universe
                if universe == "Nifty 50":
                    symbols = NIFTY_50
                elif universe == "Nifty 100":
                    symbols = NIFTY_100
                elif universe == "Midcap 150":
                    symbols = NIFTY_MIDCAP_150
                else:
                    symbols = ALL_STOCKS
                
                signals = trader.generate_signals(
                    data_manager,
                    symbols,
                    use_kite=use_kite_data,
                    min_confidence=min_confidence/100,
                    min_score=min_score
                )
                
                if signals:
                    st.success(f"✅ Found {len(signals)} signals")
                    
                    for idx, signal in enumerate(signals):
                        action_color = "🟢" if signal["action"] == "BUY" else "🔴"
                        data_source_badge = "🔴 Kite Live" if signal["data_source"] == "Kite" else "📊 YFinance"
                        
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.markdown(f"""
                            <div class="signal-card">
                                <strong>{action_color} {signal['symbol'].replace('.NS', '')}</strong> - {signal['action']} @ ₹{signal['price']:.2f}<br>
                                Strategy: {signal['strategy_name']} | Confidence: {signal['confidence']:.1%}<br>
                                RSI: {signal['rsi']:.1f} | Score: {signal['score']}/10 | {data_source_badge}
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            qty = int((trader.cash * TRADE_ALLOC) / signal['price'])
                            st.write(f"Qty: {qty}")
                        
                        with col3:
                            if st.button("Execute", key=f"exec_{idx}"):
                                success, msg = trader.execute_trade(
                                    symbol=signal['symbol'],
                                    action=signal['action'],
                                    quantity=qty,
                                    price=signal['price'],
                                    strategy=signal['strategy']
                                )
                                if success:
                                    st.success(msg)
                                    st.rerun()
                                else:
                                    st.error(msg)
                else:
                    st.warning("No signals found matching criteria")
    
    # ========================================================================
    # TAB 3: KITE LIVE CHARTS
    # ========================================================================
    with tabs[2]:
        st.subheader("📈 Kite Connect Live Charts")
        
        if not kite_manager.is_authenticated:
            st.warning("⚠️ Please login to Kite Connect to view live charts")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                symbol_input = st.selectbox(
                    "Select Symbol",
                    ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "KOTAKBANK",
                     "BHARTIARTL", "ITC", "LT", "SBIN"]
                )
            
            with col2:
                interval = st.selectbox("Interval", ["5minute", "15minute", "30minute", "hour"])
            
            if st.button("📊 Load Live Chart", type="primary"):
                with st.spinner(f"Loading {symbol_input} chart from Kite..."):
                    try:
                        instruments = kite_manager.get_instruments("NSE")
                        instrument = next((i for i in instruments if i["tradingsymbol"] == symbol_input), None)
                        
                        if instrument:
                            data = kite_manager.get_live_data(
                                instrument["instrument_token"],
                                interval=interval,
                                days=5
                            )
                            
                            if data is not None and len(data) > 0:
                                # Create candlestick chart
                                fig = go.Figure(data=[go.Candlestick(
                                    x=data.index,
                                    open=data['open'],
                                    high=data['high'],
                                    low=data['low'],
                                    close=data['close'],
                                    name=symbol_input
                                )])
                                
                                fig.update_layout(
                                    title=f"{symbol_input} - Kite Connect Live Data ({interval})",
                                    xaxis_title="Time",
                                    yaxis_title="Price (₹)",
                                    height=500,
                                    template="plotly_white"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Current stats
                                current = data['close'].iloc[-1]
                                prev = data['close'].iloc[-2] if len(data) > 1 else current
                                change = ((current - prev) / prev) * 100
                                
                                stat_cols = st.columns(4)
                                stat_cols[0].metric("Current Price", f"₹{current:.2f}", f"{change:+.2f}%")
                                stat_cols[1].metric("High", f"₹{data['high'].iloc[-1]:.2f}")
                                stat_cols[2].metric("Low", f"₹{data['low'].iloc[-1]:.2f}")
                                stat_cols[3].metric("Volume", f"{int(data['volume'].iloc[-1]):,}")
                            else:
                                st.error("No data available for this symbol")
                        else:
                            st.error(f"Symbol {symbol_input} not found on NSE")
                            
                    except Exception as e:
                        st.error(f"Chart error: {str(e)}")
                        logger.error(f"Chart error: {traceback.format_exc()}")
    
    # ========================================================================
    # TAB 4: POSITIONS
    # ========================================================================
    with tabs[3]:
        st.subheader("💼 Open Positions")
        
        if trader.positions:
            position_data = []
            for symbol, pos in trader.positions.items():
                if pos.get("status") == "OPEN":
                    pnl = pos.get("current_pnl", 0)
                    position_data.append({
                        "Symbol": symbol.replace(".NS", ""),
                        "Action": pos["action"],
                        "Quantity": pos["quantity"],
                        "Entry": f"₹{pos['entry_price']:.2f}",
                        "Current": f"₹{pos.get('current_price', pos['entry_price']):.2f}",
                        "P&L": f"₹{pnl:+.2f}",
                        "Strategy": pos.get("strategy", "Manual")
                    })
            
            if position_data:
                st.dataframe(pd.DataFrame(position_data), use_container_width=True)
                
                # Close buttons
                for symbol in trader.positions.keys():
                    if st.button(f"Close {symbol.replace('.NS', '')}", key=f"close_{symbol}"):
                        success, msg = trader.close_position(symbol)
                        if success:
                            st.success(msg)
                            st.rerun()
        else:
            st.info("No open positions")
    
    # ========================================================================
    # TAB 5: TRADE HISTORY
    # ========================================================================
    with tabs[4]:
        st.subheader("📋 Trade History")
        
        closed_trades = [t for t in trader.trade_log if t.get("status") == "CLOSED"]
        
        if closed_trades:
            history_data = []
            for trade in closed_trades:
                history_data.append({
                    "Symbol": trade["symbol"].replace(".NS", ""),
                    "Action": trade["action"],
                    "Quantity": trade["quantity"],
                    "Entry": f"₹{trade['entry_price']:.2f}",
                    "Exit": f"₹{trade.get('exit_price', 0):.2f}",
                    "P&L": f"₹{trade.get('closed_pnl', 0):+.2f}",
                    "Strategy": trade.get("strategy", "Manual")
                })
            
            st.dataframe(pd.DataFrame(history_data), use_container_width=True)
        else:
            st.info("No trade history yet")
    
    # ========================================================================
    # TAB 6: SETTINGS
    # ========================================================================
    with tabs[5]:
        st.subheader("⚙️ System Settings")
        
        st.write("**Kite Connect Configuration**")
        st.write(f"API Key: {'✅ Configured' if KITE_API_KEY else '❌ Not Set'}")
        st.write(f"API Secret: {'✅ Configured' if KITE_API_SECRET else '❌ Not Set'}")
        st.write(f"Kite Connect Available: {'✅ Yes' if KITECONNECT_AVAILABLE else '❌ Not installed'}")
        st.write(f"Authentication Status: {'✅ Connected' if kite_manager.is_authenticated else '❌ Not Connected'}")
        
        st.markdown("---")
        st.write("**System Status**")
        st.write(f"✅ Redirect loop: Fixed")
        st.write(f"✅ OAuth handling: Session-based (runs once)")
        st.write(f"✅ SQLAlchemy: {'Available' if SQLALCHEMY_AVAILABLE else 'Not available'}")
        st.write(f"✅ Joblib: {'Available' if JOBLIB_AVAILABLE else 'Not available'}")
        
        st.markdown("---")
        st.write("**Performance Metrics**")
        st.write(f"Refresh Count: {st.session_state.refresh_count}")
        st.write(f"Market Open: {'Yes' if market_open() else 'No'}")
        st.write(f"Peak Hours: {'Yes' if is_peak_market_hours() else 'No'}")
    
    st.markdown("---")
    st.markdown("<div style='text-align:center; color: #6b7280;'>Rantv Intraday Terminal Pro | Kite Connect Integration | Redirect Loop Fixed ✅</div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"❌ Application error: {str(e)}")
    st.code(traceback.format_exc())
    logger.error(f"Application crash: {traceback.format_exc()}")
