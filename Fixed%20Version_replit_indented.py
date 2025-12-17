# Rantv Intraday Trading Signals & Market Analysis - KITE CONNECT FIXED
# FIXES: Redirect loop, OAuth handling, live Kite data integration
# Enhanced with proper session state management and Kite Connect charts

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
import warnings
import os
import logging
import traceback
import subprocess
import sys

# ============================================================================
# STREAMLIT CONFIG - MUST BE FIRST
# ============================================================================
st.set_page_config(
    page_title="Rantv Intraday Terminal Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

IND_TZ = pytz.timezone("Asia/Kolkata")

# ============================================================================
# KITE CONNECT SETUP WITH REDIRECT FIX
# ============================================================================
KITE_API_KEY = os.environ.get("KITE_API_KEY", "")
KITE_API_SECRET = os.environ.get("KITE_API_SECRET", "")

# Import Kite Connect
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

# ============================================================================
# TRADING CONSTANTS
# ============================================================================
CAPITAL = 2_000_000.0
TRADE_ALLOC = 0.15
MAX_DAILY_TRADES = 10
MAX_STOCK_TRADES = 10
MAX_AUTO_TRADES = 10

SIGNAL_REFRESH_MS = 120000
PRICE_REFRESH_MS = 100000

# Stock Universes
NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "LT.NS",
    "SBIN.NS", "ASIANPAINT.NS", "HCLTECH.NS", "AXISBANK.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "NTPC.NS"
]

# ============================================================================
# KITE CONNECT MANAGER (FIXED FOR REDIRECT LOOP)
# ============================================================================
class KiteConnectManager:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.kite = None
        self.is_authenticated = False
        
    def handle_oauth_callback(self):
        """Handle OAuth callback ONCE per session"""
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
                
                # CRITICAL: Clear the request_token from URL
                st.query_params.clear()
                
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
    
    def get_live_data(self, instrument_token, interval="5minute", days=1):
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

def macd(close, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# ============================================================================
# DATA MANAGER
# ============================================================================
class DataManager:
    def __init__(self, kite_manager=None):
        self.price_cache = {}
        self.kite_manager = kite_manager
    
    def get_stock_data(self, symbol, interval="15m", use_kite=False):
        """Get stock data from yfinance or Kite Connect"""
        
        # Try Kite Connect first if available
        if use_kite and self.kite_manager and self.kite_manager.is_authenticated:
            try:
                # Get instrument token
                instruments = self.kite_manager.get_instruments("NSE")
                trading_symbol = symbol.replace(".NS", "")
                
                instrument = next((i for i in instruments if i["tradingsymbol"] == trading_symbol), None)
                
                if instrument:
                    kite_data = self.kite_manager.get_live_data(
                        instrument["instrument_token"],
                        interval="5minute" if interval == "5m" else "15minute",
                        days=7
                    )
                    
                    if kite_data is not None and len(kite_data) > 0:
                        # Add technical indicators
                        df = self._add_indicators(kite_data)
                        return df
            except Exception as e:
                logger.error(f"Kite data fetch failed for {symbol}: {e}")
        
        # Fallback to yfinance
        try:
            period = "7d" if interval == "15m" else "2d"
            df = yf.download(symbol, period=period, interval=interval, progress=False)
            
            if df is not None and len(df) > 0:
                df = df.rename(columns=str.capitalize)
                df = self._add_indicators(df)
                return df
        except Exception as e:
            logger.error(f"YFinance failed for {symbol}: {e}")
        
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
            
            # VWAP
            df["VWAP"] = (((df["High"] + df["Low"] + df["Close"]) / 3) * df["Volume"]).cumsum() / df["Volume"].cumsum()
            
            return df
        except Exception as e:
            logger.error(f"Error adding indicators: {e}")
            return df

# ============================================================================
# TRADING ENGINE (SIMPLIFIED)
# ============================================================================
class SimpleTradingEngine:
    def __init__(self, capital=CAPITAL):
        self.initial_capital = float(capital)
        self.cash = float(capital)
        self.positions = {}
        self.trade_log = []
    
    def generate_signals(self, data_manager, symbols, use_kite=False):
        """Generate trading signals"""
        signals = []
        
        for symbol in symbols[:10]:  # Limit to 10 for performance
            try:
                data = data_manager.get_stock_data(symbol, "15m", use_kite=use_kite)
                
                if data is None or len(data) < 30:
                    continue
                
                current_price = float(data["Close"].iloc[-1])
                ema8 = float(data["EMA8"].iloc[-1])
                ema21 = float(data["EMA21"].iloc[-1])
                rsi_val = float(data["RSI14"].iloc[-1])
                vwap = float(data["VWAP"].iloc[-1])
                
                # Simple EMA + VWAP + RSI strategy
                if ema8 > ema21 and current_price > vwap and 40 < rsi_val < 65:
                    signals.append({
                        "symbol": symbol,
                        "action": "BUY",
                        "price": current_price,
                        "confidence": 0.75,
                        "rsi": rsi_val,
                        "data_source": "Kite" if use_kite else "YFinance"
                    })
                
                elif ema8 < ema21 and current_price < vwap and 35 < rsi_val < 60:
                    signals.append({
                        "symbol": symbol,
                        "action": "SELL",
                        "price": current_price,
                        "confidence": 0.72,
                        "rsi": rsi_val,
                        "data_source": "Kite" if use_kite else "YFinance"
                    })
            
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
                continue
        
        return signals

# ============================================================================
# INITIALIZE KITE MANAGER
# ============================================================================
if "kite_manager" not in st.session_state:
    st.session_state.kite_manager = KiteConnectManager(KITE_API_KEY, KITE_API_SECRET)

kite_manager = st.session_state.kite_manager

# ============================================================================
# HANDLE OAUTH CALLBACK (RUNS ONCE)
# ============================================================================
if not st.session_state.oauth_processed:
    if kite_manager.handle_oauth_callback():
        st.success("✅ Kite Connect authenticated successfully!")
        st.rerun()

# Check if already logged in
kite_manager.is_logged_in()

# ============================================================================
# INITIALIZE DATA MANAGER & TRADER
# ============================================================================
if "data_manager" not in st.session_state:
    st.session_state.data_manager = DataManager(kite_manager)

if "trader" not in st.session_state:
    st.session_state.trader = SimpleTradingEngine()

data_manager = st.session_state.data_manager
trader = st.session_state.trader

# ============================================================================
# ENHANCED CSS
# ============================================================================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #fff9e6 0%, #fff0d6 100%);
    }
    
    .kite-status-connected {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #059669;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .kite-status-disconnected {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #dc2626;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .signal-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1e3a8a;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MAIN UI
# ============================================================================
st.markdown("<h1 style='text-align:center; color: #1e3a8a;'>Rantv Intraday Terminal Pro</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color: #6b7280;'>Kite Connect Integration - Redirect Loop Fixed</h4>", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - KITE CONNECT STATUS
# ============================================================================
st.sidebar.header("🔐 Kite Connect Status")

if kite_manager.is_authenticated:
    st.sidebar.markdown("""
    <div class="kite-status-connected">
        <strong>✅ Connected to Kite</strong>
    </div>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("🚪 Logout from Kite"):
        kite_manager.logout()
        st.rerun()
else:
    st.sidebar.markdown("""
    <div class="kite-status-disconnected">
        <strong>❌ Not Connected</strong>
    </div>
    """, unsafe_allow_html=True)
    
    if KITE_API_KEY and KITE_API_SECRET:
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
    else:
        st.sidebar.error("⚠️ Kite API credentials not configured")

# ============================================================================
# MARKET OVERVIEW
# ============================================================================
cols = st.columns(5)
cols[0].metric("Market Status", "🟢 OPEN" if market_open() else "🔴 CLOSED")
cols[1].metric("IST Time", now_indian().strftime("%H:%M:%S"))
cols[2].metric("Kite Status", "✅ Connected" if kite_manager.is_authenticated else "❌ Disconnected")
cols[3].metric("Cash", f"₹{trader.cash:,.0f}")
cols[4].metric("Positions", len(trader.positions))

st.markdown("---")

# ============================================================================
# MAIN TABS
# ============================================================================
tabs = st.tabs([
    "📊 Dashboard",
    "🚦 Trading Signals",
    "📈 Kite Live Charts",
    "💼 Positions",
    "⚙️ Settings"
])

# ============================================================================
# TAB 1: DASHBOARD
# ============================================================================
with tabs[0]:
    st.subheader("📊 Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Capital", f"₹{trader.initial_capital:,.0f}")
    col2.metric("Current Cash", f"₹{trader.cash:,.0f}")
    col3.metric("Total Trades", len(trader.trade_log))
    col4.metric("Open Positions", len(trader.positions))
    
    if kite_manager.is_authenticated:
        st.success("✅ Kite Connect is active - live data available")
    else:
        st.info("ℹ️ Connect to Kite for live market data")

# ============================================================================
# TAB 2: TRADING SIGNALS
# ============================================================================
with tabs[1]:
    st.subheader("🚦 Trading Signals")
    
    use_kite_data = st.checkbox(
        "Use Kite Connect Data",
        value=kite_manager.is_authenticated,
        disabled=not kite_manager.is_authenticated
    )
    
    if st.button("🔄 Generate Signals", type="primary"):
        with st.spinner("Generating signals..."):
            signals = trader.generate_signals(data_manager, NIFTY_50, use_kite=use_kite_data)
            
            if signals:
                st.success(f"✅ Found {len(signals)} signals")
                
                for signal in signals:
                    action_color = "🟢" if signal["action"] == "BUY" else "🔴"
                    
                    st.markdown(f"""
                    <div class="signal-card">
                        <strong>{action_color} {signal['symbol'].replace('.NS', '')}</strong> - {signal['action']} @ ₹{signal['price']:.2f}<br>
                        Confidence: {signal['confidence']:.1%} | RSI: {signal['rsi']:.1f} | Source: {signal['data_source']}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No signals found")

# ============================================================================
# TAB 3: KITE LIVE CHARTS
# ============================================================================
with tabs[2]:
    st.subheader("📈 Kite Connect Live Charts")
    
    if not kite_manager.is_authenticated:
        st.warning("⚠️ Please login to Kite Connect to view live charts")
    else:
        symbol_input = st.selectbox(
            "Select Symbol",
            ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
        )
        
        interval = st.selectbox("Interval", ["5minute", "15minute", "30minute"])
        
        if st.button("📊 Load Chart"):
            with st.spinner(f"Loading {symbol_input} chart..."):
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
                            fig = go.Figure(data=[go.Candlestick(
                                x=data.index,
                                open=data['open'],
                                high=data['high'],
                                low=data['low'],
                                close=data['close'],
                                name=symbol_input
                            )])
                            
                            fig.update_layout(
                                title=f"{symbol_input} - Kite Connect Live Data",
                                xaxis_title="Time",
                                yaxis_title="Price (₹)",
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show current stats
                            current = data['close'].iloc[-1]
                            prev = data['close'].iloc[-2] if len(data) > 1 else current
                            change = ((current - prev) / prev) * 100
                            
                            st.metric(
                                f"Current Price",
                                f"₹{current:.2f}",
                                f"{change:+.2f}%"
                            )
                        else:
                            st.error("No data available")
                    else:
                        st.error("Symbol not found")
                        
                except Exception as e:
                    st.error(f"Chart error: {str(e)}")

# ============================================================================
# TAB 4: POSITIONS
# ============================================================================
with tabs[3]:
    st.subheader("💼 Open Positions")
    
    if trader.positions:
        st.dataframe(pd.DataFrame(trader.positions.values()))
    else:
        st.info("No open positions")

# ============================================================================
# TAB 5: SETTINGS
# ============================================================================
with tabs[4]:
    st.subheader("⚙️ Settings")
    
    st.write("**Kite Connect Configuration**")
    st.write(f"API Key: {'✅ Set' if KITE_API_KEY else '❌ Not Set'}")
    st.write(f"API Secret: {'✅ Set' if KITE_API_SECRET else '❌ Not Set'}")
    
    st.markdown("---")
    st.write("**System Status**")
    st.write(f"✅ Redirect loop: Fixed")
    st.write(f"✅ OAuth handling: Session-based")
    st.write(f"✅ Kite authentication: {'Active' if kite_manager.is_authenticated else 'Inactive'}")

st.markdown("---")
st.markdown("<div style='text-align:center; color: #6b7280;'>Rantv Intraday Terminal Pro | Kite Connect Integration</div>", unsafe_allow_html=True)
