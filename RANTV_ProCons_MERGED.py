#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RANTV Intraday Terminal Pro — FULL MERGE (Pros + Cons)
=====================================================

This file merges **both** of your originals, preserving the professional UI,
**full stock universes**, **Live Charts (Kite WebSocket)**, and your strategies,
while adding the advanced SMC + risk scaling from your second version.

Run:
    streamlit run RANTV_Pro_MERGED_FULL.py
"""

# ===================== IMPORTS & OPTIONALS =====================
import os, sys, time, json, logging, threading, warnings, smtplib, webbrowser
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Tuple, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import subprocess

# Friendly optional install helper (used only when missing)
def install_package(pkg: str) -> bool:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return True
    except Exception:
        return False

# Optional imports
try:
    import pytz
    PYTZ_AVAILABLE = True
except Exception:
    PYTZ_AVAILABLE = False
    pytz = None

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False
    st = None

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except Exception:
    AUTOREFRESH_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    YFINANCE_AVAILABLE = False
    yf = None

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False
    go = None

import pandas as pd
import numpy as np

# Kite Connect
try:
    from kiteconnect import KiteConnect, KiteTicker
    KITECONNECT_AVAILABLE = True
except Exception:
    KITECONNECT_AVAILABLE = False

# WebSocket client (for fallback live feed)
try:
    import websocket  # websocket-client
    WEBSOCKET_AVAILABLE = True
except Exception:
    WEBSOCKET_AVAILABLE = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RANTV_FULL_MERGE")

# ===================== CONFIGURATION =====================
class AppConfig:
    KITE_API_KEY: str = os.environ.get("KITE_API_KEY", "")
    KITE_API_SECRET: str = os.environ.get("KITE_API_SECRET", "")

    INITIAL_CAPITAL: float = float(os.environ.get("INITIAL_CAPITAL", "2000000"))
    BASE_RISK: float = 0.01
    MAX_RISK: float = 0.02
    MIN_RISK: float = 0.005
    MAX_DAILY_DD: float = 0.03
    SL_ATR: float = 1.5
    TP_ATR: float = 3.0
    TRAIL_ATR: float = 1.2

    ALGO_MAX_POSITIONS: int = int(os.environ.get("ALGO_MAX_POSITIONS", "5"))
    ALGO_MAX_DAILY_LOSS: float = float(os.environ.get("ALGO_MAX_DAILY_LOSS", "50000"))
    ALGO_MIN_CONFIDENCE: float = float(os.environ.get("ALGO_MIN_CONFIDENCE", "0.80"))

    MAX_DAILY_TRADES: int = int(os.environ.get("MAX_DAILY_TRADES", "10"))
    MAX_STOCK_TRADES: int = int(os.environ.get("MAX_STOCK_TRADES", "10"))

    EMAIL_SENDER: str = os.environ.get("EMAIL_SENDER", "")
    EMAIL_PASSWORD: str = os.environ.get("EMAIL_PASSWORD", "")
    EMAIL_RECEIVER: str = os.environ.get("EMAIL_RECEIVER", "rantv2002@gmail.com")

    if PYTZ_AVAILABLE:
        try:
            IND_TZ = pytz.timezone("Asia/Kolkata")
        except Exception:
            IND_TZ = None
    else:
        IND_TZ = None

    INDIA_OPEN = dt_time(9, 15)
    INDIA_CLOSE = dt_time(15, 30)
    DAILY_EXIT = dt_time(15, 35)

config = AppConfig()

# ===================== TIME HELPERS =====================
def now_indian() -> datetime:
    return datetime.now(config.IND_TZ) if config.IND_TZ else datetime.now()

def market_open() -> bool:
    t = now_indian().time()
    return config.INDIA_OPEN <= t <= config.INDIA_CLOSE

def is_peak_hours() -> bool:
    t = now_indian().time()
    return dt_time(9, 30) <= t <= dt_time(14, 30)

def should_exit_all_positions() -> bool:
    return now_indian().time() >= config.DAILY_EXIT

# ===================== STOCK UNIVERSES (FULL) =====================
# Preserving your full lists from the first file
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
ALL_STOCKS = list(dict.fromkeys(NIFTY_50 + NIFTY_100 + NIFTY_MIDCAP_150))

# ===================== INDICATORS =====================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    d = series.diff(); gain = d.clip(lower=0).rolling(period).mean(); loss = (-d.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan); rs = rs.fillna(0); return 100 - (100 / (1 + rs))

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low; tr2 = (high - close.shift()).abs(); tr3 = (low - close.shift()).abs(); tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    ef = ema(close, fast); es = ema(close, slow); ml = ef - es; sl = ema(ml, signal); return ml, sl, ml - sl

def bollinger_bands(close: pd.Series, period=20, std_dev=2):
    sma = close.rolling(period).mean(); std = close.rolling(period).std(); return sma + std*std_dev, sma, sma - std*std_dev

def calc_support_resistance(high: pd.Series, low: pd.Series, close: pd.Series, period=20) -> Dict[str, Any]:
    try:
        res, sup = [], []
        ln = len(high)
        if ln < period*2+1:
            return {"support": float(close.iloc[-1]*0.98), "resistance": float(close.iloc[-1]*1.02)}
        for i in range(period, ln - period):
            if high.iloc[i] >= high.iloc[i-period:i+period+1].max(): res.append(float(high.iloc[i]))
            if low.iloc[i]  <= low.iloc[i-period:i+period+1].min(): sup.append(float(low.iloc[i]))
        recent_res = sorted(res)[-3:] if res else [float(close.iloc[-1]*1.02)]
        recent_sup = sorted(sup)[:3] if sup else [float(close.iloc[-1]*0.98)]
        return {"support": float(np.mean(recent_sup)), "resistance": float(np.mean(recent_res)), "support_levels": recent_sup, "resistance_levels": recent_res}
    except Exception:
        cp = float(close.iloc[-1]); return {"support": cp*0.98, "resistance": cp*1.02, "support_levels": [], "resistance_levels": []}

# ===================== SMC MODULE =====================
class AdvancedSMC:
    @staticmethod
    def detect_BOS(df: pd.DataFrame, lookback: int = 6) -> Optional[str]:
        try:
            if len(df) < lookback+1: return None
            if df['High'].iloc[-1] > df['High'].iloc[-lookback:-1].max(): return 'BULLISH'
            if df['Low'].iloc[-1]  < df['Low'].iloc[-lookback:-1].min(): return 'BEARISH'
            return None
        except Exception: return None
    @staticmethod
    def detect_FVG(df: pd.DataFrame) -> Optional[Tuple[str, float, float]]:
        try:
            if len(df) < 3: return None
            a,b,c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
            if a['High'] < c['Low']:  return ('BULLISH', float(a['High']), float(c['Low']))
            if a['Low']  > c['High']: return ('BEARISH', float(c['High']), float(a['Low']))
            return None
        except Exception: return None
    @staticmethod
    def detect_order_block(df: pd.DataFrame) -> Optional[str]:
        try:
            if len(df) < 2: return None
            last = df.iloc[-2]
            if last['Close'] > last['Open']: return 'BULLISH'
            if last['Close'] < last['Open']: return 'BEARISH'
            return None
        except Exception: return None
    @staticmethod
    def detect_liquidity_grab(df: pd.DataFrame) -> Optional[str]:
        try:
            if len(df) < 10: return None
            rh = df['High'].iloc[-10:-1].max(); rl = df['Low'].iloc[-10:-1].min()
            ch = df['High'].iloc[-1]; cl = df['Low'].iloc[-1]; cc = df['Close'].iloc[-1]
            if cl < rl and cc > rl: return 'BULLISH'
            if ch > rh and cc < rh: return 'BEARISH'
            return None
        except Exception: return None

def volume_profile_poc(df: pd.DataFrame, bins: int = 24) -> float:
    try:
        if df is None or len(df) < bins: return float(df['Close'].iloc[-1]) if df is not None and len(df) else 0.0
        prices = df['Close'].values; vols = df['Volume'].values
        hist, edges = np.histogram(prices, bins=bins, weights=vols)
        i = int(np.argmax(hist)); return float((edges[i] + edges[i+1]) / 2) if i < len(edges)-1 else float(df['Close'].iloc[-1])
    except Exception: return float(df['Close'].iloc[-1]) if df is not None and len(df) else 0.0

# ===================== KITE MANAGER (Auth + Historical) =====================
class KiteManager:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key; self.api_secret = api_secret
        self.kite = None; self.is_authenticated = False
        if KITECONNECT_AVAILABLE and api_key and api_secret:
            try:
                self.kite = KiteConnect(api_key=api_key)
            except Exception as e:
                logger.warning(f"Kite init failed: {e}")

    def login_url(self) -> Optional[str]:
        try:
            return self.kite.login_url() if self.kite else None
        except Exception: return None

    def authenticate(self, request_token: str) -> Tuple[bool, str]:
        try:
            if not self.kite: return False, "Kite not available"
            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            access = data.get('access_token');
            if not access: return False, "No access token"
            self.kite.set_access_token(access); self.is_authenticated = True
            return True, f"Authenticated as {data.get('user_name', 'User')}"
        except Exception as e:
            return False, f"Auth error: {e}"

    def historical(self, symbol: str, interval: str = "15m", days: int = 7) -> Optional[pd.DataFrame]:
        try:
            if not (self.kite and self.is_authenticated): return None
            token_map = {
                "RELIANCE.NS":738561, "TCS.NS":2953217, "HDFCBANK.NS":341249, "INFY.NS":408065,
                "ICICIBANK.NS":1270529, "ITC.NS":424961, "LT.NS":2939649, "SBIN.NS":779521,
                "AXISBANK.NS":1510401, "BAJFINANCE.NS":81153,
            }
            token = token_map.get(symbol)
            if not token: return None
            imap = {"1m":"minute","5m":"5minute","15m":"15minute","30m":"30minute","1h":"60minute","1d":"day"}
            iv = imap.get(interval, "15minute")
            f, t = datetime.now().date()-timedelta(days=days), datetime.now().date()
            data = self.kite.historical_data(token, f, t, iv, continuous=False, oi=False)
            if not data: return None
            df = pd.DataFrame(data)
            if 'open' in df.columns:
                df = df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
            df['date'] = pd.to_datetime(df['date']); df.set_index('date', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Kite historical error: {e}"); return None

# ===================== LIVE TICKER (WebSocket) — from your first file =====================
class KiteLiveTicker:
    def __init__(self, kite_mgr: KiteManager):
        self.kite_mgr = kite_mgr; self.ws = None; self.tick_data = {}; self.subscribed_tokens = set()
        self.is_connected = False; self.last_update = {}; self.candle_data = {}; self.thread = None
    def connect(self) -> bool:
        if not (self.kite_mgr and self.kite_mgr.is_authenticated): return False
        try:
            access = self.kite_mgr.kite.access_token; api_key = self.kite_mgr.api_key
            url = f"wss://ws.kite.trade?api_key={api_key}&access_token={access}"
            if not WEBSOCKET_AVAILABLE:
                logger.error("websocket-client not available"); return False
            self.ws = websocket.WebSocketApp(url, on_open=self._on_open, on_message=self._on_message,
                                             on_error=self._on_error, on_close=self._on_close)
            self.thread = threading.Thread(target=self.ws.run_forever, daemon=True); self.thread.start()
            self.is_connected = True; logger.info("Kite WebSocket connected"); return True
        except Exception as e:
            logger.error(f"WS connect error: {e}"); return False
    def disconnect(self):
        if self.ws: self.ws.close(); self.is_connected=False; self.subscribed_tokens.clear(); logger.info("WS disconnected")
    def _on_open(self, ws):
        logger.info("WS opened"); self.subscribe([256265, 260105])  # NIFTY, BANKNIFTY
    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            if isinstance(data, dict) and data.get('type')=='ticks': self._process_ticks(data)
        except Exception as e: logger.error(f"WS msg error: {e}")
    def _on_error(self, ws, e): logger.error(f"WS error: {e}")
    def _on_close(self, ws, code, msg): logger.info(f"WS closed {code} {msg}"); self.is_connected=False; self.subscribed_tokens.clear()
    def _process_ticks(self, data):
        ticks = data.get('ticks', [])
        for tick in ticks:
            token = tick.get('instrument_token');
            if not token: continue
            self.tick_data[token] = tick; self.last_update[token] = time.time(); self._update_candle(token, tick)
    def _update_candle(self, token, tick):
        if token not in self.candle_data:
            self.candle_data[token] = {'open':[],'high':[],'low':[],'close':[],'volume':[],'timestamp':[]}
        now_ts = time.time(); interval = 60
        if not self.candle_data[token]['timestamp']:
            start = now_ts - (now_ts % interval)
            self.candle_data[token]['timestamp'].append(start)
            for k in ['open','high','low','close','volume']: self.candle_data[token][k].append(tick.get('last_price',0)) if k!='volume' else self.candle_data[token][k].append(tick.get('volume_traded',0))
        else:
            start = self.candle_data[token]['timestamp'][-1]
            if now_ts - start >= interval:
                new_start = now_ts - (now_ts % interval)
                self.candle_data[token]['timestamp'].append(new_start)
                for k in ['open','high','low','close','volume']: self.candle_data[token][k].append(tick.get('last_price',0)) if k!='volume' else self.candle_data[token][k].append(tick.get('volume_traded',0))
            else:
                i = -1; p = tick.get('last_price',0)
                if self.candle_data[token]['high'][i] < p: self.candle_data[token]['high'][i] = p
                if self.candle_data[token]['low'][i]  > p: self.candle_data[token]['low'][i] = p
                self.candle_data[token]['close'][i]   = p
                self.candle_data[token]['volume'][i]  = tick.get('volume_traded',0)
    def subscribe(self, tokens: List[int]) -> bool:
        if not (self.is_connected and self.ws): return False
        try:
            for t in tokens: self.subscribed_tokens.add(t)
            self.ws.send(json.dumps({"a":"subscribe","v":tokens})); logger.info(f"Subscribed {tokens}"); return True
        except Exception as e: logger.error(f"Sub error {e}"); return False
    def get_candles(self, token: int, n: int = 100) -> Optional[Dict[str, List[float]]]:
        if token not in self.candle_data: return None
        d = self.candle_data[token];
        m = min(n, len(d['timestamp']))
        return {k: d[k][-m:] for k in d}

# ===================== DATA MANAGER (Unified + Enrichment) =====================
class UnifiedDataManager:
    def __init__(self, kite_mgr: Optional[KiteManager] = None):
        self.kite_mgr = kite_mgr; self.cache: Dict[str, Tuple[pd.DataFrame, float]] = {}; self.ttl = 30
        self.smc = AdvancedSMC()
    def _cache_get(self, key: str):
        if key in self.cache:
            df, ts = self.cache[key]
            if time.time()-ts < self.ttl: return df
        return None
    def _cache_put(self, key: str, df: pd.DataFrame): self.cache[key] = (df.copy(), time.time())
    def get_stock(self, symbol: str, interval: str = "15m", use_kite=True) -> Optional[pd.DataFrame]:
        key = f"{symbol}_{interval}"; c = self._cache_get(key)
        if c is not None: return c
        # kite first
        if use_kite and self.kite_mgr and self.kite_mgr.is_authenticated:
            df = self.kite_mgr.historical(symbol, interval, days=7)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = self._enrich(df); self._cache_put(key, df); return df
        # yahoo fallback
        try:
            if not YFINANCE_AVAILABLE: return None
            pm = {"1m":"1d","5m":"5d","15m":"15d","30m":"30d","1h":"60d","1d":"1y"}
            period = pm.get(interval, "15d")
            df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
            if df is None or df.empty or len(df) < 20: return None
            df.columns = [c.capitalize() for c in df.columns]
            df = df[["Open","High","Low","Close","Volume"]].dropna()
            df = self._enrich(df); self._cache_put(key, df); return df
        except Exception as e:
            logger.error(f"Yahoo error {symbol}: {e}"); return None
    def _enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty: return df
        df["EMA8"]=ema(df["Close"],8); df["EMA21"]=ema(df["Close"],21); df["EMA50"]=ema(df["Close"],50)
        df["RSI14"]=rsi(df["Close"],14); df["ATR"]=calculate_atr(df["High"],df["Low"],df["Close"])
        m,s,h = macd(df["Close"]); df["MACD"],df["MACD_Signal"],df["MACD_Hist"] = m,s,h
        u,mid,l = bollinger_bands(df["Close"]); df["BB_Upper"],df["BB_Middle"],df["BB_Lower"] = u,mid,l
        tp=(df['High']+df['Low']+df['Close'])/3; df['VWAP']=(tp*df['Volume']).cumsum()/df['Volume'].cumsum()
        sr = calc_support_resistance(df['High'], df['Low'], df['Close']); df['Support']=sr['support']; df['Resistance']=sr['resistance']
        # SMC tags
        try:
            df['SMC_BOS']=self.smc.detect_BOS(df) or 'NONE'; df['SMC_FVG']='PRESENT' if self.smc.detect_FVG(df) else 'NONE'
            df['SMC_OB']=self.smc.detect_order_block(df) or 'NONE'; df['SMC_Liquidity']=self.smc.detect_liquidity_grab(df) or 'NONE'
            df['Volume_POC']=volume_profile_poc(df)
        except Exception: pass
        return df
    def live_price(self, symbol: str) -> Optional[float]:
        df = self.get_stock(symbol, interval="5m", use_kite=False)
        return float(df['Close'].iloc[-1]) if df is not None and len(df)>0 else None

# ===================== STRATEGY & SIGNALS =====================
TRADING_STRATEGIES = {
    "EMA_VWAP_Confluence":{"name":"EMA + VWAP Confluence","weight":3},
    "RSI_MeanReversion":   {"name":"RSI Mean Reversion","weight":2},
    "Bollinger_Reversion": {"name":"Bollinger Band Reversion","weight":2},
    "MACD_Momentum":       {"name":"MACD Momentum","weight":2},
    "Support_Resistance":  {"name":"Support/Resistance","weight":3},
    "MTF_SMC":             {"name":"MTF SMC + Volume Profile","weight":5}
}

class AdvancedSignalGenerator:
    def __init__(self, dm: UnifiedDataManager):
        self.dm=dm; self.smc=AdvancedSMC(); self.last_scan=None
    def smc_signal(self, symbol: str) -> Optional[Dict[str,Any]]:
        mtf = {iv:self.dm.get_stock(symbol,iv) for iv in ["15m","1h","1d"]}
        if not mtf.get("15m") or not mtf.get("1h"): return None
        ltf = mtf["15m"]; htf = mtf["1h"]; 
        if len(ltf)<50 or len(htf)<20 or not market_open(): return None
        price = float(ltf['Close'].iloc[-1])
        bos = self.smc.detect_BOS(htf); fvg = self.smc.detect_FVG(ltf); ob = self.smc.detect_order_block(ltf); liq = self.smc.detect_liquidity_grab(ltf)
        poc = volume_profile_poc(ltf); atr = float(ltf['ATR'].iloc[-1]) if 'ATR' in ltf.columns else max(0.02*price,1)
        sig=None; conf=0.0
        if bos=='BULLISH' and (fvg and fvg[0]=='BULLISH') and ob=='BULLISH' and price>poc:
            conf = 0.95 if liq=='BULLISH' else 0.85; sig={'action':'BUY','stop_loss':price-config.SL_ATR*atr,'target':price+config.TP_ATR*atr}
        elif bos=='BEARISH' and (fvg and fvg[0]=='BEARISH') and ob=='BEARISH' and price<poc:
            conf = 0.95 if liq=='BEARISH' else 0.85; sig={'action':'SELL','stop_loss':price+config.SL_ATR*atr,'target':price-config.TP_ATR*atr}
        if not sig: return None
        win_prob = min(0.98, conf*0.95)
        return {'symbol':symbol,'price':round(price,2),'confidence':round(conf,3),'win_probability':round(win_prob,3),'strategy':'MTF SMC + Volume Profile','atr':round(atr,2),'poc':round(poc,2),**sig}
    def tech_confluence(self, symbol: str) -> Optional[Dict[str,Any]]:
        df = self.dm.get_stock(symbol,"15m"); 
        if df is None or len(df)<50: return None
        price = float(df['Close'].iloc[-1]); out=[]
        # EMA+VWAP
        if price>df['EMA8'].iloc[-1]>df['EMA21'].iloc[-1] and price>df['VWAP'].iloc[-1]: out.append({'action':'BUY','confidence':0.8})
        elif price<df['EMA8'].iloc[-1]<df['EMA21'].iloc[-1] and price<df['VWAP'].iloc[-1]: out.append({'action':'SELL','confidence':0.8})
        # RSI
        r=float(df['RSI14'].iloc[-1]); 
        if r<30: out.append({'action':'BUY','confidence':0.75});
        elif r>70: out.append({'action':'SELL','confidence':0.75})
        # BB
        if price<=df['BB_Lower'].iloc[-1]*1.01: out.append({'action':'BUY','confidence':0.7})
        elif price>=df['BB_Upper'].iloc[-1]*0.99: out.append({'action':'SELL','confidence':0.7})
        # MACD cross
        m=df['MACD'].iloc[-1]; s=df['MACD_Signal'].iloc[-1]; pm=df['MACD'].iloc[-2]; ps=df['MACD_Signal'].iloc[-2]
        if m>s and pm<=ps: out.append({'action':'BUY','confidence':0.75})
        elif m<s and pm>=ps: out.append({'action':'SELL','confidence':0.75})
        # S/R
        sup=df['Support'].iloc[-1]; res=df['Resistance'].iloc[-1]
        if price<=sup*1.01: out.append({'action':'BUY','confidence':0.8})
        elif price>=res*0.99: out.append({'action':'SELL','confidence':0.8})
        if not out: return None
        buy=[d for d in out if d['action']=='BUY']; sell=[d for d in out if d['action']=='SELL']
        sb=sum(d['confidence'] for d in buy); ss=sum(d['confidence'] for d in sell)
        action='BUY' if sb>ss else 'SELL'; conf=(sb if action=='BUY' else ss)/max(1,len(buy if action=='BUY' else sell))
        atr=float(df['ATR'].iloc[-1]) if 'ATR' in df.columns else max(0.02*price,1)
        sl = price - 1.5*atr if action=='BUY' else price + 1.5*atr
        tg = price + 3.0*atr if action=='BUY' else price - 3.0*atr
        return {'symbol':symbol,'price':round(price,2),'action':action,'stop_loss':round(sl,2),'target':round(tg,2),'confidence':round(conf,3),'win_probability':round(min(0.97,conf*0.85),3),'strategy':'Multi-Strategy Confluence'}
    def generate(self, symbol: str) -> Optional[Dict[str,Any]]:
        s = self.smc_signal(symbol)
        if s and s['confidence']>=config.ALGO_MIN_CONFIDENCE: return s
        return self.tech_confluence(symbol)
    def scan(self, universe: List[str], max_signals: int = 15) -> List[Dict[str,Any]]:
        sigs=[]
        for s in universe:
            g=self.generate(s)
            if g and g['confidence']>=config.ALGO_MIN_CONFIDENCE: sigs.append(g)
        sigs.sort(key=lambda d:(d.get('confidence',0),d.get('win_probability',0)), reverse=True)
        self.last_scan = now_indian(); return sigs[:max_signals]

# ===================== RISK & TRADER =====================
class AutoRiskScaler:
    def __init__(self, initial_capital: float):
        self.equity=[initial_capital]; self.returns=[]; self.max_dd=0.0; self.cur_dd=0.0
    def update(self, pnl: float):
        new = self.equity[-1]+pnl; self.equity.append(new)
        if len(self.equity)>1:
            r=(new-self.equity[-2])/max(1e-9,self.equity[-2]); self.returns.append(r)
        peak=max(self.equity); self.cur_dd=(peak-new)/peak if peak>0 else 0.0; self.max_dd=max(self.max_dd,self.cur_dd)
    def risk(self) -> float:
        if len(self.returns)<10: return config.BASE_RISK
        vol=np.std(self.returns[-10:]) or 0.02; avg=np.mean(self.returns[-10:]) if self.returns else 0.0
        sharpe_like=avg/vol if vol>0 else 0.0; r=config.BASE_RISK
        if self.cur_dd < config.MAX_DAILY_DD*0.5 and sharpe_like>0: r*=min(2.0,1.0+8.0*sharpe_like)
        elif self.cur_dd>config.MAX_DAILY_DD*0.8 or sharpe_like<0: r*=max(0.3,1.0-5.0*abs(sharpe_like))
        return float(max(config.MIN_RISK, min(config.MAX_RISK, r)))

class RiskManager:
    def __init__(self):
        self.scaler=AutoRiskScaler(config.INITIAL_CAPITAL); self.daily_pnl=0.0; self.pos_open=0; self.trades_today=0
    def status(self) -> Dict[str,Any]:
        return {'daily_pnl':self.daily_pnl,'positions_open':self.pos_open,'trades_today':self.trades_today,'dynamic_risk':self.scaler.risk(),'within_limits': self.daily_pnl> -config.ALGO_MAX_DAILY_LOSS and self.pos_open<=config.ALGO_MAX_POSITIONS,'drawdown':self.scaler.cur_dd}
    def approve(self, confidence: float) -> Tuple[bool,str]:
        stt=self.status()
        if not market_open(): return False, "Market closed"
        if confidence<config.ALGO_MIN_CONFIDENCE: return False, "Low confidence"
        if not stt['within_limits']: return False, "Risk limits breached"
        if self.trades_today>=config.MAX_DAILY_TRADES: return False, "Daily trade limit reached"
        return True, "OK"
    def on_open(self): self.pos_open+=1; self.trades_today+=1
    def on_close(self, pnl: float): self.pos_open=max(0,self.pos_open-1); self.daily_pnl+=pnl; self.scaler.update(pnl)

class AdvancedPaperTrader:
    def __init__(self, dm: UnifiedDataManager, risk: RiskManager):
        self.dm=dm; self.risk=risk; self.cash=config.INITIAL_CAPITAL
        self.positions: Dict[str, Dict[str,Any]]={}; self.trades: List[Dict[str,Any]]=[]
    def size(self, price: float, is_smc: bool) -> int:
        pct=self.risk.status()['dynamic_risk']; pct*=1.2 if is_smc else 1.0
        val=self.cash*pct; qty=int(val/max(1e-6,price)); return max(1,min(qty,200))
    def open_from_signal(self, sig: Dict[str,Any]) -> Tuple[bool,str]:
        ok, reason = self.risk.approve(sig.get('confidence',0.0))
        if not ok: return False, reason
        sym=sig['symbol']; price=sig['price']; action=sig['action']; is_smc=('poc' in sig)
        qty=self.size(price,is_smc); val=qty*price
        if action=='BUY' and val>self.cash: return False, "Insufficient cash"
        self.positions[sym]={'symbol':sym,'action':action,'qty':qty,'entry':price,'sl':sig.get('stop_loss'),'tg':sig.get('target'),'opened':now_indian(),'strategy':sig.get('strategy','N/A'),'is_smc':is_smc}
        if action=='BUY': self.cash-=val
        else: self.cash-=val*0.2
        self.trades.append({**self.positions[sym], 'status':'OPEN'}); self.risk.on_open(); return True, f"Opened {action} {qty} {sym} @ ₹{price:.2f}"
    def mark(self, sym: str) -> Optional[float]: return self.dm.live_price(sym)
    def update_and_maybe_close(self):
        for sym,pos in list(self.positions.items()):
            p=self.mark(sym); 
            if p is None: continue
            pos['last']=p; pnl=(p-pos['entry'])*pos['qty'] if pos['action']=='BUY' else (pos['entry']-p)*pos['qty']; pos['pnl']=pnl
            sl=pos.get('sl'); tg=pos.get('tg')
            hit_sl=(pos['action']=='BUY' and sl is not None and p<=sl) or (pos['action']=='SELL' and sl is not None and p>=sl)
            hit_tg=(pos['action']=='BUY' and tg is not None and p>=tg) or (pos['action']=='SELL' and tg is not None and p<=tg)
            if hit_sl or hit_tg or should_exit_all_positions(): self.close(sym, exit_price=p)
    def close(self, sym: str, exit_price: Optional[float]=None) -> Tuple[bool,str]:
        if sym not in self.positions: return False, "Not found"
        pos=self.positions[sym]; p=exit_price or self.mark(sym) or pos['entry']
        if pos['action']=='BUY': pnl=(p-pos['entry'])*pos['qty']; self.cash+=pos['qty']*p
        else: pnl=(pos['entry']-p)*pos['qty']; self.cash+=pos['qty']*pos['entry']*0.2
        pos['status']='CLOSED'; pos['exit']=p; pos['closed']=now_indian(); self.trades.append({**pos}); del self.positions[sym]; self.risk.on_close(pnl)
        return True, f"Closed {sym} @ ₹{p:.2f} | P&L: ₹{pnl:+.2f}"
    def close_all(self):
        for sym in list(self.positions.keys()): self.close(sym)
    def perf(self) -> Dict[str,Any]:
        closed=[t for t in self.trades if t.get('status')=='CLOSED']
        wins=[t for t in closed if ((t['action']=='BUY' and t['exit']>t['entry']) or (t['action']=='SELL' and t['exit']<t['entry']))]
        total_pnl=sum(((t['exit']-t['entry'])*t['qty']) if t['action']=='BUY' else ((t['entry']-t['exit'])*t['qty']) for t in closed)
        return {'total_trades':len(closed),'win_rate':len(wins)/len(closed) if closed else 0.0,'total_pnl':total_pnl,'open_positions':len(self.positions),'cash':self.cash}

# ===================== ALGO ENGINE =====================
class AlgoEngine:
    def __init__(self, trader: AdvancedPaperTrader, sig: AdvancedSignalGenerator):
        self.trader=trader; self.sig=sig; self._stop=threading.Event(); self.state='stopped'; self.thread=None
        self.universe = NIFTY_50  # default; can expand to ALL_STOCKS from UI
    def start(self):
        if self.state=='running': return False
        self.state='running'; self._stop.clear(); self.thread=threading.Thread(target=self._loop, daemon=True); self.thread.start(); logger.info("Algo started"); return True
    def stop(self):
        self.state='stopped'; self._stop.set(); 
        if self.thread: self.thread.join(timeout=5); logger.info("Algo stopped")
    def _loop(self):
        last_scan=0.0
        while not self._stop.is_set():
            try:
                if self.state!='running' or not market_open(): time.sleep(2); continue
                if should_exit_all_positions():
                    self.trader.close_all(); 
                    try: send_daily_report_email(self.trader)
                    except Exception: pass
                    time.sleep(60); continue
                self.trader.update_and_maybe_close()
                if time.time()-last_scan>300:
                    sigs=self.sig.scan(self.universe, max_signals=10)
                    for s in sigs:
                        if s['symbol'] not in self.trader.positions and self.trader.risk.pos_open < config.ALGO_MAX_POSITIONS:
                            ok,msg=self.trader.open_from_signal(s); logger.info(msg)
                    last_scan=time.time()
                time.sleep(5)
            except Exception as e:
                logger.error(f"Algo error: {e}"); time.sleep(5)

# ===================== EMAIL DAILY REPORT =====================
def send_daily_report_email(trader: AdvancedPaperTrader) -> bool:
    try:
        if not config.EMAIL_SENDER or not config.EMAIL_PASSWORD:
            logger.warning("Email creds not set"); return False
        perf=trader.perf(); today=now_indian().date().strftime('%Y-%m-%d')
        subject=f"Daily Trading Report - {today}"
        rows=[]
        for sym,pos in trader.positions.items():
            rows.append(f"<tr><td>{sym}</td><td>{pos['action']}</td><td>{pos['qty']}</td><td>₹{pos['entry']:.2f}</td><td>OPEN</td></tr>")
        closed=[t for t in trader.trades if t.get('status')=='CLOSED']
        for t in closed[-30:]:
            pnl=((t['exit']-t['entry'])*t['qty']) if t['action']=='BUY' else ((t['entry']-t['exit'])*t['qty'])
            rows.append(f"<tr><td>{t['symbol']}</td><td>{t['action']}</td><td>{t['qty']}</td><td>₹{t['entry']:.2f}</td><td>₹{t['exit']:.2f} (₹{pnl:+.2f})</td></tr>")
        html=f"""
        <html><body style="font-family:Arial,sans-serif">
        <h2 style="color:#f97316">📈 Daily Trading Report</h2>
        <p><b>Total trades:</b> {perf['total_trades']} | <b>Win rate:</b> {perf['win_rate']:.1%} | <b>Total P&L:</b> ₹{perf['total_pnl']:+.2f}</p>
        <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;width:100%">
        <tr style="background:#faf0e6"><th>Symbol</th><th>Action</th><th>Qty</th><th>Entry</th><th>Exit/P&L</th></tr>
        {''.join(rows) if rows else '<tr><td colspan=5>No trades today</td></tr>'}
        </table>
        <p style="color:#64748b;font-size:12px">Generated at {now_indian().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </body></html>
        """
        msg=MIMEMultipart('alternative'); msg['Subject']=subject; msg['From']=config.EMAIL_SENDER; msg['To']=config.EMAIL_RECEIVER
        msg.attach(MIMEText(html,'html'))
        with smtplib.SMTP_SSL('smtp.gmail.com',465) as s:
            s.login(config.EMAIL_SENDER, config.EMAIL_PASSWORD); s.send_message(msg)
        logger.info(f"Daily report emailed to {config.EMAIL_RECEIVER}"); return True
    except Exception as e:
        logger.error(f"Email error: {e}"); return False

# ===================== STREAMLIT UI (Preserved Orange Theme + Live Charts) =====================
if STREAMLIT_AVAILABLE:
    st.set_page_config(page_title="RANTV Intraday Terminal Pro — FULL MERGE", layout="wide", initial_sidebar_state="expanded")

    # ---- CSS (preserved style) ----
    st.markdown("""
    <style>
    .stApp {background: linear-gradient(135deg,#fff5e6 0%, #ffe8cc 100%);} 
    .main .block-container {background-color: transparent; padding-top: 0.8rem;} 
    .main-header {text-align:center; margin:10px auto; padding:12px; background: linear-gradient(135deg,#ff8c00 0%, #ff6b00 100%); border-radius: 14px; color: white; box-shadow: 0 8px 25px rgba(255,140,0,0.30);} 
    .stTabs [data-baseweb="tab-list"] {gap:4px; background: linear-gradient(135deg,#ffe8cc 0%, #ffd9a6 50%, #ffca80 100%); padding:8px; border-radius:12px; margin-bottom:10px;} 
    .stTabs [data-baseweb="tab"] { height: 60px; white-space: pre-wrap; background:#ffffff; border-radius:8px; gap:8px; padding:12px 20px; font-weight:600; font-size:14px; color:#d97706; border:2px solid transparent; box-shadow:0 2px 4px rgba(0,0,0,0.1);} 
    .stTabs [aria-selected="true"] {background: linear-gradient(135deg,#ff8c00 0%, #ff6b00 100%); color:white; border:2px solid #ff8c00; box-shadow:0 4px 8px rgba(255,140,0,0.3);} 
    [data-testid="stMetricValue"]{color:#ff8c00;font-weight:800}
    .good{color:#059669}.bad{color:#dc2626}
    </style>
    """, unsafe_allow_html=True)

    # ---- Sidebar: Kite Auth ----
    st.sidebar.header("🔐 Kite Connect (Live)")
    if 'kite_mgr' not in st.session_state:
        st.session_state.kite_mgr = KiteManager(config.KITE_API_KEY, config.KITE_API_SECRET) if (config.KITE_API_KEY and config.KITE_API_SECRET) else None
    kite_mgr = st.session_state.kite_mgr

    if kite_mgr and not kite_mgr.is_authenticated:
        colA, colB = st.sidebar.columns(2)
        if colA.button("Get Login URL"):
            url = kite_mgr.login_url()
            if url: st.sidebar.success("Login URL generated"); st.sidebar.code(url)
            else: st.sidebar.error("Failed to get login URL")
        req = st.sidebar.text_input("Paste request_token here")
        if st.sidebar.button("Authenticate") and req:
            ok,msg = kite_mgr.authenticate(req)
            (st.sidebar.success if ok else st.sidebar.error)(msg)

    # ---- Compose system ----
    if 'dm' not in st.session_state: st.session_state.dm = UnifiedDataManager(kite_mgr)
    if 'risk' not in st.session_state: st.session_state.risk = RiskManager()
    if 'sig' not in st.session_state: st.session_state.sig = AdvancedSignalGenerator(st.session_state.dm)
    if 'trader' not in st.session_state: st.session_state.trader = AdvancedPaperTrader(st.session_state.dm, st.session_state.risk)
    if 'algo' not in st.session_state: st.session_state.algo = AlgoEngine(st.session_state.trader, st.session_state.sig)

    trader: AdvancedPaperTrader = st.session_state.trader
    algo: AlgoEngine = st.session_state.algo

    # ---- Header ----
    st.markdown('<div class="main-header"><h2>RANTV Intraday Terminal Pro — FULL MERGE</h2><p>Live Charts + Full Universes + SMC + Risk Scaling</p></div>', unsafe_allow_html=True)

    # ---- Metrics ----
    perf = trader.perf(); risk = st.session_state.risk.status()
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Cash", f"₹{perf['cash']:,.0f}")
    m2.metric("Open Positions", perf['open_positions'])
    m3.metric("Closed Trades", perf['total_trades'])
    m4.metric("Win Rate", f"{perf['win_rate']:.1%}")
    m5.metric("Total P&L", f"₹{perf['total_pnl']:+,.0f}", delta_color="normal" if perf['total_pnl']>=0 else "inverse")

    tabs = st.tabs(["📊 Signals", "📈 Live Charts", "💼 Positions", "🤖 Algo", "🧠 Universes", "📬 Reports"]) 

    # ---- Signals ----
    with tabs[0]:
        st.subheader("Top Signals (select universe)")
        uni_name = st.selectbox("Universe", ["NIFTY 50","NIFTY 100","MIDCAP 150","ALL"], index=0)
        if uni_name=="NIFTY 50": universe=NIFTY_50
        elif uni_name=="NIFTY 100": universe=NIFTY_100
        elif uni_name=="MIDCAP 150": universe=NIFTY_MIDCAP_150
        else: universe=ALL_STOCKS
        max_sigs = st.slider("Max signals", 5, 30, 15)
        if st.button("Scan Now", type="primary"):
            sigs = st.session_state.sig.scan(universe, max_signals=max_sigs)
            st.session_state['latest_sigs']=sigs
        sigs=st.session_state.get('latest_sigs', [])
        if sigs:
            for s in sigs:
                c1,c2,c3,c4,c5,c6 = st.columns([2,1,1,1,2,1])
                c1.write(f"**{s['symbol']}**")
                c2.write(s['action'])
                c3.write(f"Conf: {s['confidence']:.2f}")
                c4.write(f"Win: {s['win_probability']:.2f}")
                c5.write(f"SL: ₹{s['stop_loss']:.2f} • TG: ₹{s['target']:.2f}")
                if c6.button("Trade", key=f"trade_{s['symbol']}_{time.time()}"):
                    ok,msg = trader.open_from_signal(s)
                    (st.success if ok else st.error)(msg)
        else:
            st.info("Click **Scan Now** to generate signals.")

    # ---- Live Charts ----
    with tabs[1]:
        st.subheader("Kite Connect Live Charts (WebSocket)")
        if not kite_mgr or not kite_mgr.is_authenticated:
            st.warning("Authenticate with Kite in the sidebar to enable live charts.")
        else:
            if 'ticker' not in st.session_state: st.session_state.ticker = KiteLiveTicker(kite_mgr)
            ticker: KiteLiveTicker = st.session_state.ticker
            colA,colB = st.columns(2)
            if not ticker.is_connected:
                if colA.button("🔗 Connect Live Feed", type="primary"): 
                    ok = ticker.connect(); (st.success if ok else st.error)("Connected" if ok else "Failed to connect")
            else:
                if colA.button("⛔ Disconnect", type="secondary"): ticker.disconnect(); st.info("Disconnected")
            status = "🟢 CONNECTED" if ticker.is_connected else "🔴 DISCONNECTED"; colB.metric("Live Feed", status)

            INDEX_MAP = {"NIFTY 50":256265, "BANK NIFTY":260105}
            STOCK_MAP = {"RELIANCE":738561, "TCS":2953217, "HDFCBANK":341249, "INFY":408065, "ICICIBANK":1270529}
            chart_tabs = st.tabs(["📈 Index", "📊 Stock"])
            with chart_tabs[0]:
                idx = st.selectbox("Index", list(INDEX_MAP.keys()))
                token = INDEX_MAP[idx]
                if ticker.is_connected:
                    ticker.subscribe([token])
                    data = ticker.get_candles(token, n=120)
                    if data and PLOTLY_AVAILABLE:
                        ts = [datetime.fromtimestamp(x) for x in data['timestamp']]
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7,0.3])
                        fig.add_trace(go.Candlestick(x=ts, open=data['open'], high=data['high'], low=data['low'], close=data['close'], name=idx, increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1, col=1)
                        fig.add_trace(go.Bar(x=ts, y=data['volume'], name='Volume', marker_color='#ff8c00', opacity=0.7), row=2, col=1)
                        fig.update_layout(title=f"{idx} Live Chart (1m)", xaxis_rangeslider_visible=False, height=600, template='plotly_white')
                        st.plotly_chart(fig, use_container_width=True)
            with chart_tabs[1]:
                stk = st.selectbox("Stock", list(STOCK_MAP.keys()))
                token = STOCK_MAP[stk]
                if ticker.is_connected:
                    ticker.subscribe([token])
                    data = ticker.get_candles(token, n=120)
                    if data and PLOTLY_AVAILABLE:
                        ts = [datetime.fromtimestamp(x) for x in data['timestamp']]
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7,0.3])
                        fig.add_trace(go.Candlestick(x=ts, open=data['open'], high=data['high'], low=data['low'], close=data['close'], name=stk, increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1, col=1)
                        fig.add_trace(go.Bar(x=ts, y=data['volume'], name='Volume', marker_color='#ff8c00', opacity=0.7), row=2, col=1)
                        fig.update_layout(title=f"{stk} Live Chart (1m)", xaxis_rangeslider_visible=False, height=600, template='plotly_white')
                        st.plotly_chart(fig, use_container_width=True)

    # ---- Positions ----
    with tabs[2]:
        st.subheader("Open Positions")
        if trader.positions:
            for sym,pos in trader.positions.items():
                c1,c2,c3,c4,c5,c6 = st.columns([2,1,1,1,1,1])
                c1.write(f"**{sym}**"); c2.write(pos['action']); c3.write(f"Qty: {pos['qty']}"); c4.write(f"Entry: ₹{pos['entry']:.2f}")
                c5.write(f"SL: {'N/A' if pos.get('sl') is None else '₹'+str(round(pos['sl'],2))}")
                if c6.button("Close", key=f"close_{sym}"):
                    ok,msg=trader.close(sym); (st.success if ok else st.error)(msg)
        else:
            st.info("No open positions")
        st.divider(); st.subheader("Closed Trades (last 30)")
        closed=[t for t in trader.trades if t.get('status')=='CLOSED'][-30:]
        if closed:
            df=pd.DataFrame([{k:v for k,v in t.items() if k in ('symbol','action','qty','entry','exit','strategy','is_smc')} for t in closed])
            st.dataframe(df, use_container_width=True)
        else:
            st.write("No closed trades yet.")

    # ---- Algo ----
    with tabs[3]:
        st.subheader("Algo Engine")
        col1,col2,col3 = st.columns(3)
        if algo.state!='running':
            if col1.button("Start Algo", type="primary"): algo.start(); st.success("Algo started")
        else:
            if col1.button("Stop Algo", type="secondary"): algo.stop(); st.warning("Algo stopped")
        uni_for_algo = st.selectbox("Algo Universe", ["NIFTY 50","NIFTY 100","MIDCAP 150","ALL"], index=0)
        algo.universe = NIFTY_50 if uni_for_algo=="NIFTY 50" else NIFTY_100 if uni_for_algo=="NIFTY 100" else NIFTY_MIDCAP_150 if uni_for_algo=="MIDCAP 150" else ALL_STOCKS
        if col2.button("Force Scan"):
            sigs=st.session_state.sig.scan(algo.universe, max_signals=10)
            for s in sigs:
                if s['symbol'] not in trader.positions and st.session_state.risk.pos_open < config.ALGO_MAX_POSITIONS:
                    ok,msg=trader.open_from_signal(s); (st.success if ok else st.error)(msg)
        if col3.button("Exit All"):
            trader.close_all(); st.warning("All positions closed")
        st.caption("Algo scans every ~5 minutes during market hours and auto-exits at 3:35 PM.")

    # ---- Universes ----
    with tabs[4]:
        st.subheader("Stock Universes (editable)")
        st.write(f"NIFTY 50 count: {len(NIFTY_50)} | NIFTY 100 count: {len(NIFTY_100)} | MIDCAP150 count: {len(NIFTY_MIDCAP_150)} | ALL: {len(ALL_STOCKS)}")
        st.expander("Show samples", expanded=False).write(pd.DataFrame({'NIFTY50':NIFTY_50[:20], 'NIFTY100':NIFTY_100[:20], 'MIDCAP150':NIFTY_MIDCAP_150[:20]}))

    # ---- Reports ----
    with tabs[5]:
        st.subheader("Daily Email Report")
        if st.button("Send Now"):
            ok=send_daily_report_email(trader); (st.success if ok else st.error)("Report sent" if ok else "Failed (check EMAIL_* env vars)")

else:
    if __name__=='__main__':
        print("Streamlit not available. Install: pip install streamlit plotly yfinance websocket-client kiteconnect")
