#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RANTV Intraday Terminal Pro — FULL MERGE with NSE Token Mapping
==============================================================

This version fixes the Kite Connect UI (enter API key/secret in the sidebar,
get login URL, authenticate) and after successful auth it builds a **full NSE
instrument token map** via `kite.instruments("NSE")`. Live Charts can then
subscribe to **any symbol** in your universes using the correct tokens.

Run:
    streamlit run RANTV_Pro_MERGED_FULL_with_tokens.py
"""

import os, sys, time, json, logging, threading, warnings, smtplib
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Tuple, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RANTV_FULL_TOKENS")

# Optional deps
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

# Kite
try:
    from kiteconnect import KiteConnect
    KITECONNECT_AVAILABLE = True
except Exception:
    KITECONNECT_AVAILABLE = False

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except Exception:
    WEBSOCKET_AVAILABLE = False

# ===================== CONFIG =====================
class AppConfig:
    KITE_API_KEY: str = os.environ.get("KITE_API_KEY", "")
    KITE_API_SECRET: str = os.environ.get("KITE_API_SECRET", "")
    INITIAL_CAPITAL: float = float(os.environ.get("INITIAL_CAPITAL", "2000000"))
    BASE_RISK: float = 0.01; MAX_RISK: float = 0.02; MIN_RISK: float = 0.005; MAX_DAILY_DD: float = 0.03
    SL_ATR: float = 1.5; TP_ATR: float = 3.0; TRAIL_ATR: float = 1.2
    ALGO_MAX_POSITIONS: int = int(os.environ.get("ALGO_MAX_POSITIONS", "5"))
    ALGO_MAX_DAILY_LOSS: float = float(os.environ.get("ALGO_MAX_DAILY_LOSS", "50000"))
    ALGO_MIN_CONFIDENCE: float = float(os.environ.get("ALGO_MIN_CONFIDENCE", "0.80"))
    MAX_DAILY_TRADES: int = int(os.environ.get("MAX_DAILY_TRADES", "10"))
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
    INDIA_OPEN = dt_time(9, 15); INDIA_CLOSE = dt_time(15, 30); DAILY_EXIT = dt_time(15, 35)
config = AppConfig()

# ===================== TIME =====================
now_indian = lambda: datetime.now(config.IND_TZ) if config.IND_TZ else datetime.now()
market_open = lambda: config.INDIA_OPEN <= now_indian().time() <= config.INDIA_CLOSE
should_exit_all_positions = lambda: now_indian().time() >= config.DAILY_EXIT

# ===================== UNIVERSES =====================
NIFTY_50 = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","HINDUNILVR.NS","ICICIBANK.NS","KOTAKBANK.NS","BHARTIARTL.NS","ITC.NS","LT.NS",
    "SBIN.NS","ASIANPAINT.NS","HCLTECH.NS","AXISBANK.NS","MARUTI.NS","SUNPHARMA.NS","TITAN.NS","ULTRACEMCO.NS","WIPRO.NS","NTPC.NS",
    "NESTLEIND.NS","POWERGRID.NS","M&M.NS","BAJFINANCE.NS","ONGC.NS","TATASTEEL.NS","JSWSTEEL.NS","ADANIPORTS.NS","COALINDIA.NS",
    "HDFCLIFE.NS","DRREDDY.NS","HINDALCO.NS","CIPLA.NS","SBILIFE.NS","GRASIM.NS","TECHM.NS","BAJAJFINSV.NS","BRITANNIA.NS","EICHERMOT.NS",
    "DIVISLAB.NS","SHREECEM.NS","APOLLOHOSP.NS","UPL.NS","BAJAJ-AUTO.NS","HEROMOTOCO.NS","INDUSINDBK.NS","ADANIENT.NS","TATACONSUM.NS","BPCL.NS"
]
NIFTY_100 = NIFTY_50 + [
    "BAJAJHLDNG.NS","TATAMOTORS.NS","VEDANTA.NS","PIDILITIND.NS","BERGEPAINT.NS","AMBUJACEM.NS","DABUR.NS","HAVELLS.NS","ICICIPRULI.NS",
    "MARICO.NS","PEL.NS","SIEMENS.NS","TORNTPHARM.NS","ACC.NS","AUROPHARMA.NS","BOSCHLTD.NS","GLENMARK.NS","MOTHERSUMI.NS","BIOCON.NS",
    "ZYDUSLIFE.NS","COLPAL.NS","CONCOR.NS","DLF.NS","GODREJCP.NS","HINDPETRO.NS","IBULHSGFIN.NS","IOC.NS","JINDALSTEL.NS","LUPIN.NS",
    "MANAPPURAM.NS","MCDOWELL-N.NS","NMDC.NS","PETRONET.NS","PFC.NS","PNB.NS","RBLBANK.NS","SAIL.NS","SRTRANSFIN.NS","TATAPOWER.NS",
    "YESBANK.NS","ZEEL.NS"
]
NIFTY_MIDCAP_150 = [
    "ABB.NS","ABCAPITAL.NS","ABFRL.NS","ACC.NS","AUBANK.NS","AIAENG.NS","APLAPOLLO.NS","ASTRAL.NS","AARTIIND.NS","BALKRISIND.NS",
    "BANKBARODA.NS","BANKINDIA.NS","BATAINDIA.NS","BEL.NS","BHARATFORG.NS","BHEL.NS","BIOCON.NS","BOSCHLTD.NS","BRIGADE.NS","CANBK.NS",
    "CANFINHOME.NS","CHOLAFIN.NS","CIPLA.NS","COALINDIA.NS","COFORGE.NS","COLPAL.NS","CONCOR.NS","COROMANDEL.NS","CROMPTON.NS","CUMMINSIND.NS",
    "DABUR.NS","DALBHARAT.NS","DEEPAKNTR.NS","DELTACORP.NS","DIVISLAB.NS","DIXON.NS","DLF.NS","DRREDDY.NS","EDELWEISS.NS","EICHERMOT.NS",
    "ESCORTS.NS","EXIDEIND.NS","FEDERALBNK.NS","GAIL.NS","GLENMARK.NS","GODREJCP.NS","GODREJPROP.NS","GRANULES.NS","GRASIM.NS","GUJGASLTD.NS",
    "HAL.NS","HAVELLS.NS","HCLTECH.NS","HDFCAMC.NS","HDFCLIFE.NS","HEROMOTOCO.NS","HINDALCO.NS","HINDPETRO.NS","HINDUNILVR.NS","ICICIPRULI.NS",
    "IDEA.NS","IDFCFIRSTB.NS","IGL.NS","INDIACEM.NS","INDIAMART.NS","INDUSTOWER.NS","INFY.NS","IOC.NS","IPCALAB.NS","JINDALSTEL.NS",
    "JSWENERGY.NS","JUBLFOOD.NS","KOTAKBANK.NS","L&TFH.NS","LICHSGFIN.NS","LT.NS","LTTS.NS","MANAPPURAM.NS","MARICO.NS","MARUTI.NS","MFSL.NS",
    "MGL.NS","MINDTREE.NS","MOTHERSUMI.NS","MPHASIS.NS","MRF.NS","MUTHOOTFIN.NS","NATIONALUM.NS","NAUKRI.NS","NESTLEIND.NS","NMDC.NS",
    "NTPC.NS","OBEROIRLTY.NS","OFSS.NS","ONGC.NS","PAGEIND.NS","PEL.NS","PETRONET.NS","PFC.NS","PIDILITIND.NS","PIIND.NS","PNB.NS",
    "POWERGRID.NS","RAJESHEXPO.NS","RAMCOCEM.NS","RBLBANK.NS","RECLTD.NS","RELIANCE.NS","SAIL.NS","SBICARD.NS","SBILIFE.NS","SHREECEM.NS",
    "SIEMENS.NS","SRF.NS","SRTRANSFIN.NS","SUNPHARMA.NS","SUNTV.NS","SYNGENE.NS","TATACHEM.NS","TATACONSUM.NS","TATAMOTORS.NS","TATAPOWER.NS",
    "TATASTEEL.NS","TCS.NS","TECHM.NS","TITAN.NS","TORNTPHARM.NS","TRENT.NS","UPL.NS","VOLTAS.NS","WIPRO.NS","YESBANK.NS","ZEEL.NS"
]
ALL_STOCKS = list(dict.fromkeys(NIFTY_50 + NIFTY_100 + NIFTY_MIDCAP_150))

# ===================== INDICATORS =====================
def ema(s: pd.Series, span: int): return s.ewm(span=span, adjust=False).mean()

def rsi(s: pd.Series, period: int = 14):
    d = s.diff(); g = d.clip(lower=0).rolling(period).mean(); l = (-d.clip(upper=0)).rolling(period).mean()
    rs = g / l.replace(0, np.nan); rs = rs.fillna(0); return 100 - (100/(1+rs))

def atr(h: pd.Series, l: pd.Series, c: pd.Series, period=14):
    tr1 = h-l; tr2=(h-c.shift()).abs(); tr3=(l-c.shift()).abs(); tr=pd.concat([tr1,tr2,tr3],axis=1).max(axis=1); return tr.rolling(period).mean()

def macd(c: pd.Series, fast=12, slow=26, signal=9):
    ef=ema(c,fast); es=ema(c,slow); ml=ef-es; sl=ema(ml,signal); return ml,sl,ml-sl

def bb(c: pd.Series, period=20, std=2):
    sma=c.rolling(period).mean(); st=c.rolling(period).std(); return sma+st*std, sma, sma-st*std

def support_resistance(h: pd.Series, l: pd.Series, c: pd.Series, period=20):
    try:
        res, sup = [], []
        ln=len(h)
        if ln < period*2+1: return {"support":float(c.iloc[-1]*0.98),"resistance":float(c.iloc[-1]*1.02)}
        for i in range(period, ln-period):
            if h.iloc[i] >= h.iloc[i-period:i+period+1].max(): res.append(float(h.iloc[i]))
            if l.iloc[i] <= l.iloc[i-period:i+period+1].min(): sup.append(float(l.iloc[i]))
        rres = sorted(res)[-3:] if res else [float(c.iloc[-1]*1.02)]
        rsup = sorted(sup)[:3] if sup else [float(c.iloc[-1]*0.98)]
        return {"support":float(np.mean(rsup)), "resistance":float(np.mean(rres)), "support_levels":rsup, "resistance_levels":rres}
    except Exception:
        cp=float(c.iloc[-1]); return {"support":cp*0.98,"resistance":cp*1.02,"support_levels":[],"resistance_levels":[]}

# ===================== SMC (simplified, from your v2) =====================
class SMC:
    @staticmethod
    def bos(df: pd.DataFrame, lookback=6):
        if len(df) < lookback+1: return None
        if df['High'].iloc[-1] > df['High'].iloc[-lookback:-1].max(): return 'BULLISH'
        if df['Low'].iloc[-1]  < df['Low'].iloc[-lookback:-1].min(): return 'BEARISH'
        return None
    @staticmethod
    def fvg(df: pd.DataFrame):
        if len(df) < 3: return None
        a,b,c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        if a['High'] < c['Low']:  return ('BULLISH', float(a['High']), float(c['Low']))
        if a['Low']  > c['High']: return ('BEARISH', float(c['High']), float(a['Low']))
        return None
    @staticmethod
    def ob(df: pd.DataFrame):
        if len(df) < 2: return None
        last = df.iloc[-2]
        if last['Close'] > last['Open']: return 'BULLISH'
        if last['Close'] < last['Open']: return 'BEARISH'
        return None
    @staticmethod
    def liquidity(df: pd.DataFrame):
        if len(df) < 10: return None
        rh=df['High'].iloc[-10:-1].max(); rl=df['Low'].iloc[-10:-1].min(); ch=df['High'].iloc[-1]; cl=df['Low'].iloc[-1]; cc=df['Close'].iloc[-1]
        if cl<rl and cc>rl: return 'BULLISH'
        if ch>rh and cc<rh: return 'BEARISH'
        return None

def poc(df: pd.DataFrame, bins=24):
    try:
        if df is None or len(df) < bins: return float(df['Close'].iloc[-1]) if df is not None and len(df) else 0.0
        prices=df['Close'].values; vols=df['Volume'].values
        hist,edges=np.histogram(prices,bins=bins,weights=vols)
        i=int(np.argmax(hist)); return float((edges[i]+edges[i+1])/2) if i < len(edges)-1 else float(df['Close'].iloc[-1])
    except Exception:
        return float(df['Close'].iloc[-1]) if df is not None and len(df) else 0.0

# ===================== Kite Manager with FULL token map =====================
class KiteManager:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key; self.api_secret = api_secret
        self.kite = None; self.is_authenticated = False
        self.token_map: Dict[str,int] = {}
        if KITECONNECT_AVAILABLE and api_key and api_secret:
            try:
                self.kite = KiteConnect(api_key=api_key)
            except Exception as e:
                logger.error(f"Init Kite failed: {e}")
    def login_url(self) -> Optional[str]:
        try:
            return self.kite.login_url() if self.kite else None
        except Exception:
            return None
    def authenticate(self, request_token: str) -> Tuple[bool,str]:
        try:
            if not self.kite: return False, "Kite not initialized"
            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            access = data.get('access_token')
            if not access: return False, "No access_token"
            self.kite.set_access_token(access); self.is_authenticated=True
            return True, f"Authenticated as {data.get('user_name','User')}"
        except Exception as e:
            return False, f"Auth error: {e}"
    def build_token_map(self) -> Tuple[bool,str]:
        try:
            if not (self.kite and self.is_authenticated): return False, "Authenticate first"
            instruments = self.kite.instruments("NSE")  # full dump
            # tradingsymbol -> instrument_token
            self.token_map = { i['tradingsymbol']: i['instrument_token'] for i in instruments if 'tradingsymbol' in i }
            logger.info(f"Token map built: {len(self.token_map)} instruments")
            return True, f"Loaded {len(self.token_map)} NSE tokens"
        except Exception as e:
            logger.error(f"Token map error: {e}"); return False, str(e)
    def token_for_symbol(self, symbol_with_ns: str) -> Optional[int]:
        # Convert 'RELIANCE.NS' -> 'RELIANCE'
        sym = symbol_with_ns.replace('.NS','')
        return self.token_map.get(sym)

# ===================== Live Ticker (WebSocket) =====================
class KiteLiveTicker:
    def __init__(self, kite_mgr: KiteManager):
        self.kite_mgr = kite_mgr; self.ws=None; self.subscribed=set(); self.is_connected=False
        self.candle: Dict[int, Dict[str, List[float]]] = {}
    def connect(self):
        if not WEBSOCKET_AVAILABLE: return False
        if not (self.kite_mgr and self.kite_mgr.is_authenticated): return False
        try:
            url=f"wss://ws.kite.trade?api_key={self.kite_mgr.api_key}&access_token={self.kite_mgr.kite.access_token}"
            self.ws = websocket.WebSocketApp(url, on_open=self._on_open, on_message=self._on_msg, on_error=self._on_err, on_close=self._on_close)
            th=threading.Thread(target=self.ws.run_forever, daemon=True); th.start(); self.is_connected=True; return True
        except Exception as e:
            logger.error(f"WS connect error: {e}"); return False
    def _on_open(self, ws):
        logger.info("WebSocket opened")
    def _on_err(self, ws, e): logger.error(f"WS error: {e}")
    def _on_close(self, ws, code, msg): logger.info(f"WS closed {code} {msg}"); self.is_connected=False; self.subscribed.clear()
    def _on_msg(self, ws, message):
        try:
            data=json.loads(message)
            if isinstance(data, dict) and data.get('type')=='ticks':
                for t in data.get('ticks',[]):
                    tok=t.get('instrument_token'); p=t.get('last_price',0); v=t.get('volume_traded',0)
                    if tok not in self.candle:
                        self.candle[tok]={'open':[],'high':[],'low':[],'close':[],'volume':[],'timestamp':[]}
                    now=time.time(); interval=60
                    cd=self.candle[tok]
                    if not cd['timestamp']:
                        start=now-(now%interval); cd['timestamp'].append(start)
                        cd['open'].append(p); cd['high'].append(p); cd['low'].append(p); cd['close'].append(p); cd['volume'].append(v)
                    else:
                        start=cd['timestamp'][-1]
                        if now-start>=interval:
                            new_start=now-(now%interval); cd['timestamp'].append(new_start)
                            cd['open'].append(p); cd['high'].append(p); cd['low'].append(p); cd['close'].append(p); cd['volume'].append(v)
                        else:
                            i=-1; cd['high'][i]=max(cd['high'][i],p); cd['low'][i]=min(cd['low'][i],p); cd['close'][i]=p; cd['volume'][i]=v
        except Exception as e:
            logger.error(f"WS msg error: {e}")
    def subscribe(self, tokens: List[int]) -> bool:
        if not (self.is_connected and self.ws): return False
        try:
            self.ws.send(json.dumps({"a":"subscribe","v":tokens}))
            for t in tokens: self.subscribed.add(t)
            return True
        except Exception as e:
            logger.error(f"Sub error: {e}"); return False
    def get_candles(self, token: int, n: int = 120) -> Optional[Dict[str,List[float]]]:
        if token not in self.candle: return None
        d=self.candle[token]; m=min(n, len(d['timestamp']))
        return {k: d[k][-m:] for k in d}

# ===================== Data Manager =====================
class DataManager:
    def __init__(self, kite_mgr: Optional[KiteManager]=None):
        self.kite_mgr=kite_mgr; self.cache={}; self.ttl=30
    def _get(self, symbol: str, interval: str):
        key=f"{symbol}_{interval}"; c=self.cache.get(key)
        if c and time.time()-c[1] < self.ttl: return c[0]
        # Kite historical optional (after auth)
        if self.kite_mgr and self.kite_mgr.is_authenticated:
            # use Yahoo for historical to avoid rate limits; keep as fallback
            pass
        # Yahoo
        try:
            pm={"1m":"1d","5m":"5d","15m":"15d","30m":"30d","1h":"60d","1d":"1y"}
            period=pm.get(interval,"15d")
            df=yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False) if YFINANCE_AVAILABLE else None
            if df is None or df.empty or len(df) < 20: return None
            df.columns=[c.capitalize() for c in df.columns]; df=df[["Open","High","Low","Close","Volume"]].dropna()
            # indicators
            df['EMA8']=ema(df['Close'],8); df['EMA21']=ema(df['Close'],21); df['EMA50']=ema(df['Close'],50)
            df['RSI14']=rsi(df['Close'],14); df['ATR']=atr(df['High'],df['Low'],df['Close'])
            m,s,h=macd(df['Close']); df['MACD'],df['MACD_Signal'],df['MACD_Hist']=m,s,h
            u,mid,l=bb(df['Close']); df['BB_Upper'],df['BB_Middle'],df['BB_Lower']=u,mid,l
            tp=(df['High']+df['Low']+df['Close'])/3; df['VWAP']=(tp*df['Volume']).cumsum()/df['Volume'].cumsum()
            sr=support_resistance(df['High'],df['Low'],df['Close']); df['Support']=sr['support']; df['Resistance']=sr['resistance']
            self.cache[key]=(df.copy(), time.time()); return df
        except Exception as e:
            logger.error(f"Yahoo error {symbol}: {e}"); return None
    def get_stock(self, symbol:str, interval="15m"): return self._get(symbol, interval)
    def live_price(self, symbol:str):
        df=self._get(symbol,"5m"); return float(df['Close'].iloc[-1]) if df is not None and len(df)>0 else None

# ===================== Signals 
class SignalGenerator:
    def __init__(self, dm: DataManager): self.dm=dm
    def smc(self, symbol:str):
        ltf=self.dm.get_stock(symbol,"15m"); htf=self.dm.get_stock(symbol,"1h")
        if ltf is None or htf is None or len(ltf)<50 or len(htf)<20 or not market_open(): return None
        price=float(ltf['Close'].iloc[-1]); _bos=SMC.bos(htf); _fvg=SMC.fvg(ltf); _ob=SMC.ob(ltf); _liq=SMC.liquidity(ltf); _poc=poc(ltf)
        atrv=float(ltf['ATR'].iloc[-1]) if 'ATR' in ltf.columns else max(0.02*price,1)
        sig=None; conf=0.0
        if _bos=='BULLISH' and (_fvg and _fvg[0]=='BULLISH') and _ob=='BULLISH' and price>_poc:
            conf=0.95 if _liq=='BULLISH' else 0.85; sig={'action':'BUY','stop_loss':price-config.SL_ATR*atrv,'target':price+config.TP_ATR*atrv}
        elif _bos=='BEARISH' and (_fvg and _fvg[0]=='BEARISH') and _ob=='BEARISH' and price<_poc:
            conf=0.95 if _liq=='BEARISH' else 0.85; sig={'action':'SELL','stop_loss':price+config.SL_ATR*atrv,'target':price-config.TP_ATR*atrv}
        if not sig: return None
        win=min(0.98, conf*0.95)
        return {'symbol':symbol,'price':round(price,2),'confidence':round(conf,3),'win_probability':round(win,3),'strategy':'MTF SMC + Volume Profile','atr':round(atrv,2),'poc':round(_poc,2),**sig}
    def tech(self, symbol:str):
        df=self.dm.get_stock(symbol,"15m"); 
        if df is None or len(df)<50: return None
        p=float(df['Close'].iloc[-1]); out=[]
        if p>df['EMA8'].iloc[-1]>df['EMA21'].iloc[-1] and p>df['VWAP'].iloc[-1]: out.append({'action':'BUY','confidence':0.8})
        elif p<df['EMA8'].iloc[-1]<df['EMA21'].iloc[-1] and p<df['VWAP'].iloc[-1]: out.append({'action':'SELL','confidence':0.8})
        r=float(df['RSI14'].iloc[-1]); 
        if r<30: out.append({'action':'BUY','confidence':0.75})
        elif r>70: out.append({'action':'SELL','confidence':0.75})
        if p<=df['BB_Lower'].iloc[-1]*1.01: out.append({'action':'BUY','confidence':0.7})
        elif p>=df['BB_Upper'].iloc[-1]*0.99: out.append({'action':'SELL','confidence':0.7})
        m=df['MACD'].iloc[-1]; s=df['MACD_Signal'].iloc[-1]; pm=df['MACD'].iloc[-2]; ps=df['MACD_Signal'].iloc[-2]
        if m>s and pm<=ps: out.append({'action':'BUY','confidence':0.75})
        elif m<s and pm>=ps: out.append({'action':'SELL','confidence':0.75})
        sup=df['Support'].iloc[-1]; res=df['Resistance'].iloc[-1]
        if p<=sup*1.01: out.append({'action':'BUY','confidence':0.8})
        elif p>=res*0.99: out.append({'action':'SELL','confidence':0.8})
        if not out: return None
        b=[d for d in out if d['action']=='BUY']; s=[d for d in out if d['action']=='SELL']
        sb=sum(d['confidence'] for d in b); ss=sum(d['confidence'] for d in s)
        act='BUY' if sb>ss else 'SELL'; conf=(sb if act=='BUY' else ss)/max(1,len(b if act=='BUY' else s))
        av=float(df['ATR'].iloc[-1]) if 'ATR' in df.columns else max(0.02*p,1)
        sl=p-1.5*av if act=='BUY' else p+1.5*av; tg=p+3.0*av if act=='BUY' else p-3.0*av
        return {'symbol':symbol,'price':round(p,2),'action':act,'stop_loss':round(sl,2),'target':round(tg,2),'confidence':round(conf,3),'win_probability':round(min(0.97,conf*0.85),3),'strategy':'Multi-Strategy Confluence'}
    def generate(self, symbol:str):
        s=self.smc(symbol)
        if s and s['confidence']>=config.ALGO_MIN_CONFIDENCE: return s
        return self.tech(symbol)
    def scan(self, universe: List[str], k:int=15):
        arr=[]
        for s in universe:
            g=self.generate(s)
            if g and g['confidence']>=config.ALGO_MIN_CONFIDENCE: arr.append(g)
        arr.sort(key=lambda d:(d.get('confidence',0), d.get('win_probability',0)), reverse=True)
        return arr[:k]

# ===================== Risk & Trader =====================
class AutoRiskScaler:
    def __init__(self, cap: float):
        self.eq = [cap]
        self.rets = []
        self.maxdd = 0.0
        self.curdd = 0.0

    def update(self, pnl: float):
        new = self.eq[-1] + pnl
        self.eq.append(new)
        if len(self.eq) > 1:
            r = (new - self.eq[-2]) / max(1e-9, self.eq[-2])
            self.rets.append(r)
        peak = max(self.eq)
        self.curdd = (peak - new) / peak if peak > 0 else 0.0
        self.maxdd = max(self.maxdd, self.curdd)

    def risk(self) -> float:
        if len(self.rets) < 10:
            return config.BASE_RISK
        vol = np.std(self.rets[-10:]) or 0.02
        avg = np.mean(self.rets[-10:]) if self.rets else 0.0
        sh = avg / vol if vol > 0 else 0.0
        r = config.BASE_RISK
        if self.curdd < config.MAX_DAILY_DD * 0.5 and sh > 0:
            r *= min(2.0, 1.0 + 8.0 * sh)
        elif self.curdd > config.MAX_DAILY_DD * 0.8 or sh < 0:
            r *= max(0.3, 1.0 - 5.0 * abs(sh))
        return float(max(config.MIN_RISK, min(config.MAX_RISK, r)))
class RiskManager:
    def __init__(self): self.scaler=AutoRiskScaler(config.INITIAL_CAPITAL); self.daily_pnl=0.0; self.pos=0; self.trades=0
    def status(self):
        return {'daily_pnl':self.daily_pnl,'positions_open':self.pos,'trades_today':self.trades,'dynamic_risk':self.scaler.risk(),'within_limits': self.daily_pnl>-config.ALGO_MAX_DAILY_LOSS and self.pos<=config.ALGO_MAX_POSITIONS,'drawdown':self.scaler.curdd}
    def approve(self, conf:float):
        stt=self.status()
        if not market_open(): return False, "Market closed"
        if conf<config.ALGO_MIN_CONFIDENCE: return False, "Low confidence"
        if not stt['within_limits']: return False, "Risk limits breached"
        if self.trades>=config.MAX_DAILY_TRADES: return False, "Daily trade limit reached"
        return True, "OK"
    def on_open(self): self.pos+=1; self.trades+=1
    def on_close(self,pnl:float): self.pos=max(0,self.pos-1); self.daily_pnl+=pnl; self.scaler.update(pnl)

class Trader:
    def __init__(self, dm: DataManager, risk: RiskManager): self.dm=dm; self.risk=risk; self.cash=config.INITIAL_CAPITAL; self.positions={}; self.trades_log=[]
    def size(self, price:float, smc:bool):
        pct=self.risk.status()['dynamic_risk']; pct*=1.2 if smc else 1.0; val=self.cash*pct; qty=int(val/max(1e-6,price)); return max(1,min(qty,200))
    def open_from_signal(self, sig:Dict[str,Any]):
        ok, reason = self.risk.approve(sig.get('confidence',0.0))
        if not ok: return False, reason
        sym=sig['symbol']; price=sig['price']; action=sig['action']; is_smc=('poc' in sig)
        qty=self.size(price,is_smc); val=qty*price
        if action=='BUY' and val>self.cash: return False, "Insufficient cash"
        self.positions[sym]={'symbol':sym,'action':action,'qty':qty,'entry':price,'sl':sig.get('stop_loss'),'tg':sig.get('target'),'opened':now_indian(),'strategy':sig.get('strategy','N/A'),'is_smc':is_smc}
        if action=='BUY': self.cash-=val
        else: self.cash-=val*0.2
        self.trades_log.append({**self.positions[sym],'status':'OPEN'}); self.risk.on_open(); return True, f"Opened {action} {qty} {sym} @ ₹{price:.2f}"
    def mark(self,sym): df=self.dm.get_stock(sym,"5m"); return float(df['Close'].iloc[-1]) if df is not None and len(df)>0 else None
    def update(self):
        for sym,pos in list(self.positions.items()):
            p=self.mark(sym); 
            if p is None: continue
            pnl=(p-pos['entry'])*pos['qty'] if pos['action']=='BUY' else (pos['entry']-p)*pos['qty']; pos['pnl']=pnl
            sl=pos.get('sl'); tg=pos.get('tg')
            if ((pos['action']=='BUY' and sl is not None and p<=sl) or (pos['action']=='SELL' and sl is not None and p>=sl)) or ((pos['action']=='BUY' and tg is not None and p>=tg) or (pos['action']=='SELL' and tg is not None and p<=tg)) or should_exit_all_positions():
                self.close(sym, p)
    def close(self,sym, p=None):
        if sym not in self.positions: return False, "Not found"
        pos=self.positions[sym]; price=p or self.mark(sym) or pos['entry']
        pnl=(price-pos['entry'])*pos['qty'] if pos['action']=='BUY' else (pos['entry']-price)*pos['qty']
        if pos['action']=='BUY': self.cash+=pos['qty']*price
        else: self.cash+=pos['qty']*pos['entry']*0.2
        pos['status']='CLOSED'; pos['exit']=price; pos['closed']=now_indian(); self.trades_log.append({**pos}); del self.positions[sym]; self.risk.on_close(pnl)
        return True, f"Closed {sym} @ ₹{price:.2f} | P&L: ₹{pnl:+.2f}"
    def close_all(self):
        for sym in list(self.positions.keys()): self.close(sym)
    def perf(self):
        closed=[t for t in self.trades_log if t.get('status')=='CLOSED']
        wins=[t for t in closed if ((t['action']=='BUY' and t['exit']>t['entry']) or (t['action']=='SELL' and t['exit']<t['entry']))]
        total_pnl=sum(((t['exit']-t['entry'])*t['qty']) if t['action']=='BUY' else ((t['entry']-t['exit'])*t['qty']) for t in closed)
        return {'total_trades':len(closed),'win_rate':len(wins)/len(closed) if closed else 0.0,'total_pnl':total_pnl,'open_positions':len(self.positions),'cash':self.cash}

# ===================== Algo =====================
class Algo:
    def __init__(self, trader: Trader, sig: SignalGenerator): self.trader=trader; self.sig=sig; self._stop=threading.Event(); self.state='stopped'; self.th=None; self.universe=NIFTY_50
    def start(self):
        if self.state=='running': return False
        self.state='running'; self._stop.clear(); self.th=threading.Thread(target=self._loop, daemon=True); self.th.start(); return True
    def stop(self): self.state='stopped'; self._stop.set();
    def _loop(self):
        last=0.0
        while not self._stop.is_set():
            if self.state!='running' or not market_open(): time.sleep(2); continue
            if should_exit_all_positions(): self.trader.close_all(); time.sleep(60); continue
            self.trader.update()
            if time.time()-last>300:
                sigs=self.sig.scan(self.universe, k=15)
                for s in sigs:
                    if s['symbol'] not in self.trader.positions and self.trader.risk.pos < config.ALGO_MAX_POSITIONS:
                        ok,msg=self.trader.open_from_signal(s); logger.info(msg)
                last=time.time()
            time.sleep(5)

# ===================== Email =====================
def send_daily_report_email(trader: Trader):
    try:
        if not config.EMAIL_SENDER or not config.EMAIL_PASSWORD: return False
        perf=trader.perf(); today=now_indian().date().strftime('%Y-%m-%d')
        subject=f"Daily Trading Report - {today}"
        rows=[]
        for sym,pos in trader.positions.items(): rows.append(f"<tr><td>{sym}</td><td>{pos['action']}</td><td>{pos['qty']}</td><td>₹{pos['entry']:.2f}</td><td>OPEN</td></tr>")
        closed=[t for t in trader.trades_log if t.get('status')=='CLOSED']
        for t in closed[-30:]: pnl=((t['exit']-t['entry'])*t['qty']) if t['action']=='BUY' else ((t['entry']-t['exit'])*t['qty'])
        html=f"""
        <html><body style="font-family:Arial,sans-serif"><h2 style="color:#f97316">📈 Daily Trading Report</h2>
        <p><b>Total trades:</b> {perf['total_trades']} | <b>Win rate:</b> {perf['win_rate']:.1%} | <b>Total P&L:</b> ₹{perf['total_pnl']:+.2f}</p>
        <p style="color:#64748b;font-size:12px">Generated at {now_indian().strftime('%Y-%m-%d %H:%M:%S')}</p></body></html>"""
        msg=MIMEMultipart('alternative'); msg['Subject']=subject; msg['From']=config.EMAIL_SENDER; msg['To']=config.EMAIL_RECEIVER; msg.attach(MIMEText(html,'html'))
        with smtplib.SMTP_SSL('smtp.gmail.com',465) as s:
            s.login(config.EMAIL_SENDER, config.EMAIL_PASSWORD); s.send_message(msg)
        return True
    except Exception as e: logger.error(f"Email error: {e}"); return False

# ===================== UI =====================
if STREAMLIT_AVAILABLE:
    st.set_page_config(page_title="RANTV Intraday Terminal Pro — FULL MERGE (Tokens)", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
    <style>
    .stApp {background: linear-gradient(135deg,#fff5e6 0%, #ffe8cc 100%);} .main .block-container {background-color: transparent; padding-top: 0.8rem;}
    .main-header {text-align:center; margin:10px auto; padding:12px; background: linear-gradient(135deg,#ff8c00 0%, #ff6b00 100%); border-radius: 14px; color: white; box-shadow: 0 8px 25px rgba(255,140,0,0.30);} 
    .stTabs [data-baseweb="tab-list"] {gap:4px; background: linear-gradient(135deg,#ffe8cc 0%, #ffd9a6 50%, #ffca80 100%); padding:8px; border-radius:12px; margin-bottom:10px;} 
    .stTabs [data-baseweb="tab"] { height: 60px; white-space: pre-wrap; background:#ffffff; border-radius:8px; gap:8px; padding:12px 20px; font-weight:600; font-size:14px; color:#d97706; border:2px solid transparent; box-shadow:0 2px 4px rgba(0,0,0,0.1);} 
    .stTabs [aria-selected="true"] {background: linear-gradient(135deg,#ff8c00 0%, #ff6b00 100%); color:white; border:2px solid #ff8c00; box-shadow:0 4px 8px rgba(255,140,0,0.3);} 
    [data-testid="stMetricValue"]{color:#ff8c00;font-weight:800}
    </style>
    """, unsafe_allow_html=True)

    # Sidebar — more robust Kite UI
    st.sidebar.header("🔐 Kite Connect (Live)")
    if 'kite_api_key' not in st.session_state: st.session_state.kite_api_key = config.KITE_API_KEY
    if 'kite_api_secret' not in st.session_state: st.session_state.kite_api_secret = config.KITE_API_SECRET
    api_key = st.sidebar.text_input("API Key", value=st.session_state.kite_api_key)
    api_secret = st.sidebar.text_input("API Secret", value=st.session_state.kite_api_secret, type="password")
    if st.sidebar.button("Save & Init"):
        st.session_state.kite_api_key = api_key; st.session_state.kite_api_secret = api_secret
        st.session_state.kite_mgr = KiteManager(api_key, api_secret) if (KITECONNECT_AVAILABLE and api_key and api_secret) else None
        if not KITECONNECT_AVAILABLE:
            st.sidebar.error("kiteconnect package not installed")
        elif st.session_state.kite_mgr is None:
            st.sidebar.error("Provide valid API key/secret")
        else:
            st.sidebar.success("Kite initialized")

    kite_mgr: Optional[KiteManager] = st.session_state.get('kite_mgr')
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
            if ok:
                # build token mapping immediately
                ok2,msg2 = kite_mgr.build_token_map()
                (st.sidebar.success if ok2 else st.sidebar.error)(msg2)
                st.session_state.token_map = kite_mgr.token_map

    # Compose system
    if 'dm' not in st.session_state: st.session_state.dm = DataManager(kite_mgr)
    if 'risk' not in st.session_state: st.session_state.risk = RiskManager()
    if 'sig' not in st.session_state: st.session_state.sig = SignalGenerator(st.session_state.dm)
    if 'trader' not in st.session_state: st.session_state.trader = Trader(st.session_state.dm, st.session_state.risk)
    if 'algo' not in st.session_state: st.session_state.algo = Algo(st.session_state.trader, st.session_state.sig)

    trader: Trader = st.session_state.trader; algo: Algo = st.session_state.algo

    st.markdown('<div class="main-header"><h2>RANTV Intraday Terminal Pro — FULL MERGE (Tokens)</h2><p>Live Charts + Full Universes + SMC + Risk Scaling + Full NSE Tokens</p></div>', unsafe_allow_html=True)

    perf = trader.perf(); m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Cash", f"₹{perf['cash']:,.0f}"); m2.metric("Open Positions", perf['open_positions']); m3.metric("Closed Trades", perf['total_trades'])
    m4.metric("Win Rate", f"{perf['win_rate']:.1%}"); m5.metric("Total P&L", f"₹{perf['total_pnl']:+,.0f}")

    tabs = st.tabs(["📊 Signals","📈 Live Charts","💼 Positions","🤖 Algo","🧠 Universes","📬 Reports"]) 

    # Signals
    with tabs[0]:
        st.subheader("Top Signals (Universe)")
        uni_name = st.selectbox("Universe", ["NIFTY 50","NIFTY 100","MIDCAP 150","ALL"], index=0)
        universe = NIFTY_50 if uni_name=="NIFTY 50" else NIFTY_100 if uni_name=="NIFTY 100" else NIFTY_MIDCAP_150 if uni_name=="MIDCAP 150" else ALL_STOCKS
        k = st.slider("Max signals", 5, 40, 15)
        if st.button("Scan Now", type="primary"):
            sigs = st.session_state.sig.scan(universe, k=k)
            st.session_state['latest_sigs']=sigs
        sigs=st.session_state.get('latest_sigs', [])
        for s in sigs:
            c1,c2,c3,c4,c5,c6 = st.columns([2,1,1,1,2,1])
            c1.write(f"**{s['symbol']}**"); c2.write(s['action']); c3.write(f"Conf: {s['confidence']:.2f}")
            c4.write(f"Win: {s['win_probability']:.2f}"); c5.write(f"SL: ₹{s['stop_loss']:.2f} • TG: ₹{s['target']:.2f}")
            if c6.button("Trade", key=f"trade_{s['symbol']}_{time.time()}"):
                ok,msg = trader.open_from_signal(s); (st.success if ok else st.error)(msg)

    # Live Charts with full token mapping
    with tabs[1]:
        st.subheader("Kite Connect Live Charts (WebSocket)")
        if not kite_mgr or not kite_mgr.is_authenticated:
            st.warning("Authenticate in the sidebar to enable live charts.")
        else:
            if 'ticker' not in st.session_state: st.session_state.ticker = KiteLiveTicker(kite_mgr)
            ticker: KiteLiveTicker = st.session_state.ticker
            colA,colB = st.columns(2)
            if not ticker.is_connected:
                if colA.button("🔗 Connect Live Feed", type="primary"): ok=ticker.connect(); (st.success if ok else st.error)("Connected" if ok else "Failed")
            else:
                if colA.button("⛔ Disconnect", type="secondary"): ticker._on_close(None,1000,"User"); st.info("Disconnected")
            colB.metric("Feed", "🟢 CONNECTED" if ticker.is_connected else "🔴 DISCONNECTED")

            # INDEX TOKENS
            INDEX_TOKENS = {"NIFTY 50":256265, "BANK NIFTY":260105}
            idx_tab, stk_tab = st.tabs(["📈 Index","📊 Stock"])
            with idx_tab:
                idx = st.selectbox("Index", list(INDEX_TOKENS.keys()))
                tok = INDEX_TOKENS[idx]
                if ticker.is_connected:
                    ticker.subscribe([tok])
                    data=ticker.get_candles(tok, n=180)
                    if data and PLOTLY_AVAILABLE:
                        ts=[datetime.fromtimestamp(x) for x in data['timestamp']]
                        fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.03,row_heights=[0.7,0.3])
                        fig.add_trace(go.Candlestick(x=ts,open=data['open'],high=data['high'],low=data['low'],close=data['close'],increasing_line_color='#26a69a',decreasing_line_color='#ef5350'),row=1,col=1)
                        fig.add_trace(go.Bar(x=ts,y=data['volume'],marker_color='#ff8c00',opacity=0.7,name='Volume'),row=2,col=1)
                        fig.update_layout(title=f"{idx} Live (1m)",xaxis_rangeslider_visible=False,height=600,template='plotly_white')
                        st.plotly_chart(fig, use_container_width=True)
            with stk_tab:
                st.write("Select from full universes using the token map built after auth.")
                universe_choice = st.radio("List", ["NIFTY 50","NIFTY 100","MIDCAP 150","ALL"], horizontal=True)
                uni = NIFTY_50 if universe_choice=="NIFTY 50" else NIFTY_100 if universe_choice=="NIFTY 100" else NIFTY_MIDCAP_150 if universe_choice=="MIDCAP 150" else ALL_STOCKS
                # map to tradingsymbols
                if 'token_map' in st.session_state and st.session_state.token_map:
                    tradings = [s.replace('.NS','') for s in uni if s.replace('.NS','') in st.session_state.token_map]
                    selected = st.selectbox("Stock (tradingsymbol)", sorted(tradings))
                    tok = st.session_state.token_map.get(selected)
                    if tok and ticker.is_connected:
                        ticker.subscribe([tok])
                        data = ticker.get_candles(tok, n=180)
                        if data and PLOTLY_AVAILABLE:
                            ts=[datetime.fromtimestamp(x) for x in data['timestamp']]
                            fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.03,row_heights=[0.7,0.3])
                            fig.add_trace(go.Candlestick(x=ts,open=data['open'],high=data['high'],low=data['low'],close=data['close'],increasing_line_color='#26a69a',decreasing_line_color='#ef5350'),row=1,col=1)
                            fig.add_trace(go.Bar(x=ts,y=data['volume'],marker_color='#ff8c00',opacity=0.7,name='Volume'),row=2,col=1)
                            fig.update_layout(title=f"{selected} Live (1m)",xaxis_rangeslider_visible=False,height=600,template='plotly_white')
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Token map not built yet. After authentication, the app will load full NSE tokens.")

    # Positions
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
        closed=[t for t in trader.trades_log if t.get('status')=='CLOSED'][-30:]
        if closed:
            df=pd.DataFrame([{k:v for k,v in t.items() if k in ('symbol','action','qty','entry','exit','strategy','is_smc')} for t in closed])
            st.dataframe(df, use_container_width=True)
        else:
            st.write("No closed trades yet.")

    # Algo
    with tabs[3]:
        st.subheader("Algo Engine")
        col1,col2,col3 = st.columns(3)
        if algo.state!='running':
            if col1.button("Start Algo", type="primary"): algo.start(); st.success("Algo started")
        else:
            if col1.button("Stop Algo", type="secondary"): algo.stop(); st.warning("Algo stopped")
        uni_for_algo = st.selectbox("Universe", ["NIFTY 50","NIFTY 100","MIDCAP 150","ALL"], index=0)
        algo.universe = NIFTY_50 if uni_for_algo=="NIFTY 50" else NIFTY_100 if uni_for_algo=="NIFTY 100" else NIFTY_MIDCAP_150 if uni_for_algo=="MIDCAP 150" else ALL_STOCKS
        if col2.button("Force Scan"):
            sigs=st.session_state.sig.scan(algo.universe, k=15)
            for s in sigs:
                if s['symbol'] not in trader.positions and st.session_state.risk.pos < config.ALGO_MAX_POSITIONS:
                    ok,msg=trader.open_from_signal(s); (st.success if ok else st.error)(msg)
        if col3.button("Exit All"):
            trader.close_all(); st.warning("All positions closed")
        st.caption("Algo scans every ~5 minutes during market hours and auto-exits at 3:35 PM.")

    # Universes
    with tabs[4]:
        st.subheader("Stock Universes")
        st.write(f"NIFTY50: {len(NIFTY_50)} | NIFTY100: {len(NIFTY_100)} | MIDCAP150: {len(NIFTY_MIDCAP_150)} | ALL: {len(ALL_STOCKS)}")
        st.expander("Show samples").write(pd.DataFrame({'NIFTY50':NIFTY_50[:20],'NIFTY100':NIFTY_100[:20],'MIDCAP150':NIFTY_MIDCAP_150[:20]}))

    # Reports
    with tabs[5]:
        st.subheader("Daily Email Report")
        if st.button("Send Now"): ok=send_daily_report_email(trader); (st.success if ok else st.error)("Report sent" if ok else "Failed (check EMAIL_* env vars)")

else:
    if __name__=='__main__':
        print("Install streamlit/plotly/yfinance/websocket-client/kiteconnect")
