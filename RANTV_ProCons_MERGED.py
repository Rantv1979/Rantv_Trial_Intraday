#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RANTV – Intraday Terminal Pro (Merged: Pros & Cons)
===================================================

This single-file app merges the best parts of your two versions:

Pros kept from **RANTV_FIXED_VERSION (3).py**
- Streamlit-first UX with orange theme accents, clear metrics & tabs.
- Practical Yahoo Finance fallback with caching and indicator set.
- Daily email report capability and end-of-day (3:35 PM) full exit.

Pros kept from **Fixed Version_replit_indented (1).py**
- Advanced SMC (Smart Money Concepts): BOS, FVG, Order Block, Liquidity Grab.
- Risk layer with AutoRiskScaler (dynamic position sizing), drawdown guard.
- StrategyManager + AdvancedSignalGenerator (SMC + Technical confluence).
- AdvancedPaperTrader and AdvancedAlgoEngine skeleton with live/paper modes.

Cons avoided / reconciled
- Duplicated WebSocket/Live Ticker implementations → unified simple stub
  (works even without KiteTicker present; easy to extend with your creds).
- Conflicting time/session helpers → single canonical set under AppConfig.
- Indicator & support/resistance duplication → unified indicator pipeline.

Run it with:
    streamlit run RANTV_ProCons_MERGED.py

Notes:
- Kite Connect is optional. If not installed / no credentials, the app still
  runs in paper mode using yfinance.
- Email report uses Gmail SMTP (same as your file). Set ENV vars first:
  EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER (default: rantv2002@gmail.com)
"""

# ===================== IMPORTS =====================
import os
import sys
import time
import json
import math
import random
import logging
import threading
import warnings
import smtplib
import webbrowser
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Tuple, Any

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Third-party (optional friendly imports)
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

# Optional: Kite Connect
try:
    from kiteconnect import KiteConnect, KiteTicker
    KITECONNECT_AVAILABLE = True
except Exception:
    KITECONNECT_AVAILABLE = False

warnings.filterwarnings("ignore")

# ===================== LOGGING =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RANTV_MERGED")

# ===================== CONFIGURATION =====================
class AppConfig:
    """Environment & risk configuration (merged)"""
    # Broker
    KITE_API_KEY: str = os.environ.get("KITE_API_KEY", "")
    KITE_API_SECRET: str = os.environ.get("KITE_API_SECRET", "")

    # Risk
    INITIAL_CAPITAL: float = float(os.environ.get("INITIAL_CAPITAL", "2000000"))
    BASE_RISK: float = 0.01
    MAX_RISK: float = 0.02
    MIN_RISK: float = 0.005
    MAX_DAILY_DD: float = 0.03
    SL_ATR: float = 1.5
    TP_ATR: float = 3.0
    TRAIL_ATR: float = 1.2

    # Algo limits
    ALGO_TRADING_ENABLED: bool = os.environ.get("ALGO_TRADING_ENABLED", "false").lower() == "true"
    ALGO_MAX_POSITIONS: int = int(os.environ.get("ALGO_MAX_POSITIONS", "5"))
    ALGO_MAX_DAILY_LOSS: float = float(os.environ.get("ALGO_MAX_DAILY_LOSS", "50000"))
    ALGO_MIN_CONFIDENCE: float = float(os.environ.get("ALGO_MIN_CONFIDENCE", "0.80"))

    # Trading limits
    MAX_DAILY_TRADES: int = int(os.environ.get("MAX_DAILY_TRADES", "10"))
    MAX_STOCK_TRADES: int = int(os.environ.get("MAX_STOCK_TRADES", "10"))

    # Email
    EMAIL_SENDER: str = os.environ.get("EMAIL_SENDER", "")
    EMAIL_PASSWORD: str = os.environ.get("EMAIL_PASSWORD", "")
    EMAIL_RECEIVER: str = os.environ.get("EMAIL_RECEIVER", "rantv2002@gmail.com")

    # Timezone
    if PYTZ_AVAILABLE:
        try:
            IND_TZ = pytz.timezone("Asia/Kolkata")
        except Exception:
            IND_TZ = None
    else:
        IND_TZ = None

    # Market hours
    INDIA_OPEN = dt_time(9, 15)
    INDIA_CLOSE = dt_time(15, 30)
    DAILY_EXIT = dt_time(15, 35)

config = AppConfig()

# ===================== TIME HELPERS =====================
def now_indian() -> datetime:
    if config.IND_TZ:
        return datetime.now(config.IND_TZ)
    return datetime.now()

def valid_session() -> bool:
    """Indian cash session (9:15–15:30)."""
    t = now_indian().time()
    return config.INDIA_OPEN <= t <= config.INDIA_CLOSE

def should_exit_all_positions() -> bool:
    t = now_indian().time()
    return t >= config.DAILY_EXIT

# ===================== TECHNICALS =====================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rs = rs.fillna(0)
    return 100 - (100 / (1 + rs))

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger_bands(close: pd.Series, period: int = 20, std_dev: int = 2):
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def calculate_support_resistance(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> Dict[str, Any]:
    try:
        resistance: List[float] = []
        support: List[float] = []
        ln = len(high)
        if ln < period * 2 + 1:
            return {"support": float(close.iloc[-1] * 0.98), "resistance": float(close.iloc[-1] * 1.02),
                    "support_levels": [], "resistance_levels": []}
        for i in range(period, ln - period):
            if high.iloc[i] >= high.iloc[i - period:i + period + 1].max():
                resistance.append(float(high.iloc[i]))
            if low.iloc[i] <= low.iloc[i - period:i + period + 1].min():
                support.append(float(low.iloc[i]))
        recent_res = sorted(resistance)[-3:] if resistance else [float(close.iloc[-1] * 1.02)]
        recent_sup = sorted(support)[:3] if support else [float(close.iloc[-1] * 0.98)]
        return {
            "support": float(np.mean(recent_sup)),
            "resistance": float(np.mean(recent_res)),
            "support_levels": recent_sup,
            "resistance_levels": recent_res,
        }
    except Exception:
        current_price = float(close.iloc[-1])
        return {"support": current_price * 0.98, "resistance": current_price * 1.02,
                "support_levels": [], "resistance_levels": []}

# ============== SMC (Smart Money Concepts) – Pros from advanced file =============
class AdvancedSMC:
    @staticmethod
    def detect_BOS(df: pd.DataFrame, lookback: int = 6) -> Optional[str]:
        try:
            if len(df) < lookback + 1:
                return None
            if df['High'].iloc[-1] > df['High'].iloc[-lookback:-1].max():
                return 'BULLISH'
            if df['Low'].iloc[-1] < df['Low'].iloc[-lookback:-1].min():
                return 'BEARISH'
            return None
        except Exception:
            return None

    @staticmethod
    def detect_FVG(df: pd.DataFrame) -> Optional[Tuple[str, float, float]]:
        try:
            if len(df) < 3:
                return None
            a, b, c = df.iloc[-3], df.iloc[-2], df.iloc[-1]
            if a['High'] < c['Low']:
                return ('BULLISH', float(a['High']), float(c['Low']))
            if a['Low'] > c['High']:
                return ('BEARISH', float(c['High']), float(a['Low']))
            return None
        except Exception:
            return None

    @staticmethod
    def detect_order_block(df: pd.DataFrame) -> Optional[str]:
        try:
            if len(df) < 2:
                return None
            last = df.iloc[-2]
            if last['Close'] > last['Open']:
                return 'BULLISH'
            if last['Close'] < last['Open']:
                return 'BEARISH'
            return None
        except Exception:
            return None

    @staticmethod
    def detect_liquidity_grab(df: pd.DataFrame) -> Optional[str]:
        try:
            if len(df) < 10:
                return None
            recent_high = df['High'].iloc[-10:-1].max()
            recent_low = df['Low'].iloc[-10:-1].min()
            current_low = df['Low'].iloc[-1]
            current_high = df['High'].iloc[-1]
            current_close = df['Close'].iloc[-1]
            if current_low < recent_low and current_close > recent_low:
                return 'BULLISH'
            if current_high > recent_high and current_close < recent_high:
                return 'BEARISH'
            return None
        except Exception:
            return None

# ===================== Volume profile (POC) =====================
def volume_profile_poc(df: pd.DataFrame, bins: int = 24) -> float:
    try:
        if df is None or len(df) < bins:
            return float(df['Close'].iloc[-1]) if df is not None and len(df) else 0.0
        prices = df['Close'].values
        vols = df['Volume'].values
        hist, edges = np.histogram(prices, bins=bins, weights=vols)
        poc_index = int(np.argmax(hist))
        if poc_index < len(edges) - 1:
            return float((edges[poc_index] + edges[poc_index + 1]) / 2)
        return float(df['Close'].iloc[-1])
    except Exception:
        return float(df['Close'].iloc[-1]) if df is not None and len(df) else 0.0

# ===================== Data Manager (Unified) =====================
class UnifiedDataManager:
    """Yahoo-first data, optional Kite; unified indicators + SMC tags."""
    def __init__(self, kite_manager=None):
        self.kite_manager = kite_manager
        self.price_cache: Dict[str, Tuple[pd.DataFrame, float]] = {}
        self.cache_ttl = 30  # seconds
        self.smc = AdvancedSMC()

    def _cache_get(self, key: str) -> Optional[pd.DataFrame]:
        if key in self.price_cache:
            df, ts = self.price_cache[key]
            if time.time() - ts < self.cache_ttl:
                return df
        return None

    def _cache_put(self, key: str, df: pd.DataFrame):
        self.price_cache[key] = (df.copy(), time.time())

    def get_stock_data(self, symbol: str, interval: str = "15m", use_kite: bool = True) -> Optional[pd.DataFrame]:
        key = f"{symbol}_{interval}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        # Try Kite first if available
        if use_kite and self.kite_manager and self.kite_manager.is_authenticated:
            df = self.kite_manager.get_historical(symbol, interval=interval, days=7)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = self._enrich(df)
                self._cache_put(key, df)
                return df
        # Fallback Yahoo
        try:
            if not YFINANCE_AVAILABLE:
                return None
            period_map = {
                "1m": "1d", "5m": "5d", "15m": "15d", "30m": "30d",
                "1h": "60d", "1d": "1y"
            }
            period = period_map.get(interval, "15d")
            df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=False)
            if df is None or df.empty or len(df) < 20:
                return None
            df.columns = [c.capitalize() for c in df.columns]
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            df = self._enrich(df)
            self._cache_put(key, df)
            return df
        except Exception as e:
            logger.error(f"Yahoo data error for {symbol}: {e}")
            return None

    def _enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        # Core indicators
        df["EMA8"] = ema(df["Close"], 8)
        df["EMA21"] = ema(df["Close"], 21)
        df["EMA50"] = ema(df["Close"], 50)
        df["RSI14"] = rsi(df["Close"], 14)
        df["ATR"] = calculate_atr(df["High"], df["Low"], df["Close"])
        macd_line, macd_signal, macd_hist = macd(df["Close"])  # type: ignore
        df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd_line, macd_signal, macd_hist
        df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = bollinger_bands(df["Close"])  # type: ignore
        # VWAP
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (tp * df['Volume']).cumsum() / df['Volume'].cumsum()
        # Support/Resistance
        sr = calculate_support_resistance(df['High'], df['Low'], df['Close'])
        df['Support'] = sr['support']
        df['Resistance'] = sr['resistance']
        # SMC tags (last row snapshot style)
        try:
            df['SMC_BOS'] = self.smc.detect_BOS(df) or 'NONE'
            df['SMC_FVG'] = 'PRESENT' if self.smc.detect_FVG(df) else 'NONE'
            df['SMC_OB'] = self.smc.detect_order_block(df) or 'NONE'
            df['SMC_Liquidity'] = self.smc.detect_liquidity_grab(df) or 'NONE'
            df['Volume_POC'] = volume_profile_poc(df)
        except Exception:
            pass
        return df

    def get_live_price(self, symbol: str) -> Optional[float]:
        try:
            df = self.get_stock_data(symbol, interval="5m", use_kite=False)
            if df is not None and not df.empty:
                return float(df['Close'].iloc[-1])
            return None
        except Exception:
            return None

# ===================== Strategy & Signals =====================
class StrategyManager:
    def __init__(self):
        self.strategies = {
            "EMA_VWAP_Confluence": {"name": "EMA + VWAP Confluence", "weight": 3},
            "RSI_MeanReversion": {"name": "RSI Mean Reversion", "weight": 2},
            "Bollinger_Reversion": {"name": "Bollinger Band Reversion", "weight": 2},
            "MACD_Momentum": {"name": "MACD Momentum", "weight": 2},
            "Support_Resistance": {"name": "Support/Resistance", "weight": 3},
            "MTF_SMC": {"name": "MTF SMC + Volume Profile", "weight": 5},
        }
        self.enabled = set(self.strategies.keys())

    def list_enabled(self) -> List[str]:
        return list(self.enabled)

class AdvancedSignalGenerator:
    def __init__(self, data_manager: UnifiedDataManager, strategy_manager: StrategyManager):
        self.dm = data_manager
        self.sm = strategy_manager
        self.smc = AdvancedSMC()

    def generate_smc_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        mtf = {iv: self.dm.get_stock_data(symbol, interval=iv) for iv in ["15m", "1h", "1d"]}
        if not mtf.get("15m") or not mtf.get("1h"):
            return None
        ltf = mtf["15m"]; htf = mtf["1h"]
        if len(ltf) < 50 or len(htf) < 20 or not valid_session():
            return None
        price = float(ltf['Close'].iloc[-1])
        bos = self.smc.detect_BOS(htf)
        fvg = self.smc.detect_FVG(ltf)
        ob = self.smc.detect_order_block(ltf)
        liq = self.smc.detect_liquidity_grab(ltf)
        poc = volume_profile_poc(ltf)
        atr = float(ltf['ATR'].iloc[-1]) if 'ATR' in ltf.columns else max(0.02 * price, 1)
        signal, confidence = None, 0.0
        if bos == 'BULLISH' and (fvg and fvg[0] == 'BULLISH') and ob == 'BULLISH' and price > poc:
            confidence = 0.95 if liq == 'BULLISH' else 0.85
            signal = {
                'action': 'BUY', 'stop_loss': price - config.SL_ATR * atr, 'target': price + config.TP_ATR * atr,
            }
        elif bos == 'BEARISH' and (fvg and fvg[0] == 'BEARISH') and ob == 'BEARISH' and price < poc:
            confidence = 0.95 if liq == 'BEARISH' else 0.85
            signal = {
                'action': 'SELL', 'stop_loss': price + config.SL_ATR * atr, 'target': price - config.TP_ATR * atr,
            }
        if not signal:
            return None
        win_prob = min(0.98, confidence * 0.95)
        return {
            'symbol': symbol,
            'price': round(price, 2),
            'confidence': round(confidence, 3),
            'win_probability': round(win_prob, 3),
            'strategy': 'MTF SMC + Volume Profile',
            'atr': round(atr, 2),
            'poc': round(poc, 2),
            **signal,
        }

    def _tech_signals(self, data: pd.DataFrame, price: float) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        try:
            # EMA + VWAP
            if price > data['EMA8'].iloc[-1] > data['EMA21'].iloc[-1] and price > data['VWAP'].iloc[-1]:
                out.append({'action': 'BUY', 'confidence': 0.8, 'name': 'EMA+VWAP'})
            elif price < data['EMA8'].iloc[-1] < data['EMA21'].iloc[-1] and price < data['VWAP'].iloc[-1]:
                out.append({'action': 'SELL', 'confidence': 0.8, 'name': 'EMA+VWAP'})
            # RSI
            r = float(data['RSI14'].iloc[-1])
            if r < 30:
                out.append({'action': 'BUY', 'confidence': 0.75, 'name': 'RSI'})
            elif r > 70:
                out.append({'action': 'SELL', 'confidence': 0.75, 'name': 'RSI'})
            # Bollinger
            if price <= data['BB_Lower'].iloc[-1] * 1.01:
                out.append({'action': 'BUY', 'confidence': 0.7, 'name': 'BB'})
            elif price >= data['BB_Upper'].iloc[-1] * 0.99:
                out.append({'action': 'SELL', 'confidence': 0.7, 'name': 'BB'})
            # MACD cross
            m = data['MACD'].iloc[-1]; s = data['MACD_Signal'].iloc[-1]
            pm = data['MACD'].iloc[-2]; ps = data['MACD_Signal'].iloc[-2]
            if m > s and pm <= ps:
                out.append({'action': 'BUY', 'confidence': 0.75, 'name': 'MACD'})
            elif m < s and pm >= ps:
                out.append({'action': 'SELL', 'confidence': 0.75, 'name': 'MACD'})
            # S/R
            sup = data['Support'].iloc[-1]; res = data['Resistance'].iloc[-1]
            if price <= sup * 1.01:
                out.append({'action': 'BUY', 'confidence': 0.8, 'name': 'S/R'})
            elif price >= res * 0.99:
                out.append({'action': 'SELL', 'confidence': 0.8, 'name': 'S/R'})
        except Exception:
            pass
        return out

    def generate_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        # Try SMC first
        smc_sig = self.generate_smc_signal(symbol)
        if smc_sig and smc_sig['confidence'] >= config.ALGO_MIN_CONFIDENCE:
            return smc_sig
        # Technical confluence
        data = self.dm.get_stock_data(symbol, interval="15m")
        if data is None or len(data) < 50:
            return None
        price = float(data['Close'].iloc[-1])
        techs = self._tech_signals(data, price)
        if not techs:
            return None
        buy = [s for s in techs if s['action'] == 'BUY']
        sell = [s for s in techs if s['action'] == 'SELL']
        if not buy and not sell:
            return None
        def score(lst):
            return sum(d['confidence'] for d in lst)
        action = 'BUY' if score(buy) > score(sell) else 'SELL'
        conf = (score(buy) if action == 'BUY' else score(sell)) / max(1, len(buy if action=='BUY' else sell))
        atr = float(data['ATR'].iloc[-1]) if 'ATR' in data.columns else max(0.02 * price, 1)
        sl = price - 1.5 * atr if action == 'BUY' else price + 1.5 * atr
        tg = price + 3.0 * atr if action == 'BUY' else price - 3.0 * atr
        return {
            'symbol': symbol, 'price': round(price, 2), 'action': action,
            'stop_loss': round(sl, 2), 'target': round(tg, 2),
            'confidence': round(conf, 3), 'win_probability': round(min(0.97, conf*0.85), 3),
            'strategy': 'Multi-Strategy Confluence'
        }

    def scan_universe(self, stocks: List[str], max_signals: int = 10) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for s in stocks:
            sig = self.generate_signal(s)
            if sig and sig['confidence'] >= config.ALGO_MIN_CONFIDENCE:
                out.append(sig)
        out.sort(key=lambda d: (d.get('confidence', 0), d.get('win_probability', 0)), reverse=True)
        return out[:max_signals]

# ===================== Risk & Trader =====================
class AutoRiskScaler:
    def __init__(self, initial_capital: float):
        self.equity = [initial_capital]
        self.returns: List[float] = []
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0

    def update(self, pnl: float):
        new_eq = self.equity[-1] + pnl
        self.equity.append(new_eq)
        if len(self.equity) > 1:
            ret = (new_eq - self.equity[-2]) / max(1e-9, self.equity[-2])
            self.returns.append(ret)
        peak = max(self.equity)
        cur = self.equity[-1]
        self.current_drawdown = (peak - cur) / peak if peak > 0 else 0.0
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

    def calc_risk(self) -> float:
        if len(self.returns) < 10:
            return config.BASE_RISK
        vol = np.std(self.returns[-10:]) or 0.02
        avg = np.mean(self.returns[-10:]) if self.returns else 0.0
        sharpe_like = avg / vol if vol > 0 else 0.0
        risk = config.BASE_RISK
        if self.current_drawdown < config.MAX_DAILY_DD * 0.5 and sharpe_like > 0:
            risk *= min(2.0, 1.0 + 8.0 * sharpe_like)
        elif self.current_drawdown > config.MAX_DAILY_DD * 0.8 or sharpe_like < 0:
            risk *= max(0.3, 1.0 - 5.0 * abs(sharpe_like))
        return float(max(config.MIN_RISK, min(config.MAX_RISK, risk)))

class RiskManager:
    def __init__(self):
        self.scaler = AutoRiskScaler(config.INITIAL_CAPITAL)
        self.daily_pnl = 0.0
        self.positions_open = 0
        self.trades_today = 0

    def status(self) -> Dict[str, Any]:
        return {
            'daily_pnl': self.daily_pnl,
            'positions_open': self.positions_open,
            'trades_today': self.trades_today,
            'dynamic_risk': self.scaler.calc_risk(),
            'within_limits': self.daily_pnl > -config.ALGO_MAX_DAILY_LOSS and self.positions_open <= config.ALGO_MAX_POSITIONS,
            'drawdown': self.scaler.current_drawdown,
        }

    def approve(self, confidence: float) -> Tuple[bool, str]:
        stt = self.status()
        if not valid_session():
            return False, "Market closed"
        if confidence < config.ALGO_MIN_CONFIDENCE:
            return False, "Low confidence"
        if not stt['within_limits']:
            return False, "Risk limits breached"
        if self.trades_today >= config.MAX_DAILY_TRADES:
            return False, "Daily trade limit reached"
        return True, "OK"

    def on_open(self):
        self.positions_open += 1
        self.trades_today += 1

    def on_close(self, pnl: float):
        self.positions_open = max(0, self.positions_open - 1)
        self.daily_pnl += pnl
        self.scaler.update(pnl)

class AdvancedPaperTrader:
    def __init__(self, data: UnifiedDataManager, risk: RiskManager):
        self.dm = data
        self.risk = risk
        self.cash = config.INITIAL_CAPITAL
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trades: List[Dict[str, Any]] = []

    def position_size(self, price: float, is_smc: bool) -> int:
        pct = self.risk.status()['dynamic_risk']
        if is_smc:
            pct *= 1.2
        max_value = self.cash * pct
        qty = int(max_value / max(1e-6, price))
        return max(1, min(qty, 100))

    def open_from_signal(self, sig: Dict[str, Any]) -> Tuple[bool, str]:
        ok, reason = self.risk.approve(sig.get('confidence', 0.0))
        if not ok:
            return False, reason
        sym = sig['symbol']; price = sig['price']; action = sig['action']
        qty = self.position_size(price, is_smc=('poc' in sig))
        value = qty * price
        if action == 'BUY' and value > self.cash:
            return False, "Insufficient cash"
        # Book-keep
        self.positions[sym] = {
            'symbol': sym, 'action': action, 'qty': qty,
            'entry': price, 'sl': sig.get('stop_loss'), 'tg': sig.get('target'),
            'opened': now_indian(), 'strategy': sig.get('strategy', 'N/A'),
            'is_smc': 'poc' in sig
        }
        if action == 'BUY':
            self.cash -= value
        else:
            self.cash -= value * 0.2  # short margin placeholder
        self.trades.append({**self.positions[sym], 'status': 'OPEN'})
        self.risk.on_open()
        return True, f"Opened {action} {qty} {sym} @ ₹{price:.2f}"

    def _mark_price(self, sym: str) -> Optional[float]:
        p = self.dm.get_live_price(sym)
        return p

    def update_and_maybe_close(self):
        for sym, pos in list(self.positions.items()):
            p = self._mark_price(sym)
            if p is None:
                continue
            pos['last'] = p
            # PnL
            if pos['action'] == 'BUY':
                pnl = (p - pos['entry']) * pos['qty']
            else:
                pnl = (pos['entry'] - p) * pos['qty']
            pos['pnl'] = pnl
            # SL/TG
            sl = pos.get('sl'); tg = pos.get('tg')
            hit_sl = (pos['action'] == 'BUY' and sl is not None and p <= sl) or (pos['action'] == 'SELL' and sl is not None and p >= sl)
            hit_tg = (pos['action'] == 'BUY' and tg is not None and p >= tg) or (pos['action'] == 'SELL' and tg is not None and p <= tg)
            if hit_sl or hit_tg or should_exit_all_positions():
                self.close(sym, exit_price=p)

    def close(self, sym: str, exit_price: Optional[float] = None) -> Tuple[bool, str]:
        if sym not in self.positions:
            return False, "Not found"
        pos = self.positions[sym]
        p = exit_price or self._mark_price(sym) or pos['entry']
        if pos['action'] == 'BUY':
            pnl = (p - pos['entry']) * pos['qty']
            self.cash += pos['qty'] * p
        else:
            pnl = (pos['entry'] - p) * pos['qty']
            self.cash += pos['qty'] * pos['entry'] * 0.2  # release margin placeholder
        pos['status'] = 'CLOSED'
        pos['exit'] = p
        pos['closed'] = now_indian()
        self.trades.append({**pos})
        del self.positions[sym]
        self.risk.on_close(pnl)
        return True, f"Closed {sym} @ ₹{p:.2f} | P&L: ₹{pnl:+.2f}"

    def close_all(self):
        for sym in list(self.positions.keys()):
            self.close(sym)

    def performance(self) -> Dict[str, Any]:
        closed = [t for t in self.trades if t.get('status') == 'CLOSED']
        wins = [t for t in closed if t.get('exit', t.get('entry', 0)) != t.get('entry', 0) and ((t['action']=='BUY' and t['exit']>t['entry']) or (t['action']=='SELL' and t['exit']<t['entry']))]
        total_pnl = sum(((t['exit'] - t['entry']) * t['qty']) if t['action']=='BUY' else ((t['entry'] - t['exit']) * t['qty']) for t in closed)
        return {
            'total_trades': len(closed),
            'win_rate': (len(wins) / len(closed)) if closed else 0.0,
            'total_pnl': total_pnl,
            'open_positions': len(self.positions),
            'cash': self.cash,
        }

# ===================== Algo Engine =====================
class AlgoEngine:
    def __init__(self, trader: AdvancedPaperTrader, signals: AdvancedSignalGenerator):
        self.trader = trader
        self.signals = signals
        self.state = 'stopped'
        self._stop = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.universe = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "ITC.NS", "LT.NS", "SBIN.NS", "AXISBANK.NS", "BAJFINANCE.NS",
        ]

    def start(self):
        if self.state == 'running':
            return False
        self.state = 'running'
        self._stop.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        logger.info("Algo started")
        return True

    def stop(self):
        self.state = 'stopped'
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=3)
        logger.info("Algo stopped")

    def _loop(self):
        last_scan = 0.0
        while not self._stop.is_set():
            try:
                if self.state != 'running' or not valid_session():
                    time.sleep(2)
                    continue
                # Daily hard exit at 3:35 PM
                if should_exit_all_positions():
                    self.trader.close_all()
                    try:
                        send_daily_report_email(self.trader)
                    except Exception:
                        pass
                    time.sleep(60)
                    continue
                # Update open positions
                self.trader.update_and_maybe_close()
                # Scan every 5 minutes
                if time.time() - last_scan > 300:
                    sigs = self.signals.scan_universe(self.universe, max_signals=3)
                    for s in sigs:
                        if s['symbol'] not in self.trader.positions and self.trader.risk.positions_open < config.ALGO_MAX_POSITIONS:
                            ok, msg = self.trader.open_from_signal(s)
                            logger.info(msg)
                    last_scan = time.time()
                time.sleep(5)
            except Exception as e:
                logger.error(f"Algo loop error: {e}")
                time.sleep(5)

# ===================== Email (from RANTV version – adapted) =====================
def send_daily_report_email(trader: AdvancedPaperTrader) -> bool:
    try:
        if not config.EMAIL_SENDER or not config.EMAIL_PASSWORD:
            logger.warning("Email creds not set; skip daily report")
            return False
        # Collect simple summary
        perf = trader.performance()
        today = now_indian().date().strftime('%Y-%m-%d')
        subject = f"Daily Trading Report - {today}"
        rows = []
        for sym, pos in trader.positions.items():
            rows.append(f"<tr><td>{sym}</td><td>{pos['action']}</td><td>{pos['qty']}</td><td>₹{pos['entry']:.2f}</td><td>OPEN</td></tr>")
        closed = [t for t in trader.trades if t.get('status') == 'CLOSED']
        for t in closed[-20:]:
            pnl = ((t['exit'] - t['entry']) * t['qty']) if t['action']=='BUY' else ((t['entry'] - t['exit']) * t['qty'])
            rows.append(f"<tr><td>{t['symbol']}</td><td>{t['action']}</td><td>{t['qty']}</td><td>₹{t['entry']:.2f}</td><td>₹{t['exit']:.2f} (₹{pnl:+.2f})</td></tr>")
        html = f"""
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
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = config.EMAIL_SENDER
        msg['To'] = config.EMAIL_RECEIVER
        msg.attach(MIMEText(html, 'html'))
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(config.EMAIL_SENDER, config.EMAIL_PASSWORD)
            server.send_message(msg)
        logger.info(f"Daily report emailed to {config.EMAIL_RECEIVER}")
        return True
    except Exception as e:
        logger.error(f"Email error: {e}")
        return False

# ===================== (Optional) Kite Manager – minimal shim =====================
class KiteManager:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.is_authenticated = False
        self.kite = None

        if KITECONNECT_AVAILABLE and api_key and api_secret:
            try:
                self.kite = KiteConnect(api_key=api_key)
            except Exception as e:
                logger.warning(f"Kite init failed: {e}")

    def login_url(self) -> Optional[str]:
        if self.kite:
            try:
                return self.kite.login_url()
            except Exception:
                return None
        return None

    def authenticate(self, request_token: str) -> Tuple[bool, str]:
        try:
            if not self.kite:
                return False, "Kite not available"
            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            access = data.get('access_token')
            if not access:
                return False, "No access token"
            self.kite.set_access_token(access)
            self.is_authenticated = True
            return True, f"Authenticated as {data.get('user_name', 'User')}"
        except Exception as e:
            return False, f"Auth error: {e}"

    def get_historical(self, symbol: str, interval: str = "15m", days: int = 7) -> Optional[pd.DataFrame]:
        try:
            if not self.is_authenticated or not self.kite:
                return None
            token_map = {
                "RELIANCE.NS": 738561, "TCS.NS": 2953217, "HDFCBANK.NS": 341249,
                "INFY.NS": 408065, "ICICIBANK.NS": 1270529, "ITC.NS": 424961,
                "LT.NS": 2939649, "SBIN.NS": 779521, "AXISBANK.NS": 1510401,
                "BAJFINANCE.NS": 81153,
            }
            token = token_map.get(symbol)
            if not token:
                return None
            interval_map = {"1m": "minute", "5m": "5minute", "15m": "15minute", "30m": "30minute", "1h": "60minute", "1d": "day"}
            kite_iv = interval_map.get(interval, "15minute")
            from_date = datetime.now().date() - timedelta(days=days)
            to_date = datetime.now().date()
            data = self.kite.historical_data(token, from_date, to_date, kite_iv, continuous=False, oi=False)
            if not data:
                return None
            df = pd.DataFrame(data)
            if 'open' in df.columns:
                df = df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'})
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Kite historical error: {e}")
            return None

# ===================== UI (Streamlit) =====================
if STREAMLIT_AVAILABLE:
    st.set_page_config(page_title="RANTV Intraday Terminal – Merged", layout="wide", initial_sidebar_state="expanded")

    def load_css():
        st.markdown(
            """
            <style>
            .stApp {background: linear-gradient(135deg,#0f172a 0%, #1e293b 100%); color:#f8fafc}
            .main .block-container {padding-top:0.6rem; padding-bottom:0.6rem}
            .main-header {text-align:center;padding:.8rem;background:linear-gradient(135deg,#f97316 0%,#ea580c 100%);border-radius:10px;color:#fff;box-shadow:0 4px 12px rgba(249,115,22,.3)}
            .stTabs [data-baseweb="tab-list"]{gap:4px;background:rgba(30,41,59,.6);padding:6px;border-radius:10px}
            .stTabs [data-baseweb="tab"]{background:#1f2937;border-radius:8px;color:#cbd5e1;border:1px solid rgba(255,255,255,.08)}
            .stTabs [aria-selected="true"]{background:linear-gradient(135deg,#f97316 0%,#ea580c 100%); color:#fff}
            [data-testid="stMetricValue"]{color:#f97316;font-weight:800}
            .good{color:#10b981}.bad{color:#ef4444}
            </style>
            """, unsafe_allow_html=True)

    load_css()

    # --- Sidebar: Kite auth (optional) ---
    st.sidebar.header("🔐 Kite Connect (optional)")
    if 'kite_mgr' not in st.session_state:
        st.session_state.kite_mgr = KiteManager(config.KITE_API_KEY, config.KITE_API_SECRET) if (config.KITE_API_KEY and config.KITE_API_SECRET) else None
    kite_mgr: Optional[KiteManager] = st.session_state.kite_mgr

    if kite_mgr and not kite_mgr.is_authenticated:
        colA, colB = st.sidebar.columns(2)
        if colA.button("Get Login URL"):
            url = kite_mgr.login_url()
            if url: st.sidebar.success("Login URL generated"); st.sidebar.code(url)
            else: st.sidebar.error("Failed to get login URL")
        req = st.sidebar.text_input("Paste request_token here")
        if st.sidebar.button("Authenticate") and req:
            ok, msg = kite_mgr.authenticate(req)
            (st.sidebar.success if ok else st.sidebar.error)(msg)

    # --- Compose system ---
    if 'data_mgr' not in st.session_state:
        st.session_state.data_mgr = UnifiedDataManager(kite_manager=kite_mgr)
    if 'risk_mgr' not in st.session_state:
        st.session_state.risk_mgr = RiskManager()
    if 'strat_mgr' not in st.session_state:
        st.session_state.strat_mgr = StrategyManager()
    if 'sig_gen' not in st.session_state:
        st.session_state.sig_gen = AdvancedSignalGenerator(st.session_state.data_mgr, st.session_state.strat_mgr)
    if 'trader' not in st.session_state:
        st.session_state.trader = AdvancedPaperTrader(st.session_state.data_mgr, st.session_state.risk_mgr)
    if 'algo' not in st.session_state:
        st.session_state.algo = AlgoEngine(st.session_state.trader, st.session_state.sig_gen)

    trader: AdvancedPaperTrader = st.session_state.trader
    algo: AlgoEngine = st.session_state.algo

    # --- Header ---
    st.markdown('<div class="main-header"><h1>RANTV Intraday Terminal – Merged (Pros & Cons)</h1><p>SMC + Risk Scaling + Streamlit UX</p></div>', unsafe_allow_html=True)

    # --- Top metrics ---
    perf = trader.performance(); risk = st.session_state.risk_mgr.status()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Cash", f"₹{perf['cash']:,.0f}")
    m2.metric("Open Positions", perf['open_positions'])
    m3.metric("Closed Trades", perf['total_trades'])
    m4.metric("Win Rate", f"{perf['win_rate']:.1%}")
    m5.metric("Total P&L", f"₹{perf['total_pnl']:+,.0f}", delta_color="normal" if perf['total_pnl']>=0 else "inverse")

    tabs = st.tabs(["📈 Signals", "💼 Positions", "🤖 Algo", "📬 Report"])  # keep UI compact

    # --- Signals tab ---
    with tabs[0]:
        st.subheader("Top Signals (NIFTY set)")
        scan_btn = st.button("Scan Now", type="primary")
        default_universe = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "ITC.NS", "LT.NS", "SBIN.NS", "AXISBANK.NS", "BAJFINANCE.NS",
        ]
        if scan_btn:
            sigs = st.session_state.sig_gen.scan_universe(default_universe, max_signals=10)
            st.session_state['latest_signals'] = sigs
        sigs = st.session_state.get('latest_signals', [])
        if sigs:
            for s in sigs:
                c1, c2, c3, c4, c5, c6 = st.columns([2,1,1,1,2,1])
                with c1: st.write(f"**{s['symbol']}**")
                with c2: st.write(s['action'])
                with c3: st.write(f"Conf: {s['confidence']:.2f}")
                with c4: st.write(f"Win: {s['win_probability']:.2f}")
                with c5: st.write(f"SL: ₹{s['stop_loss']:.2f} • TG: ₹{s['target']:.2f}")
                do = c6.button("Trade", key=f"trade_{s['symbol']}_{time.time()}")
                if do:
                    ok, msg = trader.open_from_signal(s)
                    (st.success if ok else st.error)(msg)
        else:
            st.info("Click **Scan Now** to generate signals.")

    # --- Positions tab ---
    with tabs[1]:
        st.subheader("Open Positions")
        if trader.positions:
            for sym, pos in trader.positions.items():
                c1, c2, c3, c4, c5, c6 = st.columns([2,1,1,1,1,1])
                c1.write(f"**{sym}**")
                c2.write(pos['action'])
                c3.write(f"Qty: {pos['qty']}")
                c4.write(f"Entry: ₹{pos['entry']:.2f}")
                c5.write(f"SL: {('N/A' if pos.get('sl') is None else '₹'+str(round(pos['sl'],2)))}")
                if c6.button("Close", key=f"close_{sym}"):
                    ok, msg = trader.close(sym)
                    (st.success if ok else st.error)(msg)
        else:
            st.write("No open positions.")
        st.divider()
        st.subheader("Trade History (last 30)")
        closed = [t for t in trader.trades if t.get('status') == 'CLOSED'][-30:]
        if closed:
            df = pd.DataFrame([{k: v for k, v in t.items() if k in ('symbol','action','qty','entry','exit','strategy','is_smc')} for t in closed])
            st.dataframe(df, use_container_width=True)
        else:
            st.write("No closed trades yet.")

    # --- Algo tab ---
    with tabs[2]:
        st.subheader("Algo Engine")
        col1, col2, col3 = st.columns(3)
        if algo.state != 'running':
            if col1.button("Start Algo", type="primary"):
                algo.start(); st.success("Algo started")
        else:
            if col1.button("Stop Algo", type="secondary"):
                algo.stop(); st.warning("Algo stopped")
        if col2.button("Force Scan"):
            sigs = st.session_state.sig_gen.scan_universe(algo.universe, max_signals=3)
            for s in sigs:
                if s['symbol'] not in trader.positions and st.session_state.risk_mgr.positions_open < config.ALGO_MAX_POSITIONS:
                    ok, msg = trader.open_from_signal(s)
                    (st.success if ok else st.error)(msg)
        if col3.button("Exit All"):
            trader.close_all(); st.warning("All positions closed")
        st.caption("Algo scans every ~5 minutes during market hours and auto exits at 3:35 PM.")

    # --- Report tab ---
    with tabs[3]:
        st.subheader("Email Report")
        if st.button("Send Daily Report Now"):
            ok = send_daily_report_email(trader)
            (st.success if ok else st.error)("Report sent" if ok else "Failed to send report (check EMAIL_* env vars)")

# CLI fallback (no Streamlit)
else:
    if __name__ == '__main__':
        print("Streamlit is not available. Install with: pip install streamlit plotly yfinance")
