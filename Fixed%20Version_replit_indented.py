"""
RANTV TERMINAL PRO - ENHANCED ALGO TRADING PLATFORM
Complete Trading System with Modular Architecture
Version 2.0 - Single File Implementation
"""

# ===================== IMPORTS =====================
import os
import sys
import time
import json
import logging
import threading
import warnings
from datetime import datetime, timedelta, time as dt_time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import math
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Data & ML Libraries
import pandas as pd
import numpy as np
import yfinance as yf

# Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# UI Framework
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Timezone
import pytz

# Suppress warnings
warnings.filterwarnings("ignore")

# ===================== CONFIGURATION =====================
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Timezone
IND_TZ = pytz.timezone("Asia/Kolkata")

# Environment Configuration
class AppConfig:
    """Application configuration"""
    # API Configuration
    KITE_API_KEY = os.environ.get("KITE_API_KEY", "")
    KITE_API_SECRET = os.environ.get("KITE_API_SECRET", "")
    
    # Algo Configuration
    ALGO_ENABLED = os.environ.get("ALGO_TRADING_ENABLED", "false").lower() == "true"
    ALGO_MAX_POSITIONS = int(os.environ.get("ALGO_MAX_POSITIONS", "5"))
    ALGO_MAX_DAILY_LOSS = float(os.environ.get("ALGO_MAX_DAILY_LOSS", "50000"))
    ALGO_MIN_CONFIDENCE = float(os.environ.get("ALGO_MIN_CONFIDENCE", "0.80"))
    
    # Email Configuration
    EMAIL_SENDER = os.environ.get("EMAIL_SENDER", "")
    EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")
    EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER", "rantv2002@gmail.com")
    
    # Trading Configuration
    INITIAL_CAPITAL = 2_000_000.0
    TRADE_ALLOCATION = 0.15
    MAX_DAILY_TRADES = 10
    MAX_STOCK_TRADES = 10
    MAX_AUTO_TRADES = 10
    SIGNAL_REFRESH_MS = 120000
    PRICE_REFRESH_MS = 100000
    
    # Database
    DATABASE_URL = "sqlite:///trading_journal.db"
    
    @classmethod
    def load_from_env(cls):
        """Load configuration from environment"""
        return cls()

# Load config
config = AppConfig.load_from_env()

# Trading Constants
class TradingConstants:
    """Trading constants"""
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
    
    # Trading Strategies
    TRADING_STRATEGIES = {
        "SMC_Liquidity_FVG": {"name": "Smart Money Concept", "weight": 4, "type": "BOTH"},
        "EMA_VWAP_Confluence": {"name": "EMA + VWAP", "weight": 3, "type": "BUY"},
        "RSI_MeanReversion": {"name": "RSI Mean Reversion", "weight": 2, "type": "BUY"},
        "Bollinger_Reversion": {"name": "Bollinger Band", "weight": 2, "type": "BUY"},
        "MACD_Momentum": {"name": "MACD Momentum", "weight": 2, "type": "BUY"},
        "Support_Resistance": {"name": "S/R Breakout", "weight": 3, "type": "BUY"},
    }
    
    # Market Hours
    MARKET_OPEN = "09:15"
    MARKET_CLOSE = "15:30"
    PEAK_START = "09:30"
    PEAK_END = "14:30"
    DAILY_EXIT = "15:35"

# ===================== UTILITY FUNCTIONS =====================
def now_indian():
    """Get current time in Indian timezone"""
    return datetime.now(IND_TZ)

def market_open():
    """Check if market is open"""
    n = now_indian()
    try:
        open_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(9, 15)))
        close_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(15, 30)))
        return open_time <= n <= close_time
    except Exception:
        return False

def is_peak_market_hours():
    """Check if current time is during peak market hours"""
    n = now_indian()
    try:
        peak_start = IND_TZ.localize(datetime.combine(n.date(), dt_time(9, 30)))
        peak_end = IND_TZ.localize(datetime.combine(n.date(), dt_time(14, 30)))
        return peak_start <= n <= peak_end
    except Exception:
        return True

def should_exit_all_positions():
    """Check if it's time to exit all positions (3:35 PM)"""
    n = now_indian()
    try:
        exit_time = IND_TZ.localize(datetime.combine(n.date(), dt_time(15, 35)))
        return n >= exit_time
    except Exception:
        return False

def ema(series, span):
    """Calculate Exponential Moving Average"""
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rs = rs.fillna(0)
    return 100 - (100 / (1 + rs))

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger_bands(close, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def calculate_support_resistance(high, low, close, period=20):
    """Calculate support and resistance levels"""
    try:
        resistance = []
        support = []
        ln = len(high)
        if ln < period * 2 + 1:
            return {"support": float(close.iloc[-1] * 0.98), "resistance": float(close.iloc[-1] * 1.02)}
        
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
            "resistance_levels": recent_res
        }
    except Exception:
        current_price = float(close.iloc[-1])
        return {
            "support": current_price * 0.98,
            "resistance": current_price * 1.02,
            "support_levels": [],
            "resistance_levels": []
        }

# ===================== DATA MANAGER =====================
class DataManager:
    """Enhanced data manager with caching"""
    
    def __init__(self):
        self.price_cache = {}
        self.signal_cache = {}
        self.cache_timeout = 30  # seconds
        
    def get_stock_data(self, symbol, interval="15m"):
        """Get stock data with technical indicators"""
        cache_key = f"{symbol}_{interval}_{datetime.now().strftime('%Y%m%d_%H')}"
        
        # Check cache
        if cache_key in self.price_cache:
            cached_data, timestamp = self.price_cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_timeout:
                return cached_data
        
        try:
            # Determine period based on interval
            period_map = {
                "1m": "1d",
                "5m": "5d",
                "15m": "15d",
                "30m": "30d",
                "1h": "60d"
            }
            
            period = period_map.get(interval, "15d")
            
            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=False)
            
            if df is None or df.empty or len(df) < 20:
                return None
            
            # Clean columns
            df.columns = [col.capitalize() for col in df.columns]
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            
            # Calculate indicators
            df["EMA8"] = ema(df["Close"], 8)
            df["EMA21"] = ema(df["Close"], 21)
            df["EMA50"] = ema(df["Close"], 50)
            df["RSI14"] = rsi(df["Close"], 14)
            df["ATR"] = calculate_atr(df["High"], df["Low"], df["Close"])
            df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = macd(df["Close"])
            df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = bollinger_bands(df["Close"])
            
            # VWAP calculation
            df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['TP_Volume'] = df['Typical_Price'] * df['Volume']
            df['VWAP'] = df['TP_Volume'].cumsum() / df['Volume'].cumsum()
            
            # Support/Resistance
            sr = calculate_support_resistance(df["High"], df["Low"], df["Close"])
            df["Support"] = sr["support"]
            df["Resistance"] = sr["resistance"]
            
            # Cache the result
            self.price_cache[cache_key] = (df.copy(), datetime.now())
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_live_price(self, symbol):
        """Get live price for a symbol"""
        try:
            data = self.get_stock_data(symbol, "5m")
            if data is not None and not data.empty:
                return float(data["Close"].iloc[-1])
            return None
        except Exception:
            return None
    
    def clear_cache(self):
        """Clear all caches"""
        self.price_cache.clear()
        self.signal_cache.clear()

# ===================== RISK MANAGER =====================
class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, config):
        self.config = config
        self.daily_stats = {
            'total_pnl': 0.0,
            'trades_today': 0,
            'positions_opened': 0,
            'max_daily_loss': config.ALGO_MAX_DAILY_LOSS
        }
        self.position_limits = {
            'max_positions': config.ALGO_MAX_POSITIONS,
            'max_trades_per_day': config.MAX_DAILY_TRADES
        }
        
    def check_trade(self, symbol, action, quantity, price, confidence=0.5):
        """Check if trade meets risk criteria"""
        checks = {
            'market_open': market_open(),
            'daily_loss_limit': self.daily_stats['total_pnl'] > -self.daily_stats['max_daily_loss'],
            'position_limit': self.daily_stats['positions_opened'] < self.position_limits['max_positions'],
            'trade_limit': self.daily_stats['trades_today'] < self.position_limits['max_trades_per_day'],
            'confidence_threshold': confidence >= config.ALGO_MIN_CONFIDENCE,
            'sufficient_capital': True  # Will be checked by trader
        }
        
        all_passed = all(checks.values())
        
        return {
            'approved': all_passed,
            'checks': checks,
            'reason': None if all_passed else "; ".join([k for k, v in checks.items() if not v])
        }
    
    def update_trade(self, pnl):
        """Update risk metrics after trade"""
        self.daily_stats['total_pnl'] += pnl
        self.daily_stats['trades_today'] += 1
    
    def update_position(self, opened=True):
        """Update position count"""
        if opened:
            self.daily_stats['positions_opened'] += 1
        else:
            self.daily_stats['positions_opened'] = max(0, self.daily_stats['positions_opened'] - 1)
    
    def reset_daily(self):
        """Reset daily statistics"""
        self.daily_stats.update({
            'total_pnl': 0.0,
            'trades_today': 0,
            'positions_opened': 0
        })
    
    def get_status(self):
        """Get risk manager status"""
        return {
            'daily_pnl': self.daily_stats['total_pnl'],
            'trades_today': self.daily_stats['trades_today'],
            'positions_open': self.daily_stats['positions_opened'],
            'daily_limit_used': abs(self.daily_stats['total_pnl']) / self.daily_stats['max_daily_loss'],
            'within_limits': self.daily_stats['total_pnl'] > -self.daily_stats['max_daily_loss']
        }

# ===================== STRATEGY MANAGER =====================
class StrategyManager:
    """Manage trading strategies"""
    
    def __init__(self):
        self.strategies = {}
        self.enabled_strategies = set()
        self.load_strategies()
    
    def load_strategies(self):
        """Load all trading strategies"""
        for strategy_id, config in TradingConstants.TRADING_STRATEGIES.items():
            self.strategies[strategy_id] = {
                'id': strategy_id,
                'name': config['name'],
                'weight': config['weight'],
                'type': config['type'],
                'enabled': True,
                'performance': {'signals': 0, 'trades': 0, 'wins': 0, 'pnl': 0.0}
            }
            self.enabled_strategies.add(strategy_id)
    
    def enable_strategy(self, strategy_id):
        """Enable a strategy"""
        if strategy_id in self.strategies:
            self.strategies[strategy_id]['enabled'] = True
            self.enabled_strategies.add(strategy_id)
    
    def disable_strategy(self, strategy_id):
        """Disable a strategy"""
        if strategy_id in self.strategies:
            self.strategies[strategy_id]['enabled'] = False
            self.enabled_strategies.discard(strategy_id)
    
    def get_all_strategies(self):
        """Get all strategies"""
        return self.strategies
    
    def get_enabled_strategies(self):
        """Get enabled strategies"""
        return [self.strategies[sid] for sid in self.enabled_strategies]
    
    def update_performance(self, strategy_id, pnl, win=True):
        """Update strategy performance"""
        if strategy_id in self.strategies:
            self.strategies[strategy_id]['performance']['trades'] += 1
            self.strategies[strategy_id]['performance']['pnl'] += pnl
            if win:
                self.strategies[strategy_id]['performance']['wins'] += 1

# ===================== SIGNAL GENERATOR =====================
class SignalGenerator:
    """Generate trading signals"""
    
    def __init__(self, data_manager, strategy_manager):
        self.data_manager = data_manager
        self.strategy_manager = strategy_manager
        self.signals_generated = 0
        self.last_scan_time = None
        
    def generate_signal(self, symbol):
        """Generate signal for a single symbol"""
        data = self.data_manager.get_stock_data(symbol, "15m")
        
        if data is None or len(data) < 50:
            return None
        
        try:
            current_price = float(data['Close'].iloc[-1])
            signals = []
            
            # Check each enabled strategy
            for strategy in self.strategy_manager.get_enabled_strategies():
                strategy_id = strategy['id']
                
                if strategy_id == "SMC_Liquidity_FVG":
                    signal = self._smc_strategy(data, current_price)
                elif strategy_id == "EMA_VWAP_Confluence":
                    signal = self._ema_vwap_strategy(data, current_price)
                elif strategy_id == "RSI_MeanReversion":
                    signal = self._rsi_strategy(data, current_price)
                elif strategy_id == "Bollinger_Reversion":
                    signal = self._bollinger_strategy(data, current_price)
                elif strategy_id == "MACD_Momentum":
                    signal = self._macd_strategy(data, current_price)
                elif strategy_id == "Support_Resistance":
                    signal = self._support_resistance_strategy(data, current_price)
                else:
                    signal = None
                
                if signal:
                    signal['strategy'] = strategy['name']
                    signal['strategy_id'] = strategy_id
                    signal['weight'] = strategy['weight']
                    signals.append(signal)
            
            if not signals:
                return None
            
            # Combine signals
            buy_signals = [s for s in signals if s['action'] == 'BUY']
            sell_signals = [s for s in signals if s['action'] == 'SELL']
            
            if not buy_signals and not sell_signals:
                return None
            
            # Determine final action based on weighted signals
            buy_score = sum(s['confidence'] * s['weight'] for s in buy_signals)
            sell_score = sum(s['confidence'] * s['weight'] for s in sell_signals)
            
            if buy_score > sell_score and buy_score > 0:
                action = 'BUY'
                confidence = buy_score / sum(s['weight'] for s in buy_signals)
                strategy = "Multi-Strategy Confluence"
            elif sell_score > buy_score and sell_score > 0:
                action = 'SELL'
                confidence = sell_score / sum(s['weight'] for s in sell_signals)
                strategy = "Multi-Strategy Confluence"
            else:
                return None
            
            # Calculate stop loss and target
            atr = float(data['ATR'].iloc[-1]) if 'ATR' in data.columns else current_price * 0.02
            
            if action == 'BUY':
                stop_loss = current_price - (atr * 1.5)
                target = current_price + (atr * 3.0)
            else:
                stop_loss = current_price + (atr * 1.5)
                target = current_price - (atr * 3.0)
            
            # Calculate win probability
            win_probability = min(0.95, confidence * 0.8)
            if is_peak_market_hours():
                win_probability = min(0.97, win_probability * 1.1)
            
            return {
                'symbol': symbol,
                'action': action,
                'price': round(current_price, 2),
                'stop_loss': round(stop_loss, 2),
                'target': round(target, 2),
                'confidence': round(confidence, 3),
                'strategy': strategy,
                'win_probability': round(win_probability, 3),
                'timestamp': now_indian(),
                'atr': round(atr, 2),
                'signal_count': len(signals),
                'rsi': float(data['RSI14'].iloc[-1]) if 'RSI14' in data.columns else 50
            }
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    def _smc_strategy(self, data, current_price):
        """Smart Money Concept strategy"""
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            # Check for liquidity sweep
            recent_high = high.iloc[-10:-1].max()
            recent_low = low.iloc[-10:-1].min()
            
            if low.iloc[-1] < recent_low and close.iloc[-1] > recent_low:
                return {'action': 'BUY', 'confidence': 0.85}
            elif high.iloc[-1] > recent_high and close.iloc[-1] < recent_high:
                return {'action': 'SELL', 'confidence': 0.85}
            
            return None
        except Exception:
            return None
    
    def _ema_vwap_strategy(self, data, current_price):
        """EMA + VWAP strategy"""
        try:
            ema8 = data['EMA8'].iloc[-1]
            ema21 = data['EMA21'].iloc[-1]
            vwap = data['VWAP'].iloc[-1]
            
            # Price above all EMAs and VWAP
            if current_price > ema8 > ema21 > vwap:
                return {'action': 'BUY', 'confidence': 0.80}
            # Price below all EMAs and VWAP
            elif current_price < ema8 < ema21 < vwap:
                return {'action': 'SELL', 'confidence': 0.80}
            
            return None
        except Exception:
            return None
    
    def _rsi_strategy(self, data, current_price):
        """RSI strategy"""
        try:
            rsi_val = data['RSI14'].iloc[-1]
            
            # Oversold bounce
            if rsi_val < 30:
                return {'action': 'BUY', 'confidence': 0.75}
            # Overbought reversal
            elif rsi_val > 70:
                return {'action': 'SELL', 'confidence': 0.75}
            
            return None
        except Exception:
            return None
    
    def _bollinger_strategy(self, data, current_price):
        """Bollinger Bands strategy"""
        try:
            bb_lower = data['BB_Lower'].iloc[-1]
            bb_upper = data['BB_Upper'].iloc[-1]
            
            # Price touches lower band
            if current_price <= bb_lower * 1.01:
                return {'action': 'BUY', 'confidence': 0.70}
            # Price touches upper band
            elif current_price >= bb_upper * 0.99:
                return {'action': 'SELL', 'confidence': 0.70}
            
            return None
        except Exception:
            return None
    
    def _macd_strategy(self, data, current_price):
        """MACD strategy"""
        try:
            macd_line = data['MACD'].iloc[-1]
            signal_line = data['MACD_Signal'].iloc[-1]
            macd_prev = data['MACD'].iloc[-2]
            signal_prev = data['MACD_Signal'].iloc[-2]
            
            # Bullish crossover
            if macd_line > signal_line and macd_prev <= signal_prev:
                return {'action': 'BUY', 'confidence': 0.75}
            # Bearish crossover
            elif macd_line < signal_line and macd_prev >= signal_prev:
                return {'action': 'SELL', 'confidence': 0.75}
            
            return None
        except Exception:
            return None
    
    def _support_resistance_strategy(self, data, current_price):
        """Support/Resistance strategy"""
        try:
            support = data['Support'].iloc[-1]
            resistance = data['Resistance'].iloc[-1]
            
            # Near support
            if current_price <= support * 1.01:
                return {'action': 'BUY', 'confidence': 0.80}
            # Near resistance
            elif current_price >= resistance * 0.99:
                return {'action': 'SELL', 'confidence': 0.80}
            
            return None
        except Exception:
            return None
    
    def scan_universe(self, universe="Nifty 50", max_stocks=30, min_confidence=0.70):
        """Scan stock universe for signals"""
        # Determine which stocks to scan
        if universe == "Nifty 50":
            stocks = TradingConstants.NIFTY_50[:max_stocks]
        elif universe == "Nifty 100":
            stocks = TradingConstants.NIFTY_100[:max_stocks]
        else:
            stocks = TradingConstants.NIFTY_50[:max_stocks]
        
        signals = []
        
        for symbol in stocks:
            signal = self.generate_signal(symbol)
            if signal and signal['confidence'] >= min_confidence:
                signals.append(signal)
        
        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        self.signals_generated += len(signals)
        self.last_scan_time = now_indian()
        
        return signals

# ===================== PAPER TRADER =====================
class PaperTrader:
    """Paper trading engine"""
    
    def __init__(self, initial_capital, risk_manager, data_manager):
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.positions = {}
        self.trade_log = []
        self.risk_manager = risk_manager
        self.data_manager = data_manager
        self.daily_stats = {
            'trades_today': 0,
            'auto_trades': 0,
            'total_pnl': 0.0
        }
        self.last_reset = now_indian().date()
        
    def reset_daily_counts(self):
        """Reset daily counts if new day"""
        current_date = now_indian().date()
        if current_date != self.last_reset:
            self.daily_stats.update({
                'trades_today': 0,
                'auto_trades': 0,
                'total_pnl': 0.0
            })
            self.last_reset = current_date
            self.risk_manager.reset_daily()
    
    def can_trade(self):
        """Check if trading is allowed"""
        self.reset_daily_counts()
        return (
            self.daily_stats['trades_today'] < config.MAX_DAILY_TRADES and
            market_open()
        )
    
    def execute_trade(self, symbol, action, quantity, price, 
                     stop_loss=None, target=None, strategy="Manual", 
                     auto_trade=False, confidence=0.5):
        """Execute a trade"""
        if not self.can_trade():
            return False, "Daily trade limit reached or market closed"
        
        # Check risk
        risk_check = self.risk_manager.check_trade(symbol, action, quantity, price, confidence)
        if not risk_check['approved']:
            return False, f"Risk check failed: {risk_check['reason']}"
        
        trade_value = float(quantity) * float(price)
        
        if action == "BUY" and trade_value > self.cash:
            return False, "Insufficient capital"
        
        # Create trade record
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
            "closed_pnl": 0.0,
            "entry_time": now_indian().strftime("%H:%M:%S"),
            "auto_trade": auto_trade,
            "strategy": strategy,
            "confidence": confidence
        }
        
        # Update positions
        if action == "BUY":
            self.positions[symbol] = record
            self.cash -= trade_value
        else:
            margin = trade_value * 0.2  # Margin for short selling
            record["margin_used"] = margin
            self.positions[symbol] = record
            self.cash -= margin
        
        # Update statistics
        self.daily_stats['trades_today'] += 1
        if auto_trade:
            self.daily_stats['auto_trades'] += 1
        
        self.trade_log.append(record)
        self.risk_manager.update_position(opened=True)
        
        msg = f"{'[AUTO] ' if auto_trade else ''}{action} {quantity} {symbol} @ ₹{price:.2f}"
        if strategy != "Manual":
            msg += f" | Strategy: {strategy}"
        
        return True, msg
    
    def execute_trade_from_signal(self, signal, max_quantity=50):
        """Execute trade based on signal"""
        symbol = signal['symbol']
        action = signal['action']
        price = signal['price']
        stop_loss = signal['stop_loss']
        target = signal['target']
        strategy = signal['strategy']
        confidence = signal['confidence']
        
        # Calculate position size
        position_size_pct = min(0.2, confidence * 0.25)
        max_trade_value = self.cash * position_size_pct
        quantity = int(max_trade_value / price)
        quantity = min(quantity, max_quantity)
        
        if quantity < 1:
            return False, "Position size too small"
        
        return self.execute_trade(
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
            target=target,
            strategy=strategy,
            auto_trade=True,
            confidence=confidence
        )
    
    def update_positions(self):
        """Update all positions with current prices"""
        for symbol, pos in list(self.positions.items()):
            if pos.get("status") != "OPEN":
                continue
            
            try:
                current_price = self.data_manager.get_live_price(symbol)
                if current_price is None:
                    continue
                
                pos["current_price"] = current_price
                entry = pos["entry_price"]
                quantity = pos["quantity"]
                
                if pos["action"] == "BUY":
                    pnl = (current_price - entry) * quantity
                else:
                    pnl = (entry - current_price) * quantity
                
                pos["current_pnl"] = float(pnl)
                
                # Check stop loss and target
                sl = pos.get("stop_loss")
                tg = pos.get("target")
                
                if sl is not None:
                    if (pos["action"] == "BUY" and current_price <= sl) or \
                       (pos["action"] == "SELL" and current_price >= sl):
                        self.close_position(symbol, exit_price=sl)
                        continue
                
                if tg is not None:
                    if (pos["action"] == "BUY" and current_price >= tg) or \
                       (pos["action"] == "SELL" and current_price <= tg):
                        self.close_position(symbol, exit_price=tg)
                        continue
                        
            except Exception as e:
                logger.error(f"Error updating position {symbol}: {e}")
                continue
    
    def close_position(self, symbol, exit_price=None):
        """Close a position"""
        if symbol not in self.positions:
            return False, "Position not found"
        
        pos = self.positions[symbol]
        
        if exit_price is None:
            current_price = self.data_manager.get_live_price(symbol)
            if current_price is None:
                current_price = pos["entry_price"]
            exit_price = current_price
        
        # Calculate P&L
        if pos["action"] == "BUY":
            pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
            self.cash += pos["quantity"] * exit_price
        else:
            pnl = (pos["entry_price"] - exit_price) * pos["quantity"]
            margin = pos.get("margin_used", 0)
            self.cash += margin + (pos["quantity"] * pos["entry_price"])
        
        # Update position record
        pos["status"] = "CLOSED"
        pos["exit_price"] = float(exit_price)
        pos["closed_pnl"] = float(pnl)
        pos["exit_time"] = now_indian()
        pos["exit_time_str"] = now_indian().strftime("%H:%M:%S")
        
        # Update statistics
        self.daily_stats['total_pnl'] += pnl
        self.risk_manager.update_trade(pnl)
        self.risk_manager.update_position(opened=False)
        
        # Remove from active positions
        del self.positions[symbol]
        
        return True, f"Closed {symbol} @ ₹{exit_price:.2f} | P&L: ₹{pnl:+.2f}"
    
    def close_all_positions(self):
        """Close all open positions"""
        results = []
        for symbol in list(self.positions.keys()):
            success, msg = self.close_position(symbol)
            results.append((success, msg))
        
        success_count = sum(1 for s, _ in results if s)
        total = len(results)
        
        return success_count > 0, f"Closed {success_count}/{total} positions"
    
    def get_open_positions(self):
        """Get all open positions"""
        self.update_positions()
        positions = []
        
        for symbol, pos in self.positions.items():
            if pos.get("status") == "OPEN":
                positions.append({
                    "Symbol": symbol.replace(".NS", ""),
                    "Action": pos["action"],
                    "Quantity": pos["quantity"],
                    "Entry Price": f"₹{pos['entry_price']:.2f}",
                    "Current Price": f"₹{pos.get('current_price', pos['entry_price']):.2f}",
                    "P&L": f"₹{pos.get('current_pnl', 0):+.2f}",
                    "Stop Loss": f"₹{pos.get('stop_loss', 0):.2f}" if pos.get('stop_loss') else "N/A",
                    "Target": f"₹{pos.get('target', 0):.2f}" if pos.get('target') else "N/A",
                    "Strategy": pos.get("strategy", "Manual"),
                    "Auto Trade": "Yes" if pos.get("auto_trade") else "No"
                })
        
        return positions
    
    def get_trade_history(self):
        """Get trade history"""
        history = []
        for trade in self.trade_log[-50:]:  # Last 50 trades
            if trade.get("status") == "CLOSED":
                history.append({
                    "Symbol": trade['symbol'].replace(".NS", ""),
                    "Action": trade['action'],
                    "Quantity": trade['quantity'],
                    "Entry Price": f"₹{trade['entry_price']:.2f}",
                    "Exit Price": f"₹{trade.get('exit_price', 0):.2f}",
                    "P&L": f"₹{trade.get('closed_pnl', 0):+.2f}",
                    "Entry Time": trade.get('entry_time', ''),
                    "Exit Time": trade.get('exit_time_str', ''),
                    "Strategy": trade.get('strategy', 'Manual'),
                    "Auto Trade": "Yes" if trade.get('auto_trade') else "No"
                })
        
        return history
    
    def get_performance_summary(self):
        """Get performance summary"""
        closed_trades = [t for t in self.trade_log if t.get("status") == "CLOSED"]
        open_positions = [p for p in self.positions.values() if p.get("status") == "OPEN"]
        
        total_trades = len(closed_trades)
        wins = len([t for t in closed_trades if t.get("closed_pnl", 0) > 0])
        
        total_pnl = sum([t.get("closed_pnl", 0) for t in closed_trades])
        open_pnl = sum([p.get("current_pnl", 0) for p in open_positions])
        
        win_rate = wins / total_trades if total_trades > 0 else 0
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        total_value = self.cash + sum([
            p.get("quantity", 0) * p.get("current_price", p.get("entry_price", 0))
            for p in open_positions
        ])
        
        return {
            'total_value': total_value,
            'available_cash': self.cash,
            'open_positions': len(open_positions),
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'open_pnl': open_pnl,
            'avg_pnl_per_trade': avg_pnl,
            'trades_today': self.daily_stats['trades_today'],
            'auto_trades': self.daily_stats['auto_trades']
        }

# ===================== ALGO ENGINE =====================
class AlgoState(Enum):
    """Algo engine states"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"

class AlgoEngine:
    """Algorithmic trading engine"""
    
    def __init__(self, trader, risk_manager, signal_generator, config):
        self.trader = trader
        self.risk_manager = risk_manager
        self.signal_generator = signal_generator
        self.config = config
        
        self.state = AlgoState.STOPPED
        self.active_positions = {}
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'today_pnl': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0
        }
        
        self._stop_event = threading.Event()
        self._scheduler_thread = None
        self.daily_exit_completed = False
        self.last_signal_scan = 0
        
        logger.info("AlgoEngine initialized")
    
    def start(self):
        """Start the algo engine"""
        if self.state == AlgoState.RUNNING:
            return False
        
        self.state = AlgoState.RUNNING
        self._stop_event.clear()
        
        self._scheduler_thread = threading.Thread(
            target=self._run_scheduler,
            daemon=True
        )
        self._scheduler_thread.start()
        
        logger.info("AlgoEngine started")
        return True
    
    def stop(self):
        """Stop the algo engine"""
        self.state = AlgoState.STOPPED
        self._stop_event.set()
        
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        
        logger.info("AlgoEngine stopped")
    
    def pause(self):
        """Pause the algo engine"""
        self.state = AlgoState.PAUSED
        logger.info("AlgoEngine paused")
    
    def resume(self):
        """Resume the algo engine"""
        if self.state == AlgoState.PAUSED:
            self.state = AlgoState.RUNNING
            logger.info("AlgoEngine resumed")
    
    def emergency_stop(self, reason="Manual trigger"):
        """Emergency stop"""
        logger.critical(f"EMERGENCY STOP: {reason}")
        self.state = AlgoState.EMERGENCY_STOP
        self._stop_event.set()
    
    def _run_scheduler(self):
        """Main scheduler loop"""
        logger.info("Algo scheduler started")
        
        while not self._stop_event.is_set():
            try:
                if self.state != AlgoState.RUNNING:
                    time.sleep(1)
                    continue
                
                # Check market hours
                if not market_open():
                    time.sleep(10)
                    continue
                
                # Check daily exit
                if should_exit_all_positions() and not self.daily_exit_completed:
                    logger.info("3:35 PM - Exiting all positions")
                    self.exit_all_positions()
                    self.daily_exit_completed = True
                    time.sleep(60)
                    continue
                
                # Reset daily exit flag
                current_time = now_indian()
                if current_time.hour == 9 and current_time.minute < 30:
                    self.daily_exit_completed = False
                
                # Update positions
                self.trader.update_positions()
                
                # Check risk limits
                if self._check_risk_limits():
                    self.emergency_stop("Risk limits breached")
                    continue
                
                # Generate and process signals
                current_time_ts = time.time()
                if current_time_ts - self.last_signal_scan > 300:  # Every 5 minutes
                    self._scan_and_process_signals()
                    self.last_signal_scan = current_time_ts
                
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(10)
        
        logger.info("Algo scheduler stopped")
    
    def _check_risk_limits(self):
        """Check if risk limits are breached"""
        risk_status = self.risk_manager.get_status()
        return not risk_status['within_limits']
    
    def _scan_and_process_signals(self):
        """Scan for and process signals"""
        if not market_open() or not is_peak_market_hours():
            return
        
        try:
            signals = self.signal_generator.scan_universe(
                universe="Nifty 50",
                max_stocks=30,
                min_confidence=self.config.ALGO_MIN_CONFIDENCE
            )
            
            if signals:
                self._process_signals(signals[:5])  # Process top 5 signals
        
        except Exception as e:
            logger.error(f"Error in signal scanning: {e}")
    
    def _process_signals(self, signals):
        """Process generated signals"""
        for signal in signals:
            # Check if already have position
            if signal['symbol'] in self.trader.positions:
                continue
            
            # Check risk limits
            risk_status = self.risk_manager.get_status()
            if risk_status['positions_open'] >= self.config.ALGO_MAX_POSITIONS:
                logger.info("Max positions limit reached")
                break
            
            # Execute trade
            success, msg = self.trader.execute_trade_from_signal(signal)
            
            if success:
                logger.info(f"Algo executed: {signal['symbol']} {signal['action']}")
                self.active_positions[signal['symbol']] = {
                    'entry_price': signal['price'],
                    'action': signal['action'],
                    'strategy': signal['strategy'],
                    'timestamp': now_indian()
                }
    
    def exit_all_positions(self):
        """Exit all positions"""
        success, msg = self.trader.close_all_positions()
        if success:
            logger.info(f"All positions exited: {msg}")
            self.active_positions.clear()
        else:
            logger.warning(f"Failed to exit all positions: {msg}")
    
    def get_state(self):
        """Get current state"""
        return self.state
    
    def get_status(self):
        """Get algo engine status"""
        # Update performance metrics
        closed_trades = [t for t in self.trader.trade_log 
                        if t.get("status") == "CLOSED" and t.get("auto_trade")]
        
        winning = [t for t in closed_trades if t.get("closed_pnl", 0) > 0]
        losing = [t for t in closed_trades if t.get("closed_pnl", 0) <= 0]
        
        total_pnl = sum(t.get("closed_pnl", 0) for t in closed_trades)
        win_pnl = sum(t.get("closed_pnl", 0) for t in winning) if winning else 0
        loss_pnl = abs(sum(t.get("closed_pnl", 0) for t in losing)) if losing else 0
        
        win_rate = len(winning) / len(closed_trades) if closed_trades else 0
        avg_win = win_pnl / len(winning) if winning else 0
        avg_loss = loss_pnl / len(losing) if losing else 0
        profit_factor = win_pnl / loss_pnl if loss_pnl > 0 else 0
        
        return {
            'state': self.state.value,
            'active_positions': len(self.active_positions),
            'total_trades': len(closed_trades),
            'today_pnl': self.performance['today_pnl'],
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'daily_exit_completed': self.daily_exit_completed
        }

# ===================== UI STYLING =====================
def load_css():
    """Load CSS styles"""
    st.markdown("""
    <style>
        /* Base styling */
        .stApp {
            background: linear-gradient(135deg, #fff5e6 0%, #ffe8cc 100%);
        }
        
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        
        /* Header */
        .main-header {
            text-align: center;
            padding: 1rem;
            background: linear-gradient(135deg, #ff8c00 0%, #ff6b00 100%);
            border-radius: 15px;
            margin-bottom: 1.5rem;
            color: white;
            box-shadow: 0 4px 12px rgba(255, 140, 0, 0.2);
        }
        
        /* Cards */
        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 1rem;
            border-left: 4px solid #ff8c00;
            box-shadow: 0 2px 8px rgba(255, 140, 0, 0.1);
            margin-bottom: 1rem;
        }
        
        /* Alerts */
        .alert-success {
            background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
            border-left: 4px solid #059669;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        .alert-warning {
            background: linear-gradient(135deg, #ffe8cc 0%, #ffd9a6 100%);
            border-left: 4px solid #ff8c00;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        .alert-danger {
            background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
            border-left: 4px solid #dc2626;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: linear-gradient(135deg, #ffe8cc 0%, #ffd9a6 100%);
            padding: 8px;
            border-radius: 12px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: white;
            border-radius: 8px;
            padding: 12px 20px;
            font-weight: 600;
            color: #d97706;
            border: 2px solid transparent;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #ff8c00 0%, #ff6b00 100%);
            color: white;
            border-color: #ff8c00;
            box-shadow: 0 4px 8px rgba(255, 140, 0, 0.3);
        }
        
        /* Buttons */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
        }
        
        /* Tables */
        .dataframe {
            border-radius: 8px;
            overflow: hidden;
        }
    </style>
    """, unsafe_allow_html=True)

# ===================== SESSION STATE =====================
def init_session_state():
    """Initialize session state"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'trader' not in st.session_state:
        st.session_state.trader = None
    if 'algo_engine' not in st.session_state:
        st.session_state.algo_engine = None
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = None
    if 'risk_manager' not in st.session_state:
        st.session_state.risk_manager = None
    if 'strategy_manager' not in st.session_state:
        st.session_state.strategy_manager = None
    if 'signal_generator' not in st.session_state:
        st.session_state.signal_generator = None
    if 'generated_signals' not in st.session_state:
        st.session_state.generated_signals = []
    if 'signal_quality' not in st.session_state:
        st.session_state.signal_quality = 0
    if 'refresh_count' not in st.session_state:
        st.session_state.refresh_count = 0
    if 'auto_trade_enabled' not in st.session_state:
        st.session_state.auto_trade_enabled = False

# ===================== TRADING SYSTEM =====================
class TradingSystem:
    """Main trading system orchestrator"""
    
    def __init__(self):
        self.initialized = False
        self.components = {}
    
    def initialize(self):
        """Initialize all system components"""
        try:
            logger.info("Initializing Trading System...")
            
            # 1. Data Manager
            self.components['data_manager'] = DataManager()
            
            # 2. Risk Manager
            self.components['risk_manager'] = RiskManager(config)
            
            # 3. Strategy Manager
            self.components['strategy_manager'] = StrategyManager()
            
            # 4. Signal Generator
            self.components['signal_generator'] = SignalGenerator(
                self.components['data_manager'],
                self.components['strategy_manager']
            )
            
            # 5. Paper Trader
            self.components['trader'] = PaperTrader(
                initial_capital=config.INITIAL_CAPITAL,
                risk_manager=self.components['risk_manager'],
                data_manager=self.components['data_manager']
            )
            
            # 6. Algo Engine
            self.components['algo_engine'] = AlgoEngine(
                trader=self.components['trader'],
                risk_manager=self.components['risk_manager'],
                signal_generator=self.components['signal_generator'],
                config=config
            )
            
            # Update session state
            st.session_state.update({
                'trader': self.components['trader'],
                'algo_engine': self.components['algo_engine'],
                'data_manager': self.components['data_manager'],
                'risk_manager': self.components['risk_manager'],
                'strategy_manager': self.components['strategy_manager'],
                'signal_generator': self.components['signal_generator'],
                'initialized': True
            })
            
            self.initialized = True
            logger.info("Trading System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize trading system: {e}")
            st.error(f"System initialization failed: {str(e)}")
            return False
    
    def get_component(self, name):
        """Get a system component"""
        return self.components.get(name)

# ===================== SIDEBAR =====================
def render_sidebar(trading_system):
    """Render sidebar with controls"""
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <h2 style='color: #ff8c00;'>⚙️ Trading Terminal</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # System Status
        st.subheader("📊 System Status")
        status_cols = st.columns(2)
        with status_cols[0]:
            status = "🟢 READY" if trading_system.initialized else "🔴 ERROR"
            st.metric("System", status)
        with status_cols[1]:
            market = "🟢 OPEN" if market_open() else "🔴 CLOSED"
            st.metric("Market", market)
        
        # Risk Settings
        st.subheader("🎯 Risk Settings")
        with st.expander("Configure", expanded=False):
            max_loss = st.number_input(
                "Max Daily Loss (₹)",
                min_value=1000,
                max_value=1000000,
                value=50000,
                step=5000
            )
            
            max_pos = st.slider(
                "Max Positions",
                min_value=1,
                max_value=20,
                value=5
            )
            
            min_conf = st.slider(
                "Min Confidence",
                min_value=0.5,
                max_value=0.95,
                value=0.70,
                step=0.05
            )
            
            if st.button("Update Settings"):
                config.ALGO_MAX_DAILY_LOSS = max_loss
                config.ALGO_MAX_POSITIONS = max_pos
                config.ALGO_MIN_CONFIDENCE = min_conf
                st.success("Settings updated!")
        
        # Trading Controls
        st.subheader("🕹️ Trading Controls")
        auto_trade = st.checkbox(
            "Enable Auto Trading",
            value=st.session_state.auto_trade_enabled
        )
        st.session_state.auto_trade_enabled = auto_trade
        
        if st.button("🔄 Refresh All"):
            st.rerun()
        
        # Strategy Selection
        st.subheader("📈 Active Strategies")
        if trading_system.get_component('strategy_manager'):
            strategies = trading_system.get_component('strategy_manager').get_all_strategies()
            for strategy_id, strategy in strategies.items():
                enabled = st.checkbox(
                    strategy['name'],
                    value=True,
                    key=f"strategy_{strategy_id}"
                )
                if enabled:
                    trading_system.get_component('strategy_manager').enable_strategy(strategy_id)
                else:
                    trading_system.get_component('strategy_manager').disable_strategy(strategy_id)
        
        # Quick Actions
        st.subheader("⚡ Quick Actions")
        if st.button("📤 Close All Positions", type="secondary"):
            if trading_system.get_component('trader'):
                success, message = trading_system.get_component('trader').close_all_positions()
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        # System Info
        st.divider()
        st.caption(f"Version: 2.0.0 | {now_indian().strftime('%H:%M:%S')}")

# ===================== DASHBOARD =====================
def render_dashboard(trading_system):
    """Render main dashboard"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style='margin: 0;'>📈 RANTV TERMINAL PRO</h1>
        <p style='margin: 5px 0 0 0; font-size: 16px;'>Complete Algo Trading Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Market Overview
    st.subheader("🌐 Market Overview")
    
    # Get market data
    try:
        nifty = yf.Ticker("^NSEI")
        nifty_data = nifty.history(period="1d")
        
        banknifty = yf.Ticker("^NSEBANK")
        banknifty_data = banknifty.history(period="1d")
        
        if not nifty_data.empty and not banknifty_data.empty:
            nifty_price = nifty_data['Close'].iloc[-1]
            nifty_prev = nifty_data['Close'].iloc[-2] if len(nifty_data) > 1 else nifty_price
            nifty_change = ((nifty_price - nifty_prev) / nifty_prev) * 100
            
            banknifty_price = banknifty_data['Close'].iloc[-1]
            banknifty_prev = banknifty_data['Close'].iloc[-2] if len(banknifty_data) > 1 else banknifty_price
            banknifty_change = ((banknifty_price - banknifty_prev) / banknifty_prev) * 100
            
            avg_change = (nifty_change + banknifty_change) / 2
            if avg_change > 0.5:
                sentiment = "BULLISH 📈"
            elif avg_change < -0.5:
                sentiment = "BEARISH 📉"
            else:
                sentiment = "NEUTRAL ➡️"
        else:
            nifty_price = 22000
            nifty_change = 0.15
            banknifty_price = 48000
            banknifty_change = 0.25
            sentiment = "NEUTRAL ➡️"
    except:
        nifty_price = 22000
        nifty_change = 0.15
        banknifty_price = 48000
        banknifty_change = 0.25
        sentiment = "NEUTRAL ➡️"
    
    # Display metrics
    cols = st.columns(5)
    with cols[0]:
        st.metric("NIFTY 50", f"₹{nifty_price:,.0f}", f"{nifty_change:+.2f}%")
    with cols[1]:
        st.metric("BANK NIFTY", f"₹{banknifty_price:,.0f}", f"{banknifty_change:+.2f}%")
    with cols[2]:
        st.metric("Market Sentiment", sentiment)
    with cols[3]:
        status = "🟢 OPEN" if market_open() else "🔴 CLOSED"
        st.metric("Market Status", status)
    with cols[4]:
        peak = "🟢 YES" if is_peak_market_hours() else "🔴 NO"
        st.metric("Peak Hours", peak)
    
    # Account Summary
    st.subheader("💰 Account Summary")
    
    if trading_system.get_component('trader'):
        trader = trading_system.get_component('trader')
        perf = trader.get_performance_summary()
        
        acc_cols = st.columns(4)
        with acc_cols[0]:
            st.metric("Total Value", f"₹{perf['total_value']:,.0f}")
        with acc_cols[1]:
            st.metric("Available Cash", f"₹{perf['available_cash']:,.0f}")
        with acc_cols[2]:
            st.metric("Open Positions", perf['open_positions'])
        with acc_cols[3]:
            pnl_color = "inverse" if perf['total_pnl'] < 0 else "normal"
            st.metric("Total P&L", f"₹{perf['total_pnl']:+,.2f}", delta_color=pnl_color)
    
    # System Health
    st.subheader("⚙️ System Health")
    
    health_cols = st.columns(4)
    with health_cols[0]:
        status = "🟢 READY" if trading_system.initialized else "🔴 ERROR"
        st.metric("Trading System", status)
    
    with health_cols[1]:
        if trading_system.get_component('algo_engine'):
            algo_state = trading_system.get_component('algo_engine').get_state().value
            status_emoji = "🟢" if algo_state == "running" else "🟡" if algo_state == "paused" else "🔴"
            st.metric("Algo Engine", f"{status_emoji} {algo_state.upper()}")
    
    with health_cols[2]:
        if trading_system.get_component('risk_manager'):
            risk_status = trading_system.get_component('risk_manager').get_status()
            status_emoji = "🟢" if risk_status['within_limits'] else "🔴"
            st.metric("Risk Engine", f"{status_emoji} {'OK' if risk_status['within_limits'] else 'LIMIT'}")
    
    with health_cols[3]:
        st.metric("Data Feed", "🟢 LIVE")
    
    # Refresh info
    st.markdown(f"<div style='text-align: right; color: #6b7280; font-size: 14px;'>Refresh: {st.session_state.refresh_count}</div>", unsafe_allow_html=True)

# ===================== SIGNALS TAB =====================
def render_signals_tab(trading_system):
    """Render signals tab"""
    st.subheader("🚦 Trading Signals")
    
    # Signal generation controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        universe = st.selectbox(
            "Stock Universe",
            ["Nifty 50", "Nifty 100"],
            key="signals_universe"
        )
    
    with col2:
        min_confidence = st.slider(
            "Min Confidence",
            min_value=0.60,
            max_value=0.95,
            value=0.70,
            step=0.05,
            key="signals_min_conf"
        )
    
    with col3:
        max_signals = st.number_input(
            "Max Signals",
            min_value=1,
            max_value=20,
            value=10,
            key="signals_max_count"
        )
    
    # Generate signals button
    if st.button("🚀 Generate Signals", type="primary", key="generate_signals_main"):
        if trading_system.get_component('signal_generator'):
            with st.spinner(f"Scanning {universe}..."):
                try:
                    scan_size = 50 if universe == "Nifty 50" else 100
                    signals = trading_system.get_component('signal_generator').scan_universe(
                        universe=universe,
                        max_stocks=min(scan_size, 30),
                        min_confidence=min_confidence
                    )
                    
                    st.session_state.generated_signals = signals[:max_signals]
                    
                    if signals:
                        confidences = [s['confidence'] for s in signals]
                        st.session_state.signal_quality = np.mean(confidences) * 100
                    
                    st.success(f"✅ Generated {len(signals)} signals")
                    
                except Exception as e:
                    st.error(f"❌ Signal generation failed: {str(e)}")
    
    # Display signals
    if st.session_state.generated_signals:
        signals = st.session_state.generated_signals
        quality = st.session_state.get('signal_quality', 0)
        
        if quality >= 70:
            quality_class = "alert-success"
            quality_text = "HIGH QUALITY"
        elif quality >= 50:
            quality_class = "alert-warning"
            quality_text = "MEDIUM QUALITY"
        else:
            quality_class = "alert-danger"
            quality_text = "LOW QUALITY"
        
        st.markdown(f"""
        <div class="{quality_class}">
            <strong>📊 Signal Quality: {quality_text}</strong> | 
            Score: {quality:.1f}/100
        </div>
        """, unsafe_allow_html=True)
        
        # Display signals table
        signal_data = []
        for i, signal in enumerate(signals):
            signal_data.append({
                "#": i + 1,
                "Symbol": signal['symbol'].replace('.NS', ''),
                "Action": f"{'🟢 BUY' if signal['action'] == 'BUY' else '🔴 SELL'}",
                "Price": f"₹{signal['price']:.2f}",
                "Stop Loss": f"₹{signal['stop_loss']:.2f}",
                "Target": f"₹{signal['target']:.2f}",
                "Confidence": f"{signal['confidence']:.1%}",
                "Strategy": signal['strategy'],
                "RSI": f"{signal.get('rsi', 0):.1f}"
            })
        
        if signal_data:
            df_signals = pd.DataFrame(signal_data)
            st.dataframe(df_signals, use_container_width=True)
            
            # Execution controls
            st.subheader("🤖 Execute Signals")
            exec_cols = st.columns(3)
            
            with exec_cols[0]:
                if st.button("📈 Execute BUY Signals", type="secondary"):
                    execute_signals(signals, "BUY", trading_system)
            
            with exec_cols[1]:
                if st.button("📉 Execute SELL Signals", type="secondary"):
                    execute_signals(signals, "SELL", trading_system)
            
            with exec_cols[2]:
                if st.button("🎯 Execute Top 3", type="primary"):
                    execute_signals(signals[:3], "ANY", trading_system)
    else:
        st.info("Click 'Generate Signals' to scan for trading opportunities")

def execute_signals(signals, action_filter, trading_system):
    """Execute filtered signals"""
    if not trading_system.get_component('trader'):
        st.error("Trader not initialized")
        return
    
    trader = trading_system.get_component('trader')
    filtered_signals = signals if action_filter == "ANY" else [s for s in signals if s['action'] == action_filter]
    
    executed = 0
    for signal in filtered_signals:
        success, msg = trader.execute_trade_from_signal(signal)
        if success:
            executed += 1
            st.success(f"✅ {msg}")
        else:
            st.warning(f"⚠️ {msg}")
    
    if executed > 0:
        st.success(f"✅ Executed {executed} trades!")
        st.rerun()

# ===================== PAPER TRADING TAB =====================
def render_paper_trading_tab(trading_system):
    """Render paper trading tab"""
    st.subheader("💰 Paper Trading")
    
    if trading_system.get_component('trader'):
        trader = trading_system.get_component('trader')
        
        # Manual trade execution
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            symbol = st.selectbox("Symbol", TradingConstants.NIFTY_50[:20], key="manual_symbol")
        
        with col2:
            action = st.selectbox("Action", ["BUY", "SELL"], key="manual_action")
        
        with col3:
            quantity = st.number_input("Quantity", min_value=1, value=10, key="manual_quantity")
        
        with col4:
            strategy = st.selectbox("Strategy", 
                                   ["Manual"] + list(TradingConstants.TRADING_STRATEGIES.keys()),
                                   key="manual_strategy")
        
        if st.button("Execute Manual Trade", type="primary"):
            data = trading_system.get_component('data_manager').get_stock_data(symbol, "15m")
            if data is not None and len(data) > 0:
                price = float(data['Close'].iloc[-1])
                atr = float(data['ATR'].iloc[-1]) if 'ATR' in data.columns else price * 0.02
                
                if action == "BUY":
                    stop_loss = price - (atr * 1.5)
                    target = price + (atr * 3)
                else:
                    stop_loss = price + (atr * 1.5)
                    target = price - (atr * 3)
                
                success, message = trader.execute_trade(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    price=price,
                    stop_loss=stop_loss,
                    target=target,
                    strategy=strategy
                )
                
                if success:
                    st.success(f"✅ {message}")
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
            else:
                st.error("Could not fetch price data")
        
        # Current positions
        st.subheader("Current Positions")
        positions = trader.get_open_positions()
        
        if positions:
            positions_df = pd.DataFrame(positions)
            st.dataframe(positions_df, use_container_width=True)
            
            if st.button("Close All Positions", type="secondary"):
                success, message = trader.close_all_positions()
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        else:
            st.info("No open positions")

# ===================== HISTORY TAB =====================
def render_history_tab(trading_system):
    """Render history tab"""
    st.subheader("📋 Trade History")
    
    if trading_system.get_component('trader'):
        trader = trading_system.get_component('trader')
        history = trader.get_trade_history()
        
        if history:
            history_df = pd.DataFrame(history)
            st.dataframe(history_df, use_container_width=True)
            
            # Performance summary
            st.subheader("Performance Summary")
            perf = trader.get_performance_summary()
            
            perf_cols = st.columns(4)
            with perf_cols[0]:
                st.metric("Total Trades", perf['total_trades'])
            with perf_cols[1]:
                st.metric("Win Rate", f"{perf['win_rate']:.1%}")
            with perf_cols[2]:
                st.metric("Total P&L", f"₹{perf['total_pnl']:+.2f}")
            with perf_cols[3]:
                st.metric("Avg P&L/Trade", f"₹{perf['avg_pnl_per_trade']:+.2f}")
        else:
            st.info("No trade history available")

# ===================== ALGO TRADING TAB =====================
def render_algo_tab(trading_system):
    """Render algo trading tab"""
    st.subheader("🤖 Algorithmic Trading Engine")
    
    if not trading_system.get_component('algo_engine'):
        st.warning("Algo engine not initialized")
        return
    
    algo_engine = trading_system.get_component('algo_engine')
    algo_status = algo_engine.get_status()
    
    # Status display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        state = algo_status['state']
        if state == "running":
            status_color = "🟢"
            status_class = "alert-success"
        elif state == "paused":
            status_color = "🟡"
            status_class = "alert-warning"
        else:
            status_color = "🔴"
            status_class = "alert-danger"
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 12px; color: #6b7280;">Engine Status</div>
            <div style="font-size: 20px; font-weight: bold;">{status_color} {state.upper()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Active Positions", algo_status['active_positions'])
    
    with col3:
        st.metric("Today's P&L", f"₹{algo_status['today_pnl']:+.2f}")
    
    with col4:
        st.metric("Total Trades", algo_status['total_trades'])
    
    # Engine controls
    st.subheader("Engine Controls")
    ctrl_cols = st.columns(5)
    
    with ctrl_cols[0]:
        if st.button("▶️ Start", type="primary", disabled=algo_status['state'] == "running"):
            if algo_engine.start():
                st.success("Algo engine started!")
                st.rerun()
    
    with ctrl_cols[1]:
        if st.button("⏸️ Pause", disabled=algo_status['state'] != "running"):
            algo_engine.pause()
            st.info("Algo engine paused")
            st.rerun()
    
    with ctrl_cols[2]:
        if st.button("▶️ Resume", disabled=algo_status['state'] != "paused"):
            algo_engine.resume()
            st.success("Algo engine resumed")
            st.rerun()
    
    with ctrl_cols[3]:
        if st.button("⏹️ Stop", disabled=algo_status['state'] == "stopped"):
            algo_engine.stop()
            st.info("Algo engine stopped")
            st.rerun()
    
    with ctrl_cols[4]:
        if st.button("🚨 Emergency Stop", type="secondary"):
            algo_engine.emergency_stop()
            st.error("EMERGENCY STOP ACTIVATED")
            st.rerun()
    
    # Daily schedule
    st.subheader("🕒 Daily Schedule")
    sched_cols = st.columns(4)
    with sched_cols[0]:
        st.metric("Market Open", "9:15 AM")
    with sched_cols[1]:
        st.metric("Auto Trading", "9:30 AM - 2:30 PM")
    with sched_cols[2]:
        st.metric("Market Close", "3:30 PM")
    with sched_cols[3]:
        st.metric("Auto Exit", "3:35 PM")
    
    # Manual daily exit
    if st.button("📤 Force Daily Exit Now", type="secondary"):
        algo_engine.exit_all_positions()
        st.success("All positions exited!")
        st.rerun()
    
    # Performance metrics
    st.subheader("📊 Performance Metrics")
    perf_cols = st.columns(4)
    with perf_cols[0]:
        st.metric("Win Rate", f"{algo_status['win_rate']:.1%}")
    with perf_cols[1]:
        st.metric("Avg Win", f"₹{algo_status['avg_win']:.2f}")
    with perf_cols[2]:
        st.metric("Avg Loss", f"₹{algo_status['avg_loss']:.2f}")
    with perf_cols[3]:
        st.metric("Profit Factor", f"{algo_status['profit_factor']:.2f}")

# ===================== BACKTESTING TAB =====================
def render_backtesting_tab():
    """Render backtesting tab"""
    st.subheader("⚙️ Backtesting")
    st.info("Backtesting module coming soon...")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now().date())
    with col3:
        initial_capital = st.number_input("Initial Capital", value=2000000, step=100000)
    
    if st.button("Run Backtest", type="primary"):
        st.warning("Backtesting engine is under development")

# ===================== MAIN APPLICATION =====================
def main():
    """Main application"""
    
    # Set page config
    st.set_page_config(
        page_title="Rantv Terminal Pro",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize
    load_css()
    init_session_state()
    
    # Initialize trading system
    trading_system = TradingSystem()
    
    # Auto-refresh
    st_autorefresh(interval=config.PRICE_REFRESH_MS, key="main_auto_refresh")
    st.session_state.refresh_count += 1
    
    # Initialize button
    if not trading_system.initialized:
        st.title("📈 RANTV TERMINAL PRO")
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### Welcome to Rantv Terminal Pro")
            st.markdown("""
            **Complete Algorithmic Trading Platform**
            
            Features:
            - 📊 Real-time market data
            - 🚦 Multi-strategy signal generation
            - 🤖 Automated algo trading
            - 🎯 Advanced risk management
            - 💰 Paper trading simulator
            - 📈 Performance analytics
            
            **Ready to start?**
            """)
            
            if st.button("🚀 Initialize Trading System", type="primary", use_container_width=True):
                with st.spinner("Initializing system components..."):
                    if trading_system.initialize():
                        st.success("✅ System initialized successfully!")
                        st.rerun()
                    else:
                        st.error("❌ System initialization failed")
        
        return
    
    # Main application
    render_sidebar(trading_system)
    render_dashboard(trading_system)
    
    # Create tabs
    tabs = st.tabs([
        "📈 Dashboard",
        "🚦 Signals",
        "💰 Paper Trading",
        "📋 History",
        "🤖 Algo Trading",
        "⚙️ Backtesting"
    ])
    
    # Tab 1: Dashboard (already rendered)
    with tabs[0]:
        st.info("Dashboard is the main view shown above")
    
    # Tab 2: Signals
    with tabs[1]:
        render_signals_tab(trading_system)
    
    # Tab 3: Paper Trading
    with tabs[2]:
        render_paper_trading_tab(trading_system)
    
    # Tab 4: History
    with tabs[3]:
        render_history_tab(trading_system)
    
    # Tab 5: Algo Trading
    with tabs[4]:
        render_algo_tab(trading_system)
    
    # Tab 6: Backtesting
    with tabs[5]:
        render_backtesting_tab()
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #6b7280; font-size: 12px;">
        <strong>Rantv Terminal Pro v2.0</strong> | Complete Algo Trading Platform | © 2024 | 
        Last Update: {now_indian().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
    """, unsafe_allow_html=True)

# ===================== RUN APPLICATION =====================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
