import ccxt
import pandas as pd
import numpy as np
import time
import asyncio
from datetime import datetime
import logging
import os
import requests
import json
import sqlite3
import statistics
from typing import Dict, List, Optional
from dotenv import load_dotenv
from telegram import Bot
from telegram.ext import Application, CommandHandler, ContextTypes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderBookAnalyzer:
    """Order book analysis component"""

    def __init__(self):
        self.db_path = "orderbook_data.db"
        self.init_database()

        # Analysis thresholds
        self.thresholds = {
            'bid_ask_imbalance_strong': 0.25,     # 25% imbalance
            'bid_ask_imbalance_extreme': 0.5,     # 50% imbalance
            'spread_normal_bps': 2,               # 2 basis points normal
            'spread_wide_bps': 5,                 # 5 basis points wide
            'spread_extreme_bps': 10,             # 10 basis points extreme
            'large_wall_multiplier': 5,           # 5x average size
            'huge_wall_multiplier': 10,           # 10x average size
            'depth_imbalance_strong': 0.3,        # 30% depth difference
            'depth_imbalance_extreme': 0.6,       # 60% depth difference
        }

    def init_database(self):
        """Initialize SQLite database for storing order book data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orderbook_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                mid_price REAL,
                spread_bps REAL,
                bid_ask_imbalance REAL,
                bid_depth_1pct REAL,
                ask_depth_1pct REAL,
                bid_depth_5pct REAL,
                ask_depth_5pct REAL,
                large_bid_walls INTEGER,
                large_ask_walls INTEGER,
                weighted_bid_price REAL,
                weighted_ask_price REAL,
                price_skew REAL,
                liquidity_score REAL,
                market_sentiment TEXT,
                key_levels TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def get_order_book_data(self, symbol: str = 'BTC/USDT') -> Optional[Dict]:
        """Fetch detailed order book from Binance for given symbol"""
        try:
            url = "https://api.binance.com/api/v3/depth"
            api_symbol = symbol.replace('/', '')
            params = {"symbol": api_symbol, "limit": 1000}
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            if not data or 'bids' not in data or 'asks' not in data:
                return None

            bids = [[float(price), float(qty)] for price, qty in data['bids']]
            asks = [[float(price), float(qty)] for price, qty in data['asks']]
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            mid_price = (best_bid + best_ask) / 2

            return {
                'bids': bids,
                'asks': asks,
                'mid_price': mid_price,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': best_ask - best_bid,
                'spread_bps': ((best_ask - best_bid) / mid_price) * 10000,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            return None

    def calculate_market_pressure(self, bids: List[List[float]], asks: List[List[float]], mid_price: float) -> Dict:
        """Calculate various market pressure metrics"""
        pressure = {}
        # Immediate, short-term, medium-term as before...
        immediate_bid_vol = sum(qty for price, qty in bids if price >= mid_price * 0.999)
        immediate_ask_vol = sum(qty for price, qty in asks if price <= mid_price * 1.001)
        total_immediate = immediate_bid_vol + immediate_ask_vol
        if total_immediate > 0:
            pressure['immediate_imbalance'] = (immediate_bid_vol - immediate_ask_vol) / total_immediate

        short_bid_vol = sum(qty for price, qty in bids if price >= mid_price * 0.99)
        short_ask_vol = sum(qty for price, qty in asks if price <= mid_price * 1.01)
        total_short = short_bid_vol + short_ask_vol
        if total_short > 0:
            pressure['short_term_imbalance'] = (short_bid_vol - short_ask_vol) / total_short

        medium_bid_vol = sum(qty for price, qty in bids if price >= mid_price * 0.95)
        medium_ask_vol = sum(qty for price, qty in asks if price <= mid_price * 1.05)
        total_medium = medium_bid_vol + medium_ask_vol
        if total_medium > 0:
            pressure['medium_term_imbalance'] = (medium_bid_vol - medium_ask_vol) / total_medium

        pressure['bid_depth_1pct'] = short_bid_vol
        pressure['ask_depth_1pct'] = short_ask_vol
        pressure['bid_depth_5pct'] = medium_bid_vol
        pressure['ask_depth_5pct'] = medium_ask_vol

        return pressure

    def detect_significant_levels(self, bids: List[List[float]], asks: List[List[float]], mid_price: float) -> Dict:
        """Detect significant support and resistance levels"""
        bid_sizes = [qty for _, qty in bids[:100]]
        ask_sizes = [qty for _, qty in asks[:100]]
        avg_bid_size = statistics.mean(bid_sizes) if bid_sizes else 0
        avg_ask_size = statistics.mean(ask_sizes) if ask_sizes else 0

        significant_bids = []
        significant_asks = []
        for price, qty in bids:
            if qty >= avg_bid_size * self.thresholds['huge_wall_multiplier']:
                distance_pct = ((mid_price - price) / mid_price) * 100
                significant_bids.append({'price': price, 'quantity': qty, 'distance_pct': distance_pct})
        for price, qty in asks:
            if qty >= avg_ask_size * self.thresholds['huge_wall_multiplier']:
                distance_pct = ((price - mid_price) / mid_price) * 100
                significant_asks.append({'price': price, 'quantity': qty, 'distance_pct': distance_pct})

        significant_bids.sort(key=lambda x: x['distance_pct'])
        significant_asks.sort(key=lambda x: x['distance_pct'])

        def _filter_levels(levels):
            filtered = []
            for lvl in levels:
                if not any(abs(lvl['distance_pct'] - f['distance_pct']) < 0.1 for f in filtered):
                    filtered.append(lvl)
            return filtered

        significant_bids = _filter_levels(significant_bids)
        significant_asks = _filter_levels(significant_asks)

        return {
            'significant_bids': significant_bids[:5],
            'significant_asks': significant_asks[:5],
            'large_bid_walls': len(significant_bids),
            'large_ask_walls': len(significant_asks)
        }

    def classify_market_sentiment(self, analysis: Dict) -> str:
        """Classify overall market sentiment based on order book"""
        spread_bps = analysis.get('spread_bps', 0)
        imbalance = analysis.get('immediate_imbalance', 0)
        bid_walls = analysis.get('large_bid_walls', 0)
        ask_walls = analysis.get('large_ask_walls', 0)
        
        # Determine sentiment based on multiple factors
        sentiment_score = 0
        
        # Imbalance factor
        if imbalance > 0.3:
            sentiment_score += 2  # Bullish
        elif imbalance > 0.1:
            sentiment_score += 1  # Slightly bullish
        elif imbalance < -0.3:
            sentiment_score -= 2  # Bearish
        elif imbalance < -0.1:
            sentiment_score -= 1  # Slightly bearish
            
        # Wall factor
        if bid_walls > ask_walls + 1:
            sentiment_score += 1  # More support
        elif ask_walls > bid_walls + 1:
            sentiment_score -= 1  # More resistance
            
        # Spread factor (tight spreads are healthier)
        if spread_bps < 2:
            sentiment_score += 0.5
        elif spread_bps > 10:
            sentiment_score -= 0.5
            
        # Classify sentiment
        if sentiment_score >= 2:
            return "STRONGLY_BULLISH"
        elif sentiment_score >= 1:
            return "BULLISH"
        elif sentiment_score <= -2:
            return "STRONGLY_BEARISH"
        elif sentiment_score <= -1:
            return "BEARISH"
        else:
            return "NEUTRAL"

    def calculate_liquidity_score(self, analysis: Dict, symbol: str = 'BTC/USDT') -> float:
        """Calculate asset-specific liquidity health score (0-100)"""
        score = 50  # Start with neutral
        
        # Asset-specific thresholds
        mid_price = analysis.get('mid_price', 0)
        
        # Determine asset tier for different expectations
        if symbol in ['BTC/USDT', 'ETH/USDT']:
            tier = 'major'  # Highest liquidity expectations
        elif symbol in ['BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT', 'DOGE/USDT']:
            tier = 'large_alt'  # High liquidity expectations
        elif mid_price > 100:
            tier = 'mid_cap'  # Medium liquidity expectations
        else:
            tier = 'small_cap'  # Lower liquidity expectations
            
        # Dynamic spread thresholds based on asset tier
        spread_bps = analysis.get('spread_bps', 0)
        if tier == 'major':
            if spread_bps < 1: score += 25
            elif spread_bps < 3: score += 15
            elif spread_bps < 8: score += 5
            elif spread_bps > 20: score -= 25
            elif spread_bps > 10: score -= 15
        elif tier == 'large_alt':
            if spread_bps < 2: score += 20
            elif spread_bps < 5: score += 10
            elif spread_bps > 25: score -= 20
            elif spread_bps > 15: score -= 10
        else:  # mid_cap and small_cap
            if spread_bps < 5: score += 15
            elif spread_bps < 10: score += 5
            elif spread_bps > 50: score -= 20
            elif spread_bps > 30: score -= 10
            
        # Dynamic depth thresholds based on price and tier
        bid_depth = analysis.get('bid_depth_1pct', 0)
        ask_depth = analysis.get('ask_depth_1pct', 0)
        total_depth = bid_depth + ask_depth
        
        # Convert to USD value for consistent comparison
        total_depth_usd = total_depth * mid_price if mid_price > 0 else total_depth
        
        if tier == 'major':
            if total_depth_usd > 5000000: score += 25
            elif total_depth_usd > 2000000: score += 15
            elif total_depth_usd > 500000: score += 5
            elif total_depth_usd < 100000: score -= 25
        elif tier == 'large_alt':
            if total_depth_usd > 2000000: score += 20
            elif total_depth_usd > 500000: score += 10
            elif total_depth_usd < 50000: score -= 20
        else:  # mid_cap and small_cap
            if total_depth_usd > 500000: score += 15
            elif total_depth_usd > 100000: score += 5
            elif total_depth_usd < 10000: score -= 15
            
        # Balance component (consistent across tiers)
        imbalance = abs(analysis.get('immediate_imbalance', 0))
        if imbalance < 0.1:
            score += 10
        elif imbalance < 0.3:
            score += 5
        elif imbalance > 0.6:
            score -= 15
            
        # Support/resistance levels (adjusted by tier)
        total_walls = analysis.get('large_bid_walls', 0) + analysis.get('large_ask_walls', 0)
        if tier == 'major':
            if total_walls > 5: score += 5
            elif total_walls == 0: score -= 10
        else:
            if total_walls > 2: score += 5
            elif total_walls == 0: score -= 5
            
        return max(0, min(100, score))

    def analyze_order_book(self, symbol: str = 'BTC/USDT') -> Optional[Dict]:
        """Run complete order book analysis for given symbol"""
        order_book_data = self.get_order_book_data(symbol)
        if not order_book_data:
            return None

        bids = order_book_data['bids']
        asks = order_book_data['asks']
        mid_price = order_book_data['mid_price']

        analysis = {
            'timestamp': order_book_data['timestamp'],
            'mid_price': mid_price,
            'spread': order_book_data['spread'],
            'spread_bps': order_book_data['spread_bps']
        }
        analysis.update(self.calculate_market_pressure(bids, asks, mid_price))
        analysis.update(self.detect_significant_levels(bids, asks, mid_price))
        # ... weighted prices, skew, sentiment, liquidity score ...
        analysis['market_sentiment'] = self.classify_market_sentiment(analysis)
        analysis['liquidity_score'] = self.calculate_liquidity_score(analysis, symbol)

        return analysis

    def generate_report(self, analysis: Dict) -> str:
        """Generate order book report"""
        report = f"📊 <b>Order Book Analysis</b>\n"
        # Use more decimals for low-price coins
        mid_price = analysis['mid_price']
        decimals = 6 if mid_price < 1 else 4 if mid_price < 100 else 2
        report += f"💰 Mid Price: ${mid_price:.{decimals}f}\n"
        report += f"📈 Spread: {analysis['spread_bps']:.2f} bps\n\n"
        
        # Market pressure
        if 'immediate_imbalance' in analysis:
            imbalance = analysis['immediate_imbalance']
            if abs(imbalance) > 0.3:
                direction = "BULLISH" if imbalance > 0 else "BEARISH"
                report += f"⚡ Market Pressure: <b>{direction}</b> ({imbalance:.1%})\n"
            else:
                report += f"⚖️ Market Pressure: BALANCED ({imbalance:.1%})\n"
        
        # Significant levels with prices
        significant_bids = analysis.get('significant_bids', [])
        significant_asks = analysis.get('significant_asks', [])
        
        if significant_bids:
            report += f"\n🟢 <b>Key Support Levels:</b>\n"
            for bid in significant_bids[:3]:  # Show top 3
                # Use more decimals for low-price coins
                decimals = 6 if bid['price'] < 1 else 4 if bid['price'] < 100 else 2
                report += f"  ${bid['price']:.{decimals}f} ({bid['distance_pct']:.2f}% below)\n"
                
        if significant_asks:
            report += f"\n🔴 <b>Key Resistance Levels:</b>\n"
            for ask in significant_asks[:3]:  # Show top 3
                # Use more decimals for low-price coins
                decimals = 6 if ask['price'] < 1 else 4 if ask['price'] < 100 else 2
                report += f"  ${ask['price']:.{decimals}f} ({ask['distance_pct']:.2f}% above)\n"
        
        # Wall counts
        if analysis.get('large_bid_walls', 0) > 0 or analysis.get('large_ask_walls', 0) > 0:
            report += f"\n📊 Large Walls: {analysis.get('large_bid_walls', 0)} bids, {analysis.get('large_ask_walls', 0)} asks\n"
            
        # Liquidity and sentiment
        report += f"\n💧 Liquidity Score: {analysis.get('liquidity_score', 0):.0f}/100\n"
        report += f"🎯 Sentiment: <b>{analysis.get('market_sentiment', 'NEUTRAL')}</b>"
        
        return report

    def store_analysis(self, analysis: Dict):
        """Store analysis in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO orderbook_analysis (
                    mid_price, spread_bps, bid_ask_imbalance,
                    bid_depth_1pct, ask_depth_1pct, bid_depth_5pct, ask_depth_5pct,
                    large_bid_walls, large_ask_walls, liquidity_score, market_sentiment
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis.get('mid_price', 0),
                analysis.get('spread_bps', 0),
                analysis.get('immediate_imbalance', 0),
                analysis.get('bid_depth_1pct', 0),
                analysis.get('ask_depth_1pct', 0),
                analysis.get('bid_depth_5pct', 0),
                analysis.get('ask_depth_5pct', 0),
                analysis.get('large_bid_walls', 0),
                analysis.get('large_ask_walls', 0),
                analysis.get('liquidity_score', 0),
                analysis.get('market_sentiment', 'NEUTRAL')
            ))
            
            conn.commit()
            conn.close()
            logger.info("Order book analysis stored in database")
            
        except Exception as e:
            logger.error(f"Failed to store analysis: {e}")


class EnhancedTelegramBot:
    def __init__(self, bot_token, chat_id):
        self.bot = Bot(token=bot_token)
        self.chat_id = chat_id
        self.exchange = ccxt.binance()
        self.symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
            'HYPE/USDT', 'PENGU/USDT', 'XRP/USDT', 'MKR/USDT', 'AAVE/USDT',
            'DOGE/USDT', 'APT/USDT', 'XLM/USDT', 'QNT/USDT', 'SUI/USDT',
            'WIF/USDT', 'TON/USDT', 'KAITO/USDT', 'LINK/USDT'
        ]
        self.lookback = 20
        self.volume_threshold = 1.5
        self.running = False
        self.orderbook_running = False

        self.orderbook_analyzer = OrderBookAnalyzer()
        self.last_macd_hist = None
        self.last_vwap_price = None

    def get_ohlcv(self, symbol, timeframe='1h', limit=50):
        """Fetch OHLCV data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None

    def calculate_levels(self, df):
        """Calculate indicators"""
        # Calculate moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Calculate EMA
        df['ema_20'] = df['close'].ewm(span=20).mean()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Calculate volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df

    def detect_price_action_patterns(self, df):
        """Detect significant price action patterns"""
        if len(df) < 3:
            return []
            
        current = df.iloc[-1]
        previous = df.iloc[-2]
        prev2 = df.iloc[-3]
        
        patterns = []
        
        # Calculate candle properties
        current_body = abs(current['close'] - current['open'])
        current_range = current['high'] - current['low']
        current_upper_shadow = current['high'] - max(current['open'], current['close'])
        current_lower_shadow = min(current['open'], current['close']) - current['low']
        
        previous_body = abs(previous['close'] - previous['open'])
        
        # Doji pattern - small body relative to range
        if current_body < (current_range * 0.1) and current_range > 0:
            patterns.append("Doji")
            
        # Hammer pattern - small upper shadow, long lower shadow, small body
        if (current_lower_shadow > current_body * 2 and 
            current_upper_shadow < current_body * 0.5 and
            current_range > 0):
            patterns.append("Hammer")
            
        # Shooting star - small lower shadow, long upper shadow, small body  
        if (current_upper_shadow > current_body * 2 and
            current_lower_shadow < current_body * 0.5 and
            current_range > 0):
            patterns.append("Shooting Star")
            
        # Bullish engulfing - current green candle completely engulfs previous red candle
        if (current['close'] > current['open'] and  # Current is green
            previous['close'] < previous['open'] and  # Previous is red
            current['open'] < previous['close'] and  # Current opens below previous close
            current['close'] > previous['open']):    # Current closes above previous open
            patterns.append("Bullish Engulfing")
            
        # Bearish engulfing - current red candle completely engulfs previous green candle
        if (current['close'] < current['open'] and  # Current is red
            previous['close'] > previous['open'] and  # Previous is green
            current['open'] > previous['close'] and  # Current opens above previous close
            current['close'] < previous['open']):    # Current closes below previous open
            patterns.append("Bearish Engulfing")
            
        # Three white soldiers - three consecutive green candles with higher closes
        if (len(df) >= 3 and
            current['close'] > current['open'] and
            previous['close'] > previous['open'] and  
            prev2['close'] > prev2['open'] and
            current['close'] > previous['close'] > prev2['close']):
            patterns.append("Three White Soldiers")
            
        # Three black crows - three consecutive red candles with lower closes
        if (len(df) >= 3 and
            current['close'] < current['open'] and
            previous['close'] < previous['open'] and
            prev2['close'] < prev2['open'] and  
            current['close'] < previous['close'] < prev2['close']):
            patterns.append("Three Black Crows")
            
        return patterns

    def get_higher_timeframe_trend(self, symbol, current_timeframe='1h'):
        """Get trend direction from higher timeframe for confirmation"""
        try:
            # Map to higher timeframes
            tf_map = {'1h': '4h', '4h': '1d', '1d': '1w'}
            higher_tf = tf_map.get(current_timeframe, '4h')
            
            # Get higher timeframe data
            higher_df = self.get_ohlcv(symbol, timeframe=higher_tf, limit=50)
            if higher_df is None or len(higher_df) < 20:
                return None
                
            # Calculate trend indicators on higher timeframe
            higher_df = self.calculate_levels(higher_df)
            current_higher = higher_df.iloc[-1]
            
            # Determine trend based on multiple factors
            trend_score = 0
            
            # EMA vs SMA comparison
            if current_higher['ema_20'] > current_higher['sma_50']:
                trend_score += 1
            else:
                trend_score -= 1
                
            # Price vs moving averages
            if current_higher['close'] > current_higher['ema_20']:
                trend_score += 1
            else:
                trend_score -= 1
                
            # Recent price action (last 5 candles trend)
            if len(higher_df) >= 5:
                recent_closes = higher_df['close'].tail(5)
                if recent_closes.iloc[-1] > recent_closes.iloc[0]:  # Overall rising
                    trend_score += 1
                else:
                    trend_score -= 1
            
            # Classify trend
            if trend_score >= 2:
                return "bullish"
            elif trend_score <= -2:
                return "bearish"
            else:
                return "neutral"
                
        except Exception as e:
            logger.error(f"Error getting higher timeframe trend for {symbol}: {e}")
            return None

    def check_breakout(self, df, symbol='BTC/USDT'):
        """Check for breakout conditions"""
        if len(df) < self.lookback:
            return None, None, None
            
        current = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else current
        
        # Calculate recent highs/lows (excluding current candle to avoid self-comparison)
        recent_data = df.iloc[:-1]  # Exclude current candle
        recent_high = recent_data['high'].rolling(window=self.lookback).max().iloc[-1]
        recent_low = recent_data['low'].rolling(window=self.lookback).min().iloc[-1]
        
        # Check for price breakouts above recent highs/lows
        resistance_break = (
            current['close'] > recent_high and
            current['high'] > recent_high and
            current['volume_ratio'] > self.volume_threshold
        )
        
        support_break = (
            current['close'] < recent_low and
            current['low'] < recent_low and
            current['volume_ratio'] > self.volume_threshold
        )
        
        # RSI filtering - avoid breakouts in extreme conditions
        rsi_allows_bullish = current['rsi'] < 70  # Not overbought
        rsi_allows_bearish = current['rsi'] > 30  # Not oversold
        
        # Price action confirmation - ensure strong candle close
        bullish_price_action = (
            current['close'] > current['open'] and  # Green candle
            (current['close'] - current['open']) / current['open'] > 0.01  # At least 1% move
        )
        
        bearish_price_action = (
            current['close'] < current['open'] and  # Red candle  
            (current['open'] - current['close']) / current['open'] > 0.01  # At least 1% move
        )
        
        # Enhanced volume confirmation
        strong_volume = current['volume_ratio'] > 2.0
        moderate_volume = current['volume_ratio'] > self.volume_threshold
        
        # Bollinger Band confirmation (only as secondary signal)
        bb_confirms_bullish = current['close'] > current['bb_upper']
        bb_confirms_bearish = current['close'] < current['bb_lower']
        
        # Price action pattern analysis
        patterns = self.detect_price_action_patterns(df)
        bullish_patterns = [p for p in patterns if p in ["Hammer", "Bullish Engulfing", "Three White Soldiers"]]
        bearish_patterns = [p for p in patterns if p in ["Shooting Star", "Bearish Engulfing", "Three Black Crows"]]
        
        # Multi-timeframe trend confirmation
        higher_tf_trend = self.get_higher_timeframe_trend(symbol)
        trend_aligns_bullish = higher_tf_trend in ["bullish", "neutral"]  # Allow neutral for sideways breakouts
        trend_aligns_bearish = higher_tf_trend in ["bearish", "neutral"]
        
        # Main breakout logic - focus on price structure with multi-timeframe confirmation
        bullish_breakout = (
            resistance_break and 
            rsi_allows_bullish and
            bullish_price_action and
            moderate_volume and
            trend_aligns_bullish  # Higher timeframe must align or be neutral
        )
        
        bearish_breakout = (
            support_break and
            rsi_allows_bearish and  
            bearish_price_action and
            moderate_volume and
            trend_aligns_bearish  # Higher timeframe must align or be neutral
        )
        
        # Return results with proper context
        if bullish_breakout:
            strength = "STRONG" if strong_volume and (bb_confirms_bullish or bullish_patterns) else "MODERATE"
            confirmations = []
            if bb_confirms_bullish:
                confirmations.append("BB+")
            if strong_volume:
                confirmations.append("Vol+")
            if bullish_patterns:
                confirmations.extend(bullish_patterns)
            if higher_tf_trend == "bullish":
                confirmations.append("4H↗")
            
            confirmation_text = f" ({'+'.join(confirmations)})" if confirmations else ""
            analysis = f"High Break{confirmation_text} | RSI: {current['rsi']:.1f} | Vol: {current['volume_ratio']:.1f}x | Broke: ${recent_high:.4f}"
            return "bullish", strength, analysis
            
        elif bearish_breakout:
            strength = "STRONG" if strong_volume and (bb_confirms_bearish or bearish_patterns) else "MODERATE" 
            confirmations = []
            if bb_confirms_bearish:
                confirmations.append("BB+")
            if strong_volume:
                confirmations.append("Vol+")
            if bearish_patterns:
                confirmations.extend(bearish_patterns)
            if higher_tf_trend == "bearish":
                confirmations.append("4H↘")
                
            confirmation_text = f" ({'+'.join(confirmations)})" if confirmations else ""
            analysis = f"Low Break{confirmation_text} | RSI: {current['rsi']:.1f} | Vol: {current['volume_ratio']:.1f}x | Broke: ${recent_low:.4f}"
            return "bearish", strength, analysis
            
        return None, None, None

    async def send_alert(self, message):
        """Send alert to Telegram"""
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML'
            )
            logger.info(f"Alert sent: {message[:50]}...")
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    async def scan_markets(self):
        """Scan all markets for breakouts"""
        alerts = []
        for symbol in self.symbols:
            try:
                df = self.get_ohlcv(symbol)
                if df is None or len(df) < self.lookback:
                    continue
                
                df = self.calculate_levels(df)
                direction, strength, analysis = self.check_breakout(df, symbol)
                
                if direction and strength:
                    message = f"🚀 <b>{symbol} {direction.upper()} BREAKOUT</b>\n"
                    message += f"Strength: {strength}\n"
                    message += f"Price: ${df['close'].iloc[-1]:.4f}\n"
                    if analysis:
                        message += f"Analysis: {analysis}"
                    
                    await self.send_alert(message)
                    alerts.append(symbol)
                    
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                
        return alerts

    async def order_book_monitor(self):
        """Monitor order book every 30 minutes"""
        while self.orderbook_running:
            try:
                logger.info("Running order book analysis for BTC/USDT...")
                analysis = self.orderbook_analyzer.analyze_order_book('BTC/USDT')
                
                if analysis:
                    report = self.orderbook_analyzer.generate_report(analysis)
                    await self.send_alert(f"📊 <b>Order Book Analysis - BTC/USDT</b>\n\n{report}")
                    self.orderbook_analyzer.store_analysis(analysis)
                    logger.info("Order book analysis completed and sent")
                else:
                    logger.warning("Failed to get order book analysis")
                    
            except Exception as e:
                logger.error(f"Order book monitoring error: {e}")
                
            await asyncio.sleep(1800)  # 30 minutes

    async def start_monitoring(self):
        """Start both breakout and order book monitoring"""
        if self.running:
            logger.info("Monitoring already running")
            return
            
        self.running = True
        self.orderbook_running = True
        
        logger.info("Starting enhanced monitoring...")
        
        # Start breakout monitoring task
        breakout_task = asyncio.create_task(self.breakout_monitor())
        logger.info("Breakout monitoring started")
        
        # Start order book monitoring task
        orderbook_task = asyncio.create_task(self.order_book_monitor())
        logger.info("Order book monitoring started")
        
        # Send startup notification
        await self.send_alert(
            "🤖 <b>Enhanced Crypto Bot Started</b>\n\n"
            "🚀 Breakout monitoring: ACTIVE\n"
            "📊 Order book analysis: ACTIVE\n\n"
            f"Monitoring {len(self.symbols)} symbols"
        )
        
        return breakout_task, orderbook_task

    async def breakout_monitor(self):
        """Monitor breakouts at candle close"""
        # Timeframe
        tf = '1h'
        sec_map = {'m':60,'h':3600,'d':86400}
        unit = tf[-1]; val = int(tf[:-1])
        seconds_per_candle = sec_map.get(unit,3600)*val

        while self.running:
            try:
                now_ts = time.time()
                next_close_ts = (int(now_ts//seconds_per_candle)+1)*seconds_per_candle
                wait = next_close_ts - now_ts
                logger.info(f"Waiting {wait:.1f}s until next candle close ({tf})")
                await asyncio.sleep(wait)

                logger.info("Scanning markets for breakouts at candle close...")
                alerts = await self.scan_markets()
                if alerts:
                    logger.info(f"Sent {len(alerts)} breakout alerts")
            except Exception as e:
                logger.error(f"Breakout monitoring error: {e}")
                await asyncio.sleep(60)

    def stop_monitoring(self):
        self.running = False
        self.orderbook_running = False


# Bot command handlers

async def start_command(update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    await update.message.reply_text(
        "🤖 <b>Enhanced Crypto Bot</b>\n\n"
        "Features:\n"
        "🚀 Breakout alerts for 19 symbols\n"
        "📊 BTC order book analysis every 30min\n"
        "\nCommands:\n"
        "/start - Show this message\n"
        "/status - Check bot status\n"
        "/symbols - Show monitored symbols\n"
        "/orderbook - Get instant BTC/USDT order book analysis\n"
        "/analyze SYMBOL - Analyze order book for specific symbol\n"
        "/analyze all - Analyze all 19 watchlist symbols\n"
        "/stop - Stop monitoring",
        parse_mode='HTML'
    )

async def status_command(update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command"""
    bot_instance = context.bot_data.get('bot_instance')
    if bot_instance:
        breakout_status = "🟢 Running" if bot_instance.running else "🔴 Stopped"
        orderbook_status = "🟢 Running" if bot_instance.orderbook_running else "🔴 Stopped"
        await update.message.reply_text(
            f"<b>Bot Status:</b>\n"
            f"🚀 Breakouts: {breakout_status}\n"
            f"📊 Order Book: {orderbook_status}",
            parse_mode='HTML'
        )
    else:
        await update.message.reply_text("<b>Bot Status:</b> 🔴 Not initialized", parse_mode='HTML')

async def symbols_command(update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /symbols command"""
    symbols = context.bot_data['bot_instance'].symbols
    symbol_list = "\n".join(f"• {s}" for s in symbols)
    await update.message.reply_text(
        f"<b>Monitored Symbols ({len(symbols)}):</b>\n{symbol_list}\n\n"
        f"<b>Order Book Analysis:</b>\n• BTC/USDT (every 30min)",
        parse_mode='HTML'
    )

async def orderbook_command(update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /orderbook command - instant analysis for BTC/USDT"""
    bot_instance = context.bot_data.get('bot_instance')
    if not bot_instance:
        await update.message.reply_text("❌ Bot not initialized", parse_mode='HTML')
        return
    await update.message.reply_text("📊 Analyzing BTC/USDT order book...", parse_mode='HTML')
    analysis = bot_instance.orderbook_analyzer.analyze_order_book('BTC/USDT')
    if analysis:
        report = bot_instance.orderbook_analyzer.generate_report(analysis)
        await update.message.reply_text(report, parse_mode='HTML')
    else:
        await update.message.reply_text("❌ Failed to fetch order book data for BTC/USDT", parse_mode='HTML')

async def analyze_command(update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /analyze <symbol> or /analyze all command for order book analysis"""
    bot_instance = context.bot_data.get('bot_instance')
    if not bot_instance:
        await update.message.reply_text("❌ Bot not initialized", parse_mode='HTML')
        return

    args = context.args
    if not args:
        await update.message.reply_text(
            "Usage: /analyze SYMBOL (e.g. BTC/USDT) or /analyze all", parse_mode='HTML'
        )
        return

    # Handle /analyze all command
    if args[0].lower() == 'all':
        await update.message.reply_text("📊 Analyzing all watchlist symbols... This may take a moment.", parse_mode='HTML')
        
        results = []
        for symbol in bot_instance.symbols:
            try:
                analysis = bot_instance.orderbook_analyzer.analyze_order_book(symbol)
                if analysis:
                    # Create condensed report for multiple symbols
                    mid_price = analysis['mid_price']
                    decimals = 6 if mid_price < 1 else 4 if mid_price < 100 else 2
                    
                    sentiment = analysis.get('market_sentiment', 'NEUTRAL')
                    liquidity = analysis.get('liquidity_score', 0)
                    spread = analysis.get('spread_bps', 0)
                    imbalance = analysis.get('immediate_imbalance', 0)
                    
                    # Get key levels
                    significant_bids = analysis.get('significant_bids', [])
                    significant_asks = analysis.get('significant_asks', [])
                    
                    # Use emojis for quick visual scanning
                    sentiment_emoji = "🟢" if "BULLISH" in sentiment else "🔴" if "BEARISH" in sentiment else "⚪"
                    
                    # Format support/resistance levels
                    support_text = ""
                    resistance_text = ""
                    
                    if significant_bids:
                        closest_support = significant_bids[0]  # Closest to current price
                        support_text = f" | 🟢${closest_support['price']:.{decimals}f}({closest_support['distance_pct']:.1f}%)"
                    
                    if significant_asks:
                        closest_resistance = significant_asks[0]  # Closest to current price
                        resistance_text = f" | 🔴${closest_resistance['price']:.{decimals}f}({closest_resistance['distance_pct']:.1f}%)"
                    
                    results.append(
                        f"{sentiment_emoji} <b>{symbol}</b>: ${mid_price:.{decimals}f} | "
                        f"L:{liquidity:.0f} | S:{spread:.1f}bps | I:{imbalance:.0%}"
                        f"{support_text}{resistance_text}"
                    )
                else:
                    results.append(f"❌ <b>{symbol}</b>: Failed to fetch data")
            except Exception as e:
                results.append(f"⚠️ <b>{symbol}</b>: Error - {str(e)[:30]}")
        
        # Send results in chunks to avoid message length limits
        chunk_size = 10
        for i in range(0, len(results), chunk_size):
            chunk = results[i:i+chunk_size]
            header = f"📊 <b>Order Book Analysis ({i+1}-{min(i+chunk_size, len(results))} of {len(results)})</b>\n\n"
            message = header + "\n".join(chunk)
            message += "\n\n<i>L=Liquidity, S=Spread, I=Imbalance, 🟢=Support, 🔴=Resistance</i>"
            await update.message.reply_text(message, parse_mode='HTML')
        
        return

    # Handle single symbol analysis
    symbol = args[0].upper()
    # Normalize symbol if given without slash (e.g., ETHUSDT → ETH/USDT)
    if '/' not in symbol and symbol.endswith('USDT'):
        symbol = f"{symbol[:-4]}/USDT"
    await update.message.reply_text(f"📊 Analyzing {symbol} order book...", parse_mode='HTML')
    analysis = bot_instance.orderbook_analyzer.analyze_order_book(symbol)
    if analysis:
        report = bot_instance.orderbook_analyzer.generate_report(analysis)
        await update.message.reply_text(report, parse_mode='HTML')
    else:
        await update.message.reply_text(
            f"❌ Failed to fetch order book data for {symbol}", parse_mode='HTML'
        )

async def stop_command(update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /stop command"""
    bot_instance = context.bot_data.get('bot_instance')
    if bot_instance:
        bot_instance.stop_monitoring()
    await update.message.reply_text("🛑 <b>All monitoring stopped</b>", parse_mode='HTML')

def main():
    """Main function to run the enhanced bot"""
    load_dotenv()
    BOT_TOKEN = os.getenv("BOT_TOKEN")
    CHAT_ID = os.getenv("CHAT_ID")

    bot = EnhancedTelegramBot(BOT_TOKEN, CHAT_ID)

    async def post_init(application):
        logger.info("Bot initialization complete - starting enhanced monitoring")
        bot_instance = application.bot_data['bot_instance']
        await bot_instance.start_monitoring()

    application = Application.builder().token(BOT_TOKEN).post_init(post_init).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("symbols", symbols_command))
    application.add_handler(CommandHandler("orderbook", orderbook_command))
    application.add_handler(CommandHandler("analyze", analyze_command))
    application.add_handler(CommandHandler("stop", stop_command))

    application.bot_data['bot_instance'] = bot

    logger.info("Starting enhanced Telegram bot...")
    application.run_polling()

if __name__ == "__main__":
    main()