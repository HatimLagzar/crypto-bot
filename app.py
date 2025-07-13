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
            'large_wall_multiplier': 3,           # 3x average size
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
    
    def get_order_book_data(self) -> Optional[Dict]:
        """Fetch detailed BTC order book from Binance"""
        try:
            url = "https://api.binance.com/api/v3/depth"
            params = {"symbol": "BTCUSDT", "limit": 1000}
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if not data or 'bids' not in data or 'asks' not in data:
                return None
            
            # Convert to float
            bids = [[float(price), float(qty)] for price, qty in data['bids']]
            asks = [[float(price), float(qty)] for price, qty in data['asks']]
            
            # Get current price
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
            logger.error(f"Error fetching order book: {e}")
            return None
    
    def calculate_market_pressure(self, bids: List[List[float]], asks: List[List[float]], mid_price: float) -> Dict:
        """Calculate various market pressure metrics"""
        pressure = {}
        
        # 1. Immediate pressure (within 0.1%)
        immediate_bid_vol = sum(qty for price, qty in bids if price >= mid_price * 0.999)
        immediate_ask_vol = sum(qty for price, qty in asks if price <= mid_price * 1.001)
        total_immediate = immediate_bid_vol + immediate_ask_vol
        
        if total_immediate > 0:
            pressure['immediate_imbalance'] = (immediate_bid_vol - immediate_ask_vol) / total_immediate
        
        # 2. Short-term pressure (within 1%)
        short_bid_vol = sum(qty for price, qty in bids if price >= mid_price * 0.99)
        short_ask_vol = sum(qty for price, qty in asks if price <= mid_price * 1.01)
        total_short = short_bid_vol + short_ask_vol
        
        if total_short > 0:
            pressure['short_term_imbalance'] = (short_bid_vol - short_ask_vol) / total_short
        
        # 3. Medium-term pressure (within 5%)
        medium_bid_vol = sum(qty for price, qty in bids if price >= mid_price * 0.95)
        medium_ask_vol = sum(qty for price, qty in asks if price <= mid_price * 1.05)
        total_medium = medium_bid_vol + medium_ask_vol
        
        if total_medium > 0:
            pressure['medium_term_imbalance'] = (medium_bid_vol - medium_ask_vol) / total_medium
        
        # Store volumes for liquidity analysis
        pressure['bid_depth_1pct'] = short_bid_vol
        pressure['ask_depth_1pct'] = short_ask_vol
        pressure['bid_depth_5pct'] = medium_bid_vol
        pressure['ask_depth_5pct'] = medium_ask_vol
        
        return pressure
    
    def detect_significant_levels(self, bids: List[List[float]], asks: List[List[float]], mid_price: float) -> Dict:
        """Detect significant support and resistance levels"""
        # Calculate average order sizes
        bid_sizes = [qty for _, qty in bids[:100]]
        ask_sizes = [qty for _, qty in asks[:100]]
        
        avg_bid_size = statistics.mean(bid_sizes) if bid_sizes else 0
        avg_ask_size = statistics.mean(ask_sizes) if ask_sizes else 0
        
        # Find significant levels
        significant_bids = []
        significant_asks = []
        
        for price, qty in bids:
            if qty >= avg_bid_size * self.thresholds['large_wall_multiplier']:
                distance_pct = ((mid_price - price) / mid_price) * 100
                significant_bids.append({
                    'price': price,
                    'quantity': qty,
                    'distance_pct': distance_pct,
                    'size_ratio': qty / avg_bid_size if avg_bid_size > 0 else 0
                })
        
        for price, qty in asks:
            if qty >= avg_ask_size * self.thresholds['large_wall_multiplier']:
                distance_pct = ((price - mid_price) / mid_price) * 100
                significant_asks.append({
                    'price': price,
                    'quantity': qty,
                    'distance_pct': distance_pct,
                    'size_ratio': qty / avg_ask_size if avg_ask_size > 0 else 0
                })
        
        # Sort by proximity to current price
        significant_bids.sort(key=lambda x: x['distance_pct'])
        significant_asks.sort(key=lambda x: x['distance_pct'])
        
        return {
            'significant_bids': significant_bids[:5],  # Top 5 closest
            'significant_asks': significant_asks[:5],
            'large_bid_walls': len(significant_bids),
            'large_ask_walls': len(significant_asks)
        }
    
    def classify_market_sentiment(self, analysis: Dict) -> str:
        """Classify overall market sentiment based on order book"""
        sentiment_scores = []
        
        # Immediate imbalance weight
        if 'immediate_imbalance' in analysis:
            imbalance = analysis['immediate_imbalance']
            if abs(imbalance) > self.thresholds['bid_ask_imbalance_extreme']:
                sentiment_scores.append(3 if imbalance > 0 else -3)
            elif abs(imbalance) > self.thresholds['bid_ask_imbalance_strong']:
                sentiment_scores.append(2 if imbalance > 0 else -2)
            else:
                sentiment_scores.append(1 if imbalance > 0 else -1)
        
        # Spread analysis
        if 'spread_bps' in analysis:
            spread = analysis['spread_bps']
            if spread > self.thresholds['spread_extreme_bps']:
                sentiment_scores.append(-2)  # Wide spread = uncertainty
            elif spread > self.thresholds['spread_wide_bps']:
                sentiment_scores.append(-1)
        
        # Wall analysis
        bid_walls = analysis.get('large_bid_walls', 0)
        ask_walls = analysis.get('large_ask_walls', 0)
        
        if bid_walls > ask_walls + 2:
            sentiment_scores.append(2)  # Strong support
        elif ask_walls > bid_walls + 2:
            sentiment_scores.append(-2)  # Strong resistance
        
        # Calculate final sentiment
        if sentiment_scores:
            avg_score = sum(sentiment_scores) / len(sentiment_scores)
            if avg_score >= 2:
                return "STRONGLY_BULLISH"
            elif avg_score >= 1:
                return "BULLISH"
            elif avg_score >= 0.5:
                return "SLIGHTLY_BULLISH"
            elif avg_score <= -2:
                return "STRONGLY_BEARISH"
            elif avg_score <= -1:
                return "BEARISH"
            elif avg_score <= -0.5:
                return "SLIGHTLY_BEARISH"
            else:
                return "NEUTRAL"
        
        return "NEUTRAL"
    
    def calculate_liquidity_score(self, analysis: Dict) -> float:
        """Calculate overall liquidity health score (0-100)"""
        score = 50  # Start neutral
        
        # Spread component (30% weight)
        if 'spread_bps' in analysis:
            spread = analysis['spread_bps']
            if spread <= self.thresholds['spread_normal_bps']:
                score += 15
            elif spread <= self.thresholds['spread_wide_bps']:
                score += 5
            else:
                score -= 10
        
        # Depth component (40% weight)
        if 'bid_depth_5pct' in analysis and 'ask_depth_5pct' in analysis:
            total_depth = analysis['bid_depth_5pct'] + analysis['ask_depth_5pct']
            if total_depth > 1000:  # High liquidity
                score += 20
            elif total_depth > 500:
                score += 10
            elif total_depth < 100:
                score -= 15
        
        # Balance component (30% weight)
        if 'medium_term_imbalance' in analysis:
            imbalance = abs(analysis['medium_term_imbalance'])
            if imbalance < 0.1:  # Well balanced
                score += 15
            elif imbalance < 0.3:
                score += 5
            else:
                score -= 10
        
        return max(0, min(100, score))
    
    def analyze_order_book(self) -> Optional[Dict]:
        """Run complete order book analysis"""
        order_book_data = self.get_order_book_data()
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
        
        # Market pressure analysis
        pressure = self.calculate_market_pressure(bids, asks, mid_price)
        analysis.update(pressure)
        
        # Significant levels
        levels = self.detect_significant_levels(bids, asks, mid_price)
        analysis.update(levels)
        
        # Calculate weighted prices
        bid_weights = [price * qty for price, qty in bids[:100]]
        bid_volumes = [qty for _, qty in bids[:100]]
        ask_weights = [price * qty for price, qty in asks[:100]]
        ask_volumes = [qty for _, qty in asks[:100]]
        
        if sum(bid_volumes) > 0:
            analysis['weighted_bid_price'] = sum(bid_weights) / sum(bid_volumes)
        if sum(ask_volumes) > 0:
            analysis['weighted_ask_price'] = sum(ask_weights) / sum(ask_volumes)
        
        # Price skew
        if 'weighted_bid_price' in analysis and 'weighted_ask_price' in analysis:
            weighted_mid = (analysis['weighted_bid_price'] + analysis['weighted_ask_price']) / 2
            analysis['price_skew'] = (mid_price - weighted_mid) / mid_price
        
        # Market sentiment and liquidity score
        analysis['market_sentiment'] = self.classify_market_sentiment(analysis)
        analysis['liquidity_score'] = self.calculate_liquidity_score(analysis)
        
        return analysis
    
    def generate_report(self, analysis: Dict) -> str:
        """Generate order book report"""
        timestamp = analysis['timestamp'].strftime('%H:%M:%S')
        mid_price = analysis['mid_price']
        
        report = f"""🔍 <b>BTC ORDER BOOK ANALYSIS</b>
⏰ {timestamp}
💰 Price: ${mid_price:,.2f}

📊 <b>SENTIMENT: {analysis['market_sentiment']}</b>
💧 Liquidity Score: {analysis.get('liquidity_score', 0):.1f}/100

🔸 <b>SPREAD</b>
• ${analysis['spread']:.2f} ({analysis['spread_bps']:.1f} bps)
• {"🟢 Tight" if analysis['spread_bps'] <= 2 else "🟡 Normal" if analysis['spread_bps'] <= 5 else "🔴 Wide"}

⚖️ <b>PRESSURE</b>"""
        
        # Pressure analysis
        if 'immediate_imbalance' in analysis:
            imm_imb = analysis['immediate_imbalance']
            emoji = "🟢" if imm_imb > 0.3 else "🔴" if imm_imb < -0.3 else "🟡"
            report += f"\n• Immediate: {emoji} {imm_imb:+.1%}"
        
        if 'short_term_imbalance' in analysis:
            short_imb = analysis['short_term_imbalance']
            emoji = "🟢" if short_imb > 0.2 else "🔴" if short_imb < -0.2 else "🟡"
            report += f"\n• Short-term: {emoji} {short_imb:+.1%}"
        
        if 'medium_term_imbalance' in analysis:
            med_imb = analysis['medium_term_imbalance']
            emoji = "🟢" if med_imb > 0.2 else "🔴" if med_imb < -0.2 else "🟡"
            report += f"\n• Medium-term: {emoji} {med_imb:+.1%}"
        
        # Significant levels
        significant_bids = analysis.get('significant_bids', [])[:3]
        significant_asks = analysis.get('significant_asks', [])[:3]
        
        if significant_bids or significant_asks:
            report += f"\n\n🧱 <b>KEY LEVELS</b>"
            
            if significant_bids:
                report += f"\n📗 <b>Support:</b>"
                for bid in significant_bids:
                    report += f"\n  ${bid['price']:,.0f} (-{bid['distance_pct']:.2f}%)"
            
            if significant_asks:
                report += f"\n📕 <b>Resistance:</b>"
                for ask in significant_asks:
                    report += f"\n  ${ask['price']:,.0f} (+{ask['distance_pct']:.2f}%)"
        
        # Liquidity info
        if 'bid_depth_1pct' in analysis and 'ask_depth_1pct' in analysis:
            report += f"\n\n💧 <b>DEPTH (1%)</b>\n• {analysis['bid_depth_1pct']:.0f} / {analysis['ask_depth_1pct']:.0f} BTC"
        
        return report
    
    def store_analysis(self, analysis: Dict):
        """Store analysis in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        key_levels = {
            'bids': analysis.get('significant_bids', [])[:3],
            'asks': analysis.get('significant_asks', [])[:3]
        }
        
        cursor.execute('''
            INSERT INTO orderbook_analysis 
            (mid_price, spread_bps, bid_ask_imbalance, bid_depth_1pct, ask_depth_1pct,
             bid_depth_5pct, ask_depth_5pct, large_bid_walls, large_ask_walls,
             weighted_bid_price, weighted_ask_price, price_skew, liquidity_score, 
             market_sentiment, key_levels)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            analysis.get('mid_price'),
            analysis.get('spread_bps'),
            analysis.get('short_term_imbalance'),
            analysis.get('bid_depth_1pct'),
            analysis.get('ask_depth_1pct'),
            analysis.get('bid_depth_5pct'),
            analysis.get('ask_depth_5pct'),
            analysis.get('large_bid_walls', 0),
            analysis.get('large_ask_walls', 0),
            analysis.get('weighted_bid_price'),
            analysis.get('weighted_ask_price'),
            analysis.get('price_skew'),
            analysis.get('liquidity_score'),
            analysis.get('market_sentiment'),
            json.dumps(key_levels)
        ))
        
        conn.commit()
        conn.close()


class EnhancedTelegramBot:
    def __init__(self, bot_token, chat_id):
        self.bot = Bot(token=bot_token)
        self.chat_id = chat_id
        self.exchange = ccxt.binance()
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT', 
                       'HYPE/USDT', 'PENGU/USDT', 'XRP/USDT', 'MKR/USDT', 'AAVE/USDT',
                       'DOGE/USDT', 'APT/USDT', 'XLM/USDT', 'QNT/USDT', 'SUI/USDT',
                       'WIF/USDT', 'TON/USDT', 'KAITO/USDT', 'LINK/USDT']
        self.lookback = 20
        self.volume_threshold = 1.5
        self.running = False
        self.orderbook_running = False
        
        # Initialize order book analyzer
        self.orderbook_analyzer = OrderBookAnalyzer()
        
    def get_ohlcv(self, symbol, timeframe='1h', limit=50):
        """Fetch OHLCV data"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None
    
    def calculate_levels(self, df):
        """Calculate support/resistance and indicators"""
        df['sma20'] = df['close'].rolling(20).mean()
        df['volume_avg'] = df['volume'].rolling(20).mean()
        df['resistance'] = df['high'].rolling(self.lookback).max()
        df['support'] = df['low'].rolling(self.lookback).min()

        # EMA crossovers
        df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()

        # 14-period RSI calculation
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        return df
    
    def check_breakout(self, df):
        """Check for breakout conditions"""
        if len(df) < 2:
            return None, None, None
            
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        volume_spike = latest['volume'] > latest['volume_avg'] * self.volume_threshold
        
        resistance_break = (latest['close'] > prev['resistance'] * 1.005 and 
                          latest['close'] > latest['sma20'])
        
        support_break = (latest['close'] < prev['support'] * 0.995 and 
                        latest['close'] < latest['sma20'])
        
        if resistance_break and volume_spike:
            return 'BULLISH', latest['close'], prev['resistance']
        elif support_break and volume_spike:
            return 'BEARISH', latest['close'], prev['support']
        
        return None, None, None
    
    async def send_alert(self, message):
        """Send alert to Telegram with rate limiting"""
        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode='HTML')
            logger.info(f"Alert sent: {message[:50]}...")
            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            await asyncio.sleep(5)
    
    async def scan_markets(self):
        """Scan all markets for breakouts"""
        alerts_sent = []
        
        for symbol in self.symbols:
            try:
                df = self.get_ohlcv(symbol)
                if df is None:
                    continue
                
                df = self.calculate_levels(df)
                signal, price, level = self.check_breakout(df)
                
                if signal:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    direction_emoji = "🚀" if signal == "BULLISH" else "📉"
                    
                    message = f"""
{direction_emoji} <b>{signal} BREAKOUT</b>
                    
<b>Symbol:</b> {symbol}
<b>Price:</b> ${price:.4f}
<b>Level:</b> ${level:.4f}
<b>Time:</b> {timestamp}

Volume spike detected! 📊
                    """
                    
                    await self.send_alert(message)
                    alerts_sent.append(f"{symbol} {signal}")

                # EMA crossover detection
                if len(df) >= 2:
                    prev_ema9, prev_ema21 = df['ema9'].iloc[-2], df['ema21'].iloc[-2]
                    curr_ema9, curr_ema21 = df['ema9'].iloc[-1], df['ema21'].iloc[-1]
                    if prev_ema9 < prev_ema21 and curr_ema9 > curr_ema21:
                        ema_signal = 'BULLISH'
                    elif prev_ema9 > prev_ema21 and curr_ema9 < curr_ema21:
                        ema_signal = 'BEARISH'
                    else:
                        ema_signal = None
                    if ema_signal:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        direction_emoji = "🚀" if ema_signal == "BULLISH" else "📉"
                        message = f"""{direction_emoji} <b>{ema_signal} EMA Crossover</b>

<b>Symbol:</b> {symbol}
<b>Price:</b> ${df['close'].iloc[-1]:.4f}
<b>Time:</b> {timestamp}
"""
                        await self.send_alert(message)
                        alerts_sent.append(f"{symbol} EMA_Crossover_{ema_signal}")

                # RSI divergence detection
                if len(df) >= 2 and 'rsi' in df:
                    prev = df.iloc[-2]
                    curr = df.iloc[-1]
                    if curr['low'] < prev['low'] and curr['rsi'] > prev['rsi']:
                        div_signal = 'BULLISH'
                    elif curr['high'] > prev['high'] and curr['rsi'] < prev['rsi']:
                        div_signal = 'BEARISH'
                    else:
                        div_signal = None
                    if div_signal:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        direction_emoji = "🚀" if div_signal == "BULLISH" else "📉"
                        message = f"""{direction_emoji} <b>{div_signal} RSI Divergence</b>

<b>Symbol:</b> {symbol}
<b>Price:</b> ${df['close'].iloc[-1]:.4f}
<b>Time:</b> {timestamp}
"""
                        await self.send_alert(message)
                        alerts_sent.append(f"{symbol} RSI_Divergence_{div_signal}")
                    
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
        
        return alerts_sent
    
    async def order_book_monitor(self):
        """Monitor order book every 30 minutes"""
        logger.info("Starting order book monitoring...")
        await self.send_alert("📊 <b>Order Book Analysis Started</b>\n\nReports every 30 minutes")
        
        while self.orderbook_running:
            try:
                analysis = self.orderbook_analyzer.analyze_order_book()
                if analysis:
                    report = self.orderbook_analyzer.generate_report(analysis)
                    await self.send_alert(report)
                    self.orderbook_analyzer.store_analysis(analysis)
                    logger.info("Order book report sent")
                
                # Wait 30 minutes
                await asyncio.sleep(1800)
                
            except Exception as e:
                logger.error(f"Order book monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def start_monitoring(self):
        """Start both breakout and order book monitoring"""
        self.running = True
        self.orderbook_running = True
        
        await self.send_alert("""🤖 <b>Enhanced Bot Started</b>

🚀 <b>Breakout Monitoring:</b> """ + ", ".join(self.symbols) + """

📊 <b>Order Book Analysis:</b> BTC/USDT every 30min

Ready to monitor! 📈""")
        
        # Start both monitoring tasks
        breakout_task = asyncio.create_task(self.breakout_monitor())
        orderbook_task = asyncio.create_task(self.order_book_monitor())
        
        await asyncio.gather(breakout_task, orderbook_task)
    
    async def breakout_monitor(self):
        """Monitor breakouts"""
        while self.running:
            try:
                logger.info("Scanning markets for breakouts...")
                alerts = await self.scan_markets()
                if alerts:
                    logger.info(f"Sent {len(alerts)} breakout alerts")
                
                await asyncio.sleep(600)  # 10 minutes
                
            except Exception as e:
                logger.error(f"Breakout monitoring error: {e}")
                await asyncio.sleep(60)
    
    def stop_monitoring(self):
        """Stop both monitoring loops"""
        self.running = False
        self.orderbook_running = False


# Bot command handlers
async def start_command(update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    await update.message.reply_text(
        "🤖 <b>Enhanced Crypto Bot</b>\n\n"
        "Features:\n"
        "🚀 Breakout alerts for 19 symbols\n"
        "📊 BTC order book analysis every 30min\n\n"
        "Commands:\n"
        "/start - Show this message\n"
        "/status - Check bot status\n"
        "/symbols - Show monitored symbols\n"
        "/orderbook - Get instant order book analysis\n"
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
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT', 
               'HYPE/USDT', 'PENGU/USDT', 'XRP/USDT', 'MKR/USDT', 'AAVE/USDT',
               'DOGE/USDT', 'APT/USDT', 'XLM/USDT', 'QNT/USDT', 'SUI/USDT',
               'WIF/USDT', 'TON/USDT', 'KAITO/USDT', 'LINK/USDT']
    symbol_list = "\n".join([f"• {symbol}" for symbol in symbols])
    await update.message.reply_text(
        f"<b>Monitored Symbols ({len(symbols)}):</b>\n{symbol_list}\n\n"
        f"<b>Order Book Analysis:</b>\n• BTC/USDT (every 30min)",
        parse_mode='HTML'
    )

async def orderbook_command(update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /orderbook command - instant analysis"""
    bot_instance = context.bot_data.get('bot_instance')
    
    if bot_instance:
        await update.message.reply_text("📊 Analyzing BTC order book...", parse_mode='HTML')
        
        try:
            analysis = bot_instance.orderbook_analyzer.analyze_order_book()
            if analysis:
                report = bot_instance.orderbook_analyzer.generate_report(analysis)
                await update.message.reply_text(report, parse_mode='HTML')
            else:
                await update.message.reply_text("❌ Failed to fetch order book data", parse_mode='HTML')
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}", parse_mode='HTML')
    else:
        await update.message.reply_text("❌ Bot not initialized", parse_mode='HTML')

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
    
    # Create enhanced bot instance
    bot = EnhancedTelegramBot(BOT_TOKEN, CHAT_ID)
    
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Add command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("symbols", symbols_command))
    application.add_handler(CommandHandler("orderbook", orderbook_command))
    application.add_handler(CommandHandler("stop", stop_command))
    
    # Store bot instance for command access
    application.bot_data['bot_instance'] = bot
    
    # Start monitoring automatically when bot starts
    async def post_init(application):
        logger.info("Post init called - starting enhanced monitoring")
        asyncio.create_task(bot.start_monitoring())
    
    application.post_init = post_init
    
    # Run bot
    logger.info("Starting enhanced Telegram bot...")
    application.run_polling()

if __name__ == "__main__":
    main()