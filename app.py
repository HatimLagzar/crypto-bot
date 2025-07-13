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

    def calculate_liquidity_score(self, analysis: Dict) -> float:
        """Calculate overall liquidity health score (0-100)"""
        score = 50  # Start with neutral
        
        # Spread component (30% weight)
        spread_bps = analysis.get('spread_bps', 0)
        if spread_bps < 2:
            score += 20
        elif spread_bps < 5:
            score += 10
        elif spread_bps > 15:
            score -= 20
        elif spread_bps > 10:
            score -= 10
            
        # Depth component (40% weight)
        bid_depth = analysis.get('bid_depth_1pct', 0)
        ask_depth = analysis.get('ask_depth_1pct', 0)
        total_depth = bid_depth + ask_depth
        
        if total_depth > 1000000:  # High liquidity
            score += 25
        elif total_depth > 500000:  # Good liquidity
            score += 15
        elif total_depth > 100000:  # Moderate liquidity
            score += 5
        elif total_depth < 10000:  # Low liquidity
            score -= 20
            
        # Balance component (20% weight)
        imbalance = abs(analysis.get('immediate_imbalance', 0))
        if imbalance < 0.1:
            score += 10
        elif imbalance < 0.3:
            score += 5
        elif imbalance > 0.6:
            score -= 15
            
        # Support/resistance levels (10% weight)
        total_walls = analysis.get('large_bid_walls', 0) + analysis.get('large_ask_walls', 0)
        if total_walls > 3:
            score += 5
        elif total_walls == 0:
            score -= 5
            
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
        analysis['liquidity_score'] = self.calculate_liquidity_score(analysis)

        return analysis

    def generate_report(self, analysis: Dict) -> str:
        """Generate order book report"""
        report = f"📊 <b>Order Book Analysis</b>\n"
        report += f"💰 Mid Price: ${analysis['mid_price']:.2f}\n"
        report += f"📈 Spread: {analysis['spread_bps']:.2f} bps\n\n"
        
        # Market pressure
        if 'immediate_imbalance' in analysis:
            imbalance = analysis['immediate_imbalance']
            if abs(imbalance) > 0.3:
                direction = "BULLISH" if imbalance > 0 else "BEARISH"
                report += f"⚡ Market Pressure: <b>{direction}</b> ({imbalance:.1%})\n"
            else:
                report += f"⚖️ Market Pressure: BALANCED ({imbalance:.1%})\n"
        
        # Significant levels
        if analysis.get('large_bid_walls', 0) > 0:
            report += f"🟢 Large Bid Walls: {analysis['large_bid_walls']}\n"
        if analysis.get('large_ask_walls', 0) > 0:
            report += f"🔴 Large Ask Walls: {analysis['large_ask_walls']}\n"
            
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

    def check_breakout(self, df):
        """Check for breakout conditions"""
        if len(df) < self.lookback:
            return None, None, None
            
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Check for price breakouts
        resistance_break = (
            current['close'] > current['bb_upper'] and
            previous['close'] <= previous['bb_upper'] and
            current['volume_ratio'] > self.volume_threshold
        )
        
        support_break = (
            current['close'] < current['bb_lower'] and
            previous['close'] >= previous['bb_lower'] and
            current['volume_ratio'] > self.volume_threshold
        )
        
        # Check for moving average crossovers
        ma_bullish = (
            current['ema_20'] > current['sma_50'] and
            previous['ema_20'] <= previous['sma_50'] and
            current['volume_ratio'] > 1.2
        )
        
        ma_bearish = (
            current['ema_20'] < current['sma_50'] and
            previous['ema_20'] >= previous['sma_50'] and
            current['volume_ratio'] > 1.2
        )
        
        # Determine direction and strength
        if resistance_break or ma_bullish:
            strength = "STRONG" if current['volume_ratio'] > 2.0 else "MODERATE"
            analysis = f"RSI: {current['rsi']:.1f}, Volume: {current['volume_ratio']:.1f}x"
            return "bullish", strength, analysis
            
        elif support_break or ma_bearish:
            strength = "STRONG" if current['volume_ratio'] > 2.0 else "MODERATE"
            analysis = f"RSI: {current['rsi']:.1f}, Volume: {current['volume_ratio']:.1f}x"
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
                direction, strength, analysis = self.check_breakout(df)
                
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
        "/analyze SYMBOL - Run order book analysis for SYMBOL (e.g. BTC/USDT)\n"
        "Commands:\n"
        "/start - Show this message\n"
        "/status - Check bot status\n"
        "/symbols - Show monitored symbols\n"
        "/orderbook - Get instant order book analysis\n"
        "/analyze SYMBOL - Analyze order book for a given symbol\n"
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
    """Handle /analyze <symbol> command for order book analysis"""
    bot_instance = context.bot_data.get('bot_instance')
    if not bot_instance:
        await update.message.reply_text("❌ Bot not initialized", parse_mode='HTML')
        return

    args = context.args
    if not args:
        await update.message.reply_text(
            "Usage: /analyze SYMBOL (e.g. BTC/USDT)", parse_mode='HTML'
        )
        return

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