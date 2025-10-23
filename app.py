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


class USDTDominanceAnalyzer:
    """USDT Dominance macro sentiment analysis"""
    
    def __init__(self):
        # Use persistent storage path if available, fallback to current directory
        data_dir = os.environ.get('DATA_DIR', '.')
        self.db_path = os.path.join(data_dir, "usdt_dominance.db")
        self.init_database()
        
    def init_database(self):
        """Initialize database for USDT.D data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS usdt_dominance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                dominance_pct REAL,
                sentiment TEXT,
                trend_direction TEXT,
                signal_strength TEXT,
                resistance_level REAL,
                support_level REAL
            )
        ''')
        
        # Create table for historical dominance data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS usdt_dominance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                dominance_pct REAL NOT NULL,
                raw_data TEXT,
                UNIQUE(timestamp)
            )
        ''')
        conn.commit()
        conn.close()
        
    def store_current_dominance(self, dominance_pct: float, raw_data: str = None):
        """Store current dominance data in history table"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Store with current timestamp, ignore duplicates
            cursor.execute('''
                INSERT OR IGNORE INTO usdt_dominance_history 
                (timestamp, dominance_pct, raw_data) 
                VALUES (datetime('now'), ?, ?)
            ''', (dominance_pct, raw_data or ""))
            
            conn.commit()
            conn.close()
            logger.info(f"Stored USDT dominance: {dominance_pct:.3f}%")
            
        except Exception as e:
            logger.error(f"Failed to store dominance data: {e}")
    
    def get_historical_dominance(self, hours: int = 168) -> Optional[pd.DataFrame]:
        """Get historical dominance data from database (default: 7 days)"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get historical data from database
            query = '''
                SELECT timestamp, dominance_pct 
                FROM usdt_dominance_history 
                WHERE timestamp >= datetime('now', '-{} hours')
                ORDER BY timestamp ASC
            '''.format(hours)
            
            df = pd.read_sql_query(query, conn, parse_dates=['timestamp'])
            conn.close()
            
            if df.empty:
                logger.warning("No historical dominance data found")
                return None
                
            df.set_index('timestamp', inplace=True)
            df.rename(columns={'dominance_pct': 'dominance'}, inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical dominance data: {e}")
            return None
    
    def get_usdt_dominance_data(self) -> Optional[pd.DataFrame]:
        """Fetch current USDT dominance and build historical dataset"""
        try:
            # Fetch current dominance from CoinGecko
            url = "https://api.coingecko.com/api/v3/global"
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if 'data' not in data:
                logger.error("Invalid dominance data response")
                return None
                
            current_dominance = data['data']['market_cap_percentage'].get('usdt', 0)
            
            # Store current dominance in history
            self.store_current_dominance(current_dominance, str(data))
            
            # Get historical data from database
            historical_df = self.get_historical_dominance(hours=168)  # 7 days
            
            if historical_df is None or len(historical_df) < 3:
                logger.warning("Insufficient historical data for analysis")
                # Create minimal dataset with just current value
                now = datetime.now()
                df = pd.DataFrame(
                    {'dominance': [current_dominance]},
                    index=[now]
                )
                return df
            
            # Ensure current value is the latest
            now = datetime.now()
            if historical_df.index[-1] < now - pd.Timedelta(minutes=30):
                # Add current value if last entry is older than 30 minutes
                new_row = pd.DataFrame(
                    {'dominance': [current_dominance]}, 
                    index=[now]
                )
                historical_df = pd.concat([historical_df, new_row])
            
            return historical_df
            
        except Exception as e:
            logger.error(f"Error fetching USDT dominance data: {e}")
            return None
            
    def get_last_stored_analysis(self) -> Optional[Dict]:
        """Get the last stored analysis for comparison"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT dominance_pct, sentiment, trend_direction, signal_strength, timestamp
                FROM usdt_dominance 
                ORDER BY timestamp DESC 
                LIMIT 1
            ''')
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'dominance': result[0],
                    'sentiment': result[1],
                    'trend_direction': result[2],
                    'signal_strength': result[3],
                    'timestamp': result[4]
                }
            return None
            
        except Exception as e:
            logger.error(f"Error fetching last analysis: {e}")
            return None
    
    def analyze_dominance_sentiment(self, df: pd.DataFrame) -> Dict:
        """Analyze USDT dominance for market sentiment"""
        if len(df) < 2:
            logger.warning("Insufficient data for analysis")
            return {}
            
        # Data validation
        if df['dominance'].isna().any():
            logger.warning("NaN values detected in dominance data")
            df = df.dropna()
            
        if len(df) < 2:
            return {}
            
        # Calculate technical indicators with proper window sizes
        min_window = min(20, len(df))
        df['sma_20'] = df['dominance'].rolling(window=min_window).mean()
        df['sma_50'] = df['dominance'].rolling(window=min(50, len(df))).mean()
        df['std_20'] = df['dominance'].rolling(window=min_window).std()
        
        # Bollinger Bands for dominance
        df['bb_upper'] = df['sma_20'] + (df['std_20'] * 2)
        df['bb_lower'] = df['sma_20'] - (df['std_20'] * 2)
        
        # RSI for dominance
        delta = df['dominance'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=min(14, len(df))).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=min(14, len(df))).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        current = df.iloc[-1]
        
        # Get last stored analysis for better percentage calculation
        last_analysis = self.get_last_stored_analysis()
        
        # Calculate percentage change using time-based lookback (not index-based)
        pct_changes = {}
        current_time = current.name if hasattr(current, 'name') else df.index[-1]

        # 1-hour change - find closest data point to 1 hour ago
        one_hour_ago = current_time - pd.Timedelta(hours=1)
        hour_mask = df.index <= one_hour_ago
        if hour_mask.any():
            previous_1h = df[hour_mask].iloc[-1]  # Most recent point before 1h ago
            time_diff = (current_time - previous_1h.name).total_seconds() / 3600  # hours
            if time_diff <= 2:  # Within 2 hours is acceptable for 1h calculation
                pct_changes['1h'] = ((current['dominance'] - previous_1h['dominance']) / previous_1h['dominance']) * 100
                logger.debug(f"1h change calculated using data from {time_diff:.1f}h ago")
            else:
                logger.debug(f"1h change skipped - nearest data is {time_diff:.1f}h old")
        else:
            logger.debug("1h change skipped - no historical data older than 1h")

        # 4-hour change - find closest data point to 4 hours ago
        four_hours_ago = current_time - pd.Timedelta(hours=4)
        four_hour_mask = df.index <= four_hours_ago
        if four_hour_mask.any():
            previous_4h = df[four_hour_mask].iloc[-1]  # Most recent point before 4h ago
            time_diff = (current_time - previous_4h.name).total_seconds() / 3600  # hours
            if time_diff <= 6:  # Within 6 hours is acceptable for 4h calculation
                pct_changes['4h'] = ((current['dominance'] - previous_4h['dominance']) / previous_4h['dominance']) * 100
                logger.debug(f"4h change calculated using data from {time_diff:.1f}h ago")
            else:
                logger.debug(f"4h change skipped - nearest data is {time_diff:.1f}h old")
        else:
            logger.debug("4h change skipped - no historical data older than 4h")

        # 24-hour change - find closest data point to 24 hours ago
        twenty_four_hours_ago = current_time - pd.Timedelta(hours=24)
        day_mask = df.index <= twenty_four_hours_ago
        if day_mask.any():
            previous_24h = df[day_mask].iloc[-1]  # Most recent point before 24h ago
            time_diff = (current_time - previous_24h.name).total_seconds() / 3600  # hours
            if time_diff <= 30:  # Within 30 hours is acceptable for 24h calculation
                pct_changes['24h'] = ((current['dominance'] - previous_24h['dominance']) / previous_24h['dominance']) * 100
                logger.debug(f"24h change calculated using data from {time_diff:.1f}h ago")
            else:
                logger.debug(f"24h change skipped - nearest data is {time_diff:.1f}h old")
        else:
            logger.debug("24h change skipped - no historical data older than 24h")
        
        # Use the most appropriate timeframe for percentage change
        if last_analysis and last_analysis['dominance']:
            # Compare with last stored analysis
            pct_change = ((current['dominance'] - last_analysis['dominance']) / last_analysis['dominance']) * 100
        elif '24h' in pct_changes:
            pct_change = pct_changes['24h']
        elif '4h' in pct_changes:
            pct_change = pct_changes['4h']
        elif '1h' in pct_changes:
            pct_change = pct_changes['1h']
        else:
            pct_change = 0.0
            
        # Determine trend and sentiment
        trend_score = 0
        
        # Short-term trend (last available periods)
        recent_periods = min(10, len(df))
        recent_trend = df['dominance'].tail(recent_periods)
        if len(recent_trend) >= 2:
            if recent_trend.iloc[-1] > recent_trend.iloc[0]:
                trend_score += 1  # Rising dominance = bearish for crypto
            else:
                trend_score -= 1  # Falling dominance = bullish for crypto
                
        # Moving average position (if we have enough data)
        if not pd.isna(current['sma_20']):
            if current['dominance'] > current['sma_20']:
                trend_score += 1
            else:
                trend_score -= 1
        
        # Rate of change significance
        if abs(pct_change) > 1.5:  # Lowered threshold for more sensitivity
            if pct_change > 0:
                trend_score += 2  # Strong bearish signal
            else:
                trend_score -= 2  # Strong bullish signal
        
        # Multiple timeframe confirmation
        if '1h' in pct_changes and '4h' in pct_changes:
            if pct_changes['1h'] > 0 and pct_changes['4h'] > 0:
                trend_score += 1  # Confirmed uptrend
            elif pct_changes['1h'] < 0 and pct_changes['4h'] < 0:
                trend_score -= 1  # Confirmed downtrend
                
        # Classify sentiment
        if trend_score >= 2:
            sentiment = "BEARISH"  # High USDT.D = people fleeing to stables
        elif trend_score <= -2:
            sentiment = "BULLISH"  # Low USDT.D = people leaving stables for crypto
        else:
            sentiment = "NEUTRAL"
            
        # Detect breakout levels
        recent_high = df['dominance'].rolling(window=min(20, len(df))).max().iloc[-1]
        recent_low = df['dominance'].rolling(window=min(20, len(df))).min().iloc[-1]
        
        # Signal strength based on multiple factors
        strength_score = 0
        
        # RSI extremes
        if not pd.isna(current['rsi']):
            if current['rsi'] > 70 or current['rsi'] < 30:
                strength_score += 1
                
        # Bollinger Band breaks
        if not pd.isna(current['bb_upper']) and not pd.isna(current['bb_lower']):
            if current['dominance'] > current['bb_upper'] or current['dominance'] < current['bb_lower']:
                strength_score += 1
                
        # Volume of change
        if abs(pct_change) > 2.5:
            strength_score += 1
            
        # Multiple timeframe alignment
        if len(pct_changes) >= 2:
            same_direction = all(x > 0 for x in pct_changes.values()) or all(x < 0 for x in pct_changes.values())
            if same_direction:
                strength_score += 1
                
        signal_strength = "STRONG" if strength_score >= 2 else "MODERATE" if strength_score >= 1 else "WEAK"
        
        return {
            'timestamp': current.name,
            'dominance': current['dominance'],
            'sentiment': sentiment,
            'signal_strength': signal_strength,
            'trend_direction': "UP" if trend_score > 0 else "DOWN",
            'pct_change': pct_change,
            'pct_changes': pct_changes,
            'rsi': current['rsi'] if not pd.isna(current['rsi']) else 50.0,
            'resistance_level': recent_high,
            'support_level': recent_low,
            'bb_upper': current['bb_upper'] if not pd.isna(current['bb_upper']) else current['dominance'],
            'bb_lower': current['bb_lower'] if not pd.isna(current['bb_lower']) else current['dominance'],
            'trend_score': trend_score,
            'strength_score': strength_score
        }
        
    def generate_dominance_alert(self, analysis: Dict) -> str:
        """Generate enhanced alert message for dominance changes"""
        sentiment = analysis['sentiment']
        strength = analysis['signal_strength']
        dominance = analysis['dominance']
        pct_change = analysis['pct_change']
        pct_changes = analysis.get('pct_changes', {})
        
        # Emoji based on sentiment
        emoji = "🔴" if sentiment == "BEARISH" else "🟢" if sentiment == "BULLISH" else "⚪"
        
        alert = f"{emoji} <b>USDT.D SENTIMENT ALERT</b>\n\n"
        alert += f"📊 Dominance: {dominance:.3f}% ({pct_change:+.2f}%)\n"
        alert += f"🎯 Market Sentiment: <b>{sentiment}</b>\n"
        alert += f"⚡ Signal Strength: {strength}\n"
        alert += f"📈 RSI: {analysis['rsi']:.1f}\n"
        
        # Add multiple timeframe changes if available
        if pct_changes:
            alert += "\n📊 <b>Timeframe Analysis:</b>\n"
            for timeframe, change in pct_changes.items():
                direction = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                alert += f"{direction} {timeframe}: {change:+.2f}%\n"
        
        # Add technical levels
        alert += f"\n🔍 <b>Technical Levels:</b>\n"
        alert += f"⬆️ Resistance: {analysis['resistance_level']:.3f}%\n"
        alert += f"⬇️ Support: {analysis['support_level']:.3f}%\n"
        
        # Interpretation
        alert += "\n💡 <b>Market Interpretation:</b>\n"
        if sentiment == "BEARISH":
            alert += "📉 <i>USDT dominance rising - traders moving to stablecoins</i>\n"
            alert += "⚠️ <i>Expect crypto weakness/selling pressure</i>"
        elif sentiment == "BULLISH":
            alert += "📈 <i>USDT dominance falling - traders leaving stablecoins</i>\n"
            alert += "🚀 <i>Expect crypto strength/buying pressure</i>"
        else:
            alert += "⚖️ <i>USDT dominance stable - neutral market sentiment</i>"
            
        return alert
        
    def store_analysis(self, analysis: Dict):
        """Store dominance analysis in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO usdt_dominance (
                    dominance_pct, sentiment, trend_direction, signal_strength,
                    resistance_level, support_level
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                analysis.get('dominance', 0),
                analysis.get('sentiment', 'NEUTRAL'),
                analysis.get('trend_direction', 'FLAT'),
                analysis.get('signal_strength', 'WEAK'),
                analysis.get('resistance_level', 0),
                analysis.get('support_level', 0)
            ))
            
            conn.commit()
            conn.close()
            logger.info("USDT dominance analysis stored")
            
        except Exception as e:
            logger.error(f"Failed to store dominance analysis: {e}")


class VolumeSurgeAnalyzer:
    """Volume surge detection for early warning alerts"""
    
    def __init__(self):
        data_dir = os.environ.get('DATA_DIR', '.')
        self.db_path = os.path.join(data_dir, "volume_surges.db")
        self.init_database()
        
        # Volume surge thresholds
        self.thresholds = {
            'notable': 2.0,      # 2.0x average volume
            'significant': 5.0,   # 5x average volume  
            'extreme': 10.0,      # 10x average volume
            'mega': 20.0         # 20x average volume (very rare)
        }
        
    def init_database(self):
        """Initialize database for volume surge data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS volume_surges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT,
                volume_ratio REAL,
                surge_type TEXT,
                price_change_pct REAL,
                volume_alignment TEXT,
                risk_assessment TEXT,
                market_cap_rank INTEGER,
                alert_sent BOOLEAN DEFAULT 0
            )
        ''')
        conn.commit()
        conn.close()
        
    def analyze_volume_surge(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """Analyze symbol for volume surges"""
        if len(df) < 21:  # Need at least 21 periods for proper analysis
            return None
            
        current = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else current
        
        # Calculate volume metrics
        volume_sma_20 = df['volume'].rolling(window=20).mean().iloc[-1]
        volume_ratio = current['volume'] / volume_sma_20 if volume_sma_20 > 0 else 0
        
        # Only proceed if volume surge is significant
        if volume_ratio < self.thresholds['notable']:
            return None
            
        # Classify surge intensity
        if volume_ratio >= self.thresholds['mega']:
            surge_type = "MEGA"
        elif volume_ratio >= self.thresholds['extreme']:
            surge_type = "EXTREME"
        elif volume_ratio >= self.thresholds['significant']:
            surge_type = "SIGNIFICANT"
        else:
            surge_type = "NOTABLE"
            
        # Calculate price context
        price_change_1h = ((current['close'] - previous['close']) / previous['close']) * 100
        
        # Determine volume alignment with price
        if price_change_1h > 2 and volume_ratio > 5:
            volume_alignment = "BULLISH_BREAKOUT"  # High volume + strong up move
        elif price_change_1h < -2 and volume_ratio > 5:
            volume_alignment = "BEARISH_BREAKDOWN"  # High volume + strong down move
        elif abs(price_change_1h) < 1 and volume_ratio > 10:
            volume_alignment = "ACCUMULATION"  # Huge volume, little price change
        elif price_change_1h > 0:
            volume_alignment = "BULLISH_INTEREST"  # Volume + modest up move
        else:
            volume_alignment = "BEARISH_PRESSURE"  # Volume + modest down move
            
        # Risk assessment based on multiple factors
        risk_score = 0
        
        # Volume intensity risk
        if volume_ratio > 20:
            risk_score += 3  # Extreme volume = high risk
        elif volume_ratio > 10:
            risk_score += 2
        elif volume_ratio > 5:
            risk_score += 1
            
        # Price volatility risk
        if abs(price_change_1h) > 10:
            risk_score += 2  # High volatility = high risk
        elif abs(price_change_1h) > 5:
            risk_score += 1
            
        # Time-based risk (avoid overnight low-volume markets)
        current_hour = datetime.now().hour
        if 0 <= current_hour <= 6:  # Overnight hours
            risk_score += 1
            
        # Classify overall risk
        if risk_score >= 5:
            risk_assessment = "VERY_HIGH"
        elif risk_score >= 3:
            risk_assessment = "HIGH"
        elif risk_score >= 2:
            risk_assessment = "MODERATE"
        else:
            risk_assessment = "LOW"
            
        return {
            'symbol': symbol,
            'timestamp': current.name if hasattr(current, 'name') else datetime.now(),
            'volume_ratio': volume_ratio,
            'surge_type': surge_type,
            'price_change_pct': price_change_1h,
            'volume_alignment': volume_alignment,
            'risk_assessment': risk_assessment,
            'current_price': current['close'],
            'volume_sma_20': volume_sma_20,
            'current_volume': current['volume']
        }
        
    def generate_volume_alert(self, analysis: Dict) -> str:
        """Generate volume surge alert message"""
        symbol = analysis['symbol']
        volume_ratio = analysis['volume_ratio']
        surge_type = analysis['surge_type']
        price_change = analysis['price_change_pct']
        alignment = analysis['volume_alignment']
        risk = analysis['risk_assessment']
        price = analysis['current_price']
        
        # Choose emoji based on surge type and alignment
        if surge_type == "MEGA":
            emoji = "🚨"
        elif surge_type == "EXTREME":
            emoji = "⚠️"
        elif alignment in ["BULLISH_BREAKOUT", "BULLISH_INTEREST"]:
            emoji = "🟢"
        elif alignment in ["BEARISH_BREAKDOWN", "BEARISH_PRESSURE"]:
            emoji = "🔴"
        else:
            emoji = "🟡"
            
        alert = f"{emoji} <b>VOLUME SURGE ALERT</b>\n\n"
        alert += f"💎 Symbol: <b>{symbol}</b>\n"
        alert += f"📊 Price: ${price:.6f if price < 1 else price:.4f if price < 100 else price:.2f} ({price_change:+.2f}%)\n"
        alert += f"📈 Volume: <b>{volume_ratio:.1f}x</b> average ({surge_type})\n"
        alert += f"🎯 Pattern: {alignment.replace('_', ' ')}\n"
        alert += f"⚠️ Risk Level: <b>{risk.replace('_', ' ')}</b>\n\n"
        
        # Add interpretation based on volume alignment
        if alignment == "BULLISH_BREAKOUT":
            alert += "🚀 <i>Strong buying pressure - potential breakout</i>\n"
            alert += "👀 <i>Watch for continuation or rejection</i>"
        elif alignment == "BEARISH_BREAKDOWN":
            alert += "📉 <i>Heavy selling pressure - potential breakdown</i>\n"
            alert += "⚠️ <i>Consider risk management</i>"
        elif alignment == "ACCUMULATION":
            alert += "🤔 <i>Huge volume, minimal price change</i>\n"
            alert += "📡 <i>Possible news, whale activity, or manipulation</i>"
        elif alignment == "BULLISH_INTEREST":
            alert += "📈 <i>Increased buying interest emerging</i>\n"
            alert += "👁️ <i>Monitor for potential breakout setup</i>"
        else:
            alert += "📊 <i>Unusual activity detected</i>\n"
            alert += "🔍 <i>Investigate for potential catalysts</i>"
            
        return alert
        
    def store_surge(self, analysis: Dict):
        """Store volume surge in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO volume_surges (
                    symbol, volume_ratio, surge_type, price_change_pct,
                    volume_alignment, risk_assessment, alert_sent
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis['symbol'],
                analysis['volume_ratio'],
                analysis['surge_type'],
                analysis['price_change_pct'],
                analysis['volume_alignment'],
                analysis['risk_assessment'],
                1  # alert_sent = True
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Volume surge stored for {analysis['symbol']}")
            
        except Exception as e:
            logger.error(f"Failed to store volume surge: {e}")


class CapitalRotationAnalyzer:
    """Detect capital rotation (volume flow) between assets."""

    def __init__(self):
        data_dir = os.environ.get('DATA_DIR', '.')
        self.db_path = os.path.join(data_dir, "rotation_data.db")
        self.init_database()

    def init_database(self):
        """Initialize database for capital rotation events"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS capital_rotation (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    timeframe TEXT,
                    from_symbol TEXT,
                    to_symbol TEXT,
                    inflow_ratio REAL,
                    outflow_ratio REAL,
                    inflow_momentum REAL,
                    outflow_momentum REAL,
                    score REAL,
                    note TEXT
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to init rotation DB: {e}")

    def _symbol_metrics(self, df: pd.DataFrame) -> Optional[Dict]:
        """Compute quote-volume ratio and short-term momentum for a symbol."""
        if df is None or len(df) < 25:
            return None

        data = df.copy()
        # Quote volume in USDT terms for cross-asset comparability
        data['quote_volume'] = data['volume'] * data['close']
        data['quote_vol_sma'] = data['quote_volume'].rolling(window=20).mean()
        if pd.isna(data['quote_vol_sma'].iloc[-1]):
            return None

        quote_vol_ratio = data['quote_volume'].iloc[-1] / data['quote_vol_sma'].iloc[-1]

        # Short-term momentum (3 bars if possible, else 1)
        momentum_window = 3 if len(data) >= 4 else 1
        try:
            momentum = data['close'].pct_change(periods=momentum_window).iloc[-1]
        except Exception:
            momentum = 0.0

        return {
            'quote_vol_ratio': float(quote_vol_ratio),
            'momentum': float(0.0 if pd.isna(momentum) else momentum),
            'last_close': float(data['close'].iloc[-1])
        }

    def _score_rotation_pair(self, inflow: Dict, outflow: Dict) -> int:
        """Score a rotation pair from outflow->inflow (0-100)."""
        in_r = max(0.0, inflow['quote_vol_ratio'])
        out_r = max(0.0, outflow['quote_vol_ratio'])
        in_m = inflow['momentum']
        out_m = outflow['momentum']

        # Components clamped to [0,1]
        in_comp = max(0.0, min(1.0, (in_r - 1.0) / 1.0))                # 1x->2x maps 0..1
        out_comp = max(0.0, min(1.0, (1.0 - out_r) / 1.0))               # 1x->0x maps 0..1
        mom_diff = in_m - out_m
        mom_comp = max(0.0, min(1.0, mom_diff / 0.05))                   # 0%..5% diff maps 0..1

        score = 100.0 * (0.4 * in_comp + 0.4 * out_comp + 0.2 * mom_comp)
        return int(round(max(0, min(100, score))))

    def analyze_rotation(self, symbols: List[str], get_ohlcv_callable, timeframe: str = '1h') -> Optional[Dict]:
        """Analyze rotation across symbols using quote-volume ratios and momentum.

        Returns dict with inflows, outflows, and paired flows with scores.
        """
        metrics = {}
        for symbol in symbols:
            try:
                df = get_ohlcv_callable(symbol, timeframe=timeframe, limit=60)
                m = self._symbol_metrics(df)
                if m:
                    metrics[symbol] = m
            except Exception as e:
                logger.debug(f"Rotation metrics failed for {symbol}: {e}")
                continue

        if not metrics:
            return None

        # Classify inflow/outflow candidates
        # Thresholds chosen to reduce noise across 1h bars
        inflow_candidates = [
            {
                'symbol': s,
                **m
            } for s, m in metrics.items()
            if m['quote_vol_ratio'] >= 1.5 and m['momentum'] >= 0.003
        ]
        outflow_candidates = [
            {
                'symbol': s,
                **m
            } for s, m in metrics.items()
            if m['quote_vol_ratio'] <= 0.8 and m['momentum'] <= -0.003
        ]

        inflow_candidates.sort(key=lambda x: (x['quote_vol_ratio'], x['momentum']), reverse=True)
        outflow_candidates.sort(key=lambda x: (x['quote_vol_ratio'], x['momentum']))

        # Pair top inflows with top outflows (up to 3 pairs)
        pairs = []
        for i in range(min(3, len(inflow_candidates), len(outflow_candidates))):
            inflow = inflow_candidates[i]
            outflow = outflow_candidates[i]
            score = self._score_rotation_pair(inflow, outflow)
            pairs.append({
                'from_symbol': outflow['symbol'],
                'to_symbol': inflow['symbol'],
                'inflow_ratio': inflow['quote_vol_ratio'],
                'outflow_ratio': outflow['quote_vol_ratio'],
                'inflow_momentum': inflow['momentum'],
                'outflow_momentum': outflow['momentum'],
                'score': score
            })

        return {
            'timeframe': timeframe,
            'generated_at': datetime.now(),
            'universe': len(metrics),
            'inflows': inflow_candidates[:5],
            'outflows': outflow_candidates[:5],
            'pairs': pairs
        }

    def generate_rotation_alert(self, analysis: Dict, min_score: int = 60) -> Optional[str]:
        """Build a human-friendly rotation alert message."""
        pairs = [p for p in analysis.get('pairs', []) if p['score'] >= min_score]
        if not pairs:
            return None

        tf = analysis.get('timeframe', '1h')
        header = f"🔄 <b>Capital Rotation Detected</b> ({tf})\n\n"
        body_lines = []
        for p in pairs:
            in_ratio = p['inflow_ratio']
            out_ratio = p['outflow_ratio']
            in_m = p['inflow_momentum'] * 100
            out_m = p['outflow_momentum'] * 100
            confidence = 'HIGH' if p['score'] >= 75 else 'MEDIUM' if p['score'] >= 60 else 'LOW'
            body_lines.append(
                (
                    f"{p['from_symbol']} ➜ {p['to_symbol']} | "
                    f"In: {in_ratio:.1f}x, Out: {out_ratio:.1f}x | "
                    f"Perf: +{in_m:.2f}% vs {out_m:.2f}% | "
                    f"Score: <b>{p['score']}</b> ({confidence})"
                )
            )

        # Add leaders/laggards context
        inflows = analysis.get('inflows', [])
        outflows = analysis.get('outflows', [])
        if inflows:
            top_in = ", ".join([f"{x['symbol']}({x['quote_vol_ratio']:.1f}x)" for x in inflows[:3]])
        else:
            top_in = "-"
        if outflows:
            top_out = ", ".join([f"{x['symbol']}({x['quote_vol_ratio']:.1f}x)" for x in outflows[:3]])
        else:
            top_out = "-"

        footer = (
            "\n\n"
            f"🏁 Inflows: {top_in}\n"
            f"💨 Outflows: {top_out}"
        )

        return header + "\n".join(body_lines) + footer

    def store_rotation(self, analysis: Dict):
        """Persist detected rotation pairs to the database."""
        try:
            pairs = analysis.get('pairs', [])
            if not pairs:
                return
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            for p in pairs:
                cursor.execute(
                    '''
                    INSERT INTO capital_rotation (
                        timeframe, from_symbol, to_symbol, inflow_ratio, outflow_ratio,
                        inflow_momentum, outflow_momentum, score, note
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        analysis.get('timeframe', '1h'),
                        p['from_symbol'],
                        p['to_symbol'],
                        p['inflow_ratio'],
                        p['outflow_ratio'],
                        p['inflow_momentum'],
                        p['outflow_momentum'],
                        p['score'],
                        ''
                    )
                )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to store rotation analysis: {e}")

class OrderBookSetupAnalyzer:
    """Professional order book setup detection for trading opportunities"""
    
    def __init__(self):
        data_dir = os.environ.get('DATA_DIR', '.')
        self.db_path = os.path.join(data_dir, "orderbook_setups.db")
        self.init_database()
        
    def init_database(self):
        """Initialize database for order book setups"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orderbook_setups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT,
                setup_type TEXT,
                direction TEXT,
                entry_price REAL,
                target_price REAL,
                stop_loss REAL,
                risk_reward_ratio REAL,
                confidence TEXT,
                wall_size REAL,
                wall_distance_pct REAL,
                setup_sent BOOLEAN DEFAULT 0
            )
        ''')
        conn.commit()
        conn.close()
        
    def find_liquidity_gaps(self, bids: List[List[float]], asks: List[List[float]], mid_price: float) -> Dict:
        """Identify significant liquidity gaps in the order book"""
        gaps = {'bid_gaps': [], 'ask_gaps': []}
        
        # Analyze bid gaps (support side)
        for i in range(len(bids) - 1):
            current_price = bids[i][0]
            next_price = bids[i + 1][0]
            gap_size = ((current_price - next_price) / current_price) * 100
            
            if gap_size > 0.1:  # Gap larger than 0.1%
                distance_from_mid = ((mid_price - current_price) / mid_price) * 100
                if distance_from_mid < 5:  # Within 5% of mid price
                    gaps['bid_gaps'].append({
                        'start_price': current_price,
                        'end_price': next_price,
                        'gap_size_pct': gap_size,
                        'distance_from_mid_pct': distance_from_mid
                    })
        
        # Analyze ask gaps (resistance side)
        for i in range(len(asks) - 1):
            current_price = asks[i][0]
            next_price = asks[i + 1][0]
            gap_size = ((next_price - current_price) / current_price) * 100
            
            if gap_size > 0.1:  # Gap larger than 0.1%
                distance_from_mid = ((current_price - mid_price) / mid_price) * 100
                if distance_from_mid < 5:  # Within 5% of mid price
                    gaps['ask_gaps'].append({
                        'start_price': current_price,
                        'end_price': next_price,
                        'gap_size_pct': gap_size,
                        'distance_from_mid_pct': distance_from_mid
                    })
        
        return gaps
    
    def analyze_wall_strength(self, orders: List[List[float]], is_bid: bool = True) -> List[Dict]:
        """Analyze order wall strength and significance"""
        if not orders:
            return []
            
        # Calculate statistics
        sizes = [qty for _, qty in orders[:200]]  # Analyze first 200 orders
        if not sizes:
            return []
            
        avg_size = statistics.mean(sizes)
        median_size = statistics.median(sizes)
        std_size = statistics.stdev(sizes) if len(sizes) > 1 else 0
        
        walls = []
        for price, qty in orders:
            # Define wall criteria
            is_large_wall = qty >= avg_size * 8  # 8x average size
            is_huge_wall = qty >= avg_size * 15  # 15x average size
            is_mega_wall = qty >= avg_size * 25  # 25x average size
            
            if is_large_wall:
                if is_mega_wall:
                    wall_strength = "MEGA"
                elif is_huge_wall:
                    wall_strength = "HUGE"
                else:
                    wall_strength = "LARGE"
                    
                walls.append({
                    'price': price,
                    'quantity': qty,
                    'strength': wall_strength,
                    'size_ratio': qty / avg_size,
                    'is_bid': is_bid
                })
                
        return walls[:10]  # Return top 10 walls
    
    def detect_breakout_setups(self, analysis: Dict, symbol: str) -> List[Dict]:
        """Detect breakout setups based on wall analysis"""
        setups = []
        mid_price = analysis['mid_price']
        significant_bids = analysis.get('significant_bids', [])
        significant_asks = analysis.get('significant_asks', [])
        
        # Bullish breakout setups (breaking resistance walls)
        for ask in significant_asks[:3]:  # Top 3 resistance levels
            if ask['distance_pct'] < 3:  # Within 3% of current price
                entry_price = ask['price'] * 1.001  # Entry slightly above wall
                
                # Find next resistance for target
                next_resistance = None
                for next_ask in significant_asks:
                    if next_ask['price'] > ask['price'] * 1.01:  # At least 1% higher
                        next_resistance = next_ask['price']
                        break
                
                if not next_resistance:
                    next_resistance = ask['price'] * 1.03  # Default 3% target
                
                # Set stop loss below current support
                stop_loss = mid_price * 0.98  # 2% below current price
                if significant_bids:
                    stop_loss = max(stop_loss, significant_bids[0]['price'] * 0.995)
                
                risk_reward = (next_resistance - entry_price) / (entry_price - stop_loss)
                
                if risk_reward >= 1.5:  # Minimum 1.5:1 R/R
                    confidence = "HIGH" if ask['quantity'] > ask.get('avg_size', 0) * 15 else "MEDIUM"
                    
                    setups.append({
                        'symbol': symbol,
                        'setup_type': 'BREAKOUT',
                        'direction': 'LONG',
                        'entry_price': entry_price,
                        'target_price': next_resistance,
                        'stop_loss': stop_loss,
                        'risk_reward_ratio': risk_reward,
                        'confidence': confidence,
                        'wall_size': ask['quantity'],
                        'wall_distance_pct': ask['distance_pct'],
                        'description': f"Break above ${entry_price:.6f if entry_price < 1 else entry_price:.4f if entry_price < 100 else entry_price:.2f} resistance wall"
                    })
        
        # Bearish breakout setups (breaking support walls)
        for bid in significant_bids[:3]:  # Top 3 support levels
            if bid['distance_pct'] < 3:  # Within 3% of current price
                entry_price = bid['price'] * 0.999  # Entry slightly below wall
                
                # Find next support for target
                next_support = None
                for next_bid in significant_bids:
                    if next_bid['price'] < bid['price'] * 0.99:  # At least 1% lower
                        next_support = next_bid['price']
                        break
                
                if not next_support:
                    next_support = bid['price'] * 0.97  # Default 3% target
                
                # Set stop loss above current resistance
                stop_loss = mid_price * 1.02  # 2% above current price
                if significant_asks:
                    stop_loss = min(stop_loss, significant_asks[0]['price'] * 1.005)
                
                risk_reward = (entry_price - next_support) / (stop_loss - entry_price)
                
                if risk_reward >= 1.5:  # Minimum 1.5:1 R/R
                    confidence = "HIGH" if bid['quantity'] > bid.get('avg_size', 0) * 15 else "MEDIUM"
                    
                    setups.append({
                        'symbol': symbol,
                        'setup_type': 'BREAKOUT',
                        'direction': 'SHORT',
                        'entry_price': entry_price,
                        'target_price': next_support,
                        'stop_loss': stop_loss,
                        'risk_reward_ratio': risk_reward,
                        'confidence': confidence,
                        'wall_size': bid['quantity'],
                        'wall_distance_pct': bid['distance_pct'],
                        'description': f"Break below ${entry_price:.6f if entry_price < 1 else entry_price:.4f if entry_price < 100 else entry_price:.2f} support wall"
                    })
        
        return setups
    
    def detect_bounce_setups(self, analysis: Dict, symbol: str) -> List[Dict]:
        """Detect bounce setups from strong support/resistance levels"""
        setups = []
        mid_price = analysis['mid_price']
        significant_bids = analysis.get('significant_bids', [])
        significant_asks = analysis.get('significant_asks', [])
        
        # Bounce from support setups
        for bid in significant_bids[:2]:  # Top 2 support levels
            if 0.5 <= bid['distance_pct'] <= 2:  # Between 0.5% and 2% below current price
                entry_price = bid['price'] * 1.002  # Entry slightly above support
                
                # Target nearest resistance
                target_price = mid_price * 1.015  # Default 1.5% target
                if significant_asks:
                    target_price = min(target_price, significant_asks[0]['price'] * 0.995)
                
                stop_loss = bid['price'] * 0.995  # Just below support
                
                risk_reward = (target_price - entry_price) / (entry_price - stop_loss)
                
                if risk_reward >= 2:  # Higher R/R required for bounces
                    confidence = "HIGH" if bid['quantity'] > bid.get('avg_size', 0) * 20 else "MEDIUM"
                    
                    setups.append({
                        'symbol': symbol,
                        'setup_type': 'BOUNCE',
                        'direction': 'LONG',
                        'entry_price': entry_price,
                        'target_price': target_price,
                        'stop_loss': stop_loss,
                        'risk_reward_ratio': risk_reward,
                        'confidence': confidence,
                        'wall_size': bid['quantity'],
                        'wall_distance_pct': bid['distance_pct'],
                        'description': f"Bounce from ${bid['price']:.6f if bid['price'] < 1 else bid['price']:.4f if bid['price'] < 100 else bid['price']:.2f} support wall"
                    })
        
        # Bounce from resistance setups (short)
        for ask in significant_asks[:2]:  # Top 2 resistance levels
            if 0.5 <= ask['distance_pct'] <= 2:  # Between 0.5% and 2% above current price
                entry_price = ask['price'] * 0.998  # Entry slightly below resistance
                
                # Target nearest support
                target_price = mid_price * 0.985  # Default 1.5% target
                if significant_bids:
                    target_price = max(target_price, significant_bids[0]['price'] * 1.005)
                
                stop_loss = ask['price'] * 1.005  # Just above resistance
                
                risk_reward = (entry_price - target_price) / (stop_loss - entry_price)
                
                if risk_reward >= 2:  # Higher R/R required for bounces
                    confidence = "HIGH" if ask['quantity'] > ask.get('avg_size', 0) * 20 else "MEDIUM"
                    
                    setups.append({
                        'symbol': symbol,
                        'setup_type': 'BOUNCE',
                        'direction': 'SHORT',
                        'entry_price': entry_price,
                        'target_price': target_price,
                        'stop_loss': stop_loss,
                        'risk_reward_ratio': risk_reward,
                        'confidence': confidence,
                        'wall_size': ask['quantity'],
                        'wall_distance_pct': ask['distance_pct'],
                        'description': f"Bounce from ${ask['price']:.6f if ask['price'] < 1 else ask['price']:.4f if ask['price'] < 100 else ask['price']:.2f} resistance wall"
                    })
        
        return setups
    
    def generate_setup_alert(self, setup: Dict) -> str:
        """Generate professional setup alert"""
        symbol = setup['symbol']
        setup_type = setup['setup_type']
        direction = setup['direction']
        entry = setup['entry_price']
        target = setup['target_price']
        stop = setup['stop_loss']
        rr = setup['risk_reward_ratio']
        confidence = setup['confidence']
        
        # Choose emoji based on setup type and direction
        if setup_type == "BREAKOUT":
            emoji = "🚀" if direction == "LONG" else "📉"
        else:  # BOUNCE
            emoji = "📈" if direction == "LONG" else "📊"
            
        alert = f"{emoji} <b>{setup_type} SETUP - {symbol}</b>\n\n"
        alert += f"📍 Direction: <b>{direction}</b>\n"
        alert += f"🎯 Entry: ${entry:.6f if entry < 1 else entry:.4f if entry < 100 else entry:.2f}\n"
        alert += f"🏆 Target: ${target:.6f if target < 1 else target:.4f if target < 100 else target:.2f}\n"
        alert += f"🛡️ Stop Loss: ${stop:.6f if stop < 1 else stop:.4f if stop < 100 else stop:.2f}\n"
        alert += f"⚡ Risk/Reward: <b>1:{rr:.1f}</b>\n"
        alert += f"🎯 Confidence: <b>{confidence}</b>\n\n"
        
        alert += f"📋 <i>{setup['description']}</i>\n\n"
        
        # Add setup-specific guidance
        if setup_type == "BREAKOUT":
            alert += "🔥 <i>Wait for clear break with volume</i>\n"
            alert += "⏰ <i>Monitor for follow-through</i>"
        else:  # BOUNCE
            alert += "⚖️ <i>Wait for bounce confirmation</i>\n"
            alert += "📏 <i>Tight risk management required</i>"
            
        return alert
    
    def store_setup(self, setup: Dict):
        """Store setup in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO orderbook_setups (
                    symbol, setup_type, direction, entry_price, target_price,
                    stop_loss, risk_reward_ratio, confidence, wall_size, wall_distance_pct, setup_sent
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                setup['symbol'], setup['setup_type'], setup['direction'],
                setup['entry_price'], setup['target_price'], setup['stop_loss'],
                setup['risk_reward_ratio'], setup['confidence'],
                setup['wall_size'], setup['wall_distance_pct'], 1
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Setup stored for {setup['symbol']}")
            
        except Exception as e:
            logger.error(f"Failed to store setup: {e}")


class OrderBookAnalyzer:
    """Order book analysis component"""

    def __init__(self):
        data_dir = os.environ.get('DATA_DIR', '.')
        self.db_path = os.path.join(data_dir, "orderbook_data.db")
        self.init_database()

        # Analysis thresholds
        self.thresholds = {
            'bid_ask_imbalance_strong': 0.25,     # 25% imbalance
            'bid_ask_imbalance_extreme': 0.5,     # 50% imbalance
            'spread_normal_bps': 2,               # 2 basis points normal
            'spread_wide_bps': 5,                 # 5 basis points wide
            'spread_extreme_bps': 10,             # 10 basis points extreme
            'large_wall_multiplier': 5,           # 5x average size
            'huge_wall_multiplier': 5,            # 5x average size (reduced from 10x)
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
            params = {"symbol": api_symbol, "limit": 2000}
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

    def calculate_entry_opportunities(self, symbol: str) -> Dict:
        """Calculate optimal entry prices for a symbol based on order book analysis"""
        order_book_data = self.get_order_book_data(symbol)
        if not order_book_data:
            return None
            
        bids = order_book_data['bids']
        asks = order_book_data['asks']
        mid_price = order_book_data['mid_price']
        spread_bps = order_book_data['spread_bps']
        
        # Calculate market pressure
        pressure = self.calculate_market_pressure(bids, asks, mid_price)
        
        # Find significant support/resistance levels
        significant_levels = self.identify_significant_levels(bids, asks, mid_price)
        
        # Calculate entry opportunities
        entries = {
            'symbol': symbol,
            'current_price': mid_price,
            'spread_bps': spread_bps,
            'long_entries': [],
            'short_entries': [],
            'market_pressure': pressure,
            'confidence': 'LOW'
        }
        
        # LONG ENTRY OPPORTUNITIES
        
        # 1. Support level entries (buy near strong bid walls)
        for level in significant_levels['support_levels']:
            if level['price'] < mid_price:  # Only below current price
                entry_score = self._calculate_entry_score(level, pressure, 'LONG')
                if entry_score > 0.5:  # Minimum threshold
                    entries['long_entries'].append({
                        'type': 'SUPPORT_BOUNCE',
                        'price': level['price'],
                        'size': level['size'],
                        'distance_pct': ((mid_price - level['price']) / mid_price) * 100,
                        'score': entry_score,
                        'confidence': 'HIGH' if entry_score > 0.7 else 'MEDIUM',
                        'reasoning': f"Strong support wall ({level['size']:.0f} USDT) at {level['price']:.4f}"
                    })
        
        # 2. Pressure-based entries (buy when selling pressure exhausted)
        if pressure.get('immediate_imbalance', 0) < -0.2:  # Strong selling pressure
            # Look for entry slightly below current price
            entry_price = mid_price * 0.998  # 0.2% below mid
            bid_support = sum(qty for price, qty in bids if price >= entry_price * 0.995)
            if bid_support > 10000:  # Decent support
                entries['long_entries'].append({
                    'type': 'PRESSURE_REVERSAL',
                    'price': entry_price,
                    'size': bid_support,
                    'distance_pct': 0.2,
                    'score': 0.6,
                    'confidence': 'MEDIUM',
                    'reasoning': f"Oversold pressure ({pressure['immediate_imbalance']:.1%}) with support"
                })
        
        # 3. Breakout entries (buy above resistance if pressure builds)
        if pressure.get('immediate_imbalance', 0) > 0.15:  # Building buying pressure
            for level in significant_levels['resistance_levels']:
                if level['price'] > mid_price and level['price'] < mid_price * 1.02:  # Within 2%
                    breakout_price = level['price'] * 1.001  # Just above resistance
                    entry_score = 0.65 + (pressure.get('immediate_imbalance', 0) * 0.5)
                    entries['long_entries'].append({
                        'type': 'BREAKOUT',
                        'price': breakout_price,
                        'size': level['size'],
                        'distance_pct': ((breakout_price - mid_price) / mid_price) * 100,
                        'score': min(entry_score, 0.9),
                        'confidence': 'HIGH' if entry_score > 0.75 else 'MEDIUM',
                        'reasoning': f"Breakout above {level['price']:.4f} with buying pressure"
                    })
        
        # SHORT ENTRY OPPORTUNITIES
        
        # 1. Resistance level entries (sell near strong ask walls)
        for level in significant_levels['resistance_levels']:
            if level['price'] > mid_price:  # Only above current price
                entry_score = self._calculate_entry_score(level, pressure, 'SHORT')
                if entry_score > 0.5:
                    entries['short_entries'].append({
                        'type': 'RESISTANCE_REJECT',
                        'price': level['price'],
                        'size': level['size'],
                        'distance_pct': ((level['price'] - mid_price) / mid_price) * 100,
                        'score': entry_score,
                        'confidence': 'HIGH' if entry_score > 0.7 else 'MEDIUM',
                        'reasoning': f"Strong resistance wall ({level['size']:.0f} USDT) at {level['price']:.4f}"
                    })
        
        # 2. Pressure-based short entries
        if pressure.get('immediate_imbalance', 0) > 0.2:  # Strong buying pressure (potentially exhausted)
            entry_price = mid_price * 1.002  # 0.2% above mid
            ask_resistance = sum(qty for price, qty in asks if price <= entry_price * 1.005)
            if ask_resistance > 10000:
                entries['short_entries'].append({
                    'type': 'PRESSURE_REVERSAL',
                    'price': entry_price,
                    'size': ask_resistance,
                    'distance_pct': 0.2,
                    'score': 0.6,
                    'confidence': 'MEDIUM',
                    'reasoning': f"Overbought pressure ({pressure['immediate_imbalance']:.1%}) with resistance"
                })
        
        # Sort entries by score
        entries['long_entries'].sort(key=lambda x: x['score'], reverse=True)
        entries['short_entries'].sort(key=lambda x: x['score'], reverse=True)
        
        # Overall confidence based on best opportunities
        best_scores = []
        if entries['long_entries']:
            best_scores.append(entries['long_entries'][0]['score'])
        if entries['short_entries']:
            best_scores.append(entries['short_entries'][0]['score'])
        
        if best_scores:
            max_score = max(best_scores)
            if max_score > 0.75:
                entries['confidence'] = 'HIGH'
            elif max_score > 0.6:
                entries['confidence'] = 'MEDIUM'
            else:
                entries['confidence'] = 'LOW'
        
        return entries
    
    def _calculate_entry_score(self, level: Dict, pressure: Dict, direction: str) -> float:
        """Calculate entry score for a specific level"""
        base_score = 0.5
        
        # Size factor (larger walls = better)
        size_factor = min(level['size'] / 50000, 0.3)  # Max 0.3 bonus for 50k+ walls
        
        # Pressure alignment
        pressure_bonus = 0
        if direction == 'LONG':
            # For longs, we want selling pressure to be exhausted (negative imbalance)
            if pressure.get('immediate_imbalance', 0) < 0:
                pressure_bonus = abs(pressure['immediate_imbalance']) * 0.2
        else:
            # For shorts, we want buying pressure to be exhausted (positive imbalance)
            if pressure.get('immediate_imbalance', 0) > 0:
                pressure_bonus = pressure['immediate_imbalance'] * 0.2
        
        # Distance factor (closer to current price = better for support/resistance)
        distance_pct = level.get('distance_pct', 5)
        distance_penalty = min(distance_pct / 100, 0.1)  # Max 0.1 penalty
        
        score = base_score + size_factor + pressure_bonus - distance_penalty
        return max(0, min(score, 1.0))  # Clamp between 0 and 1

    def calculate_depth_profile(self, bids: List[List[float]], asks: List[List[float]], mid_price: float) -> List[Dict]:
        """Build depth profile across multiple percentage bands around mid price"""
        depth_windows = [0.1, 0.25, 0.5, 0.75, 1.0, 5.0]
        profile = []
        for pct in depth_windows:
            bid_cutoff = mid_price * (1 - pct / 100)
            ask_cutoff = mid_price * (1 + pct / 100)
            bid_depth = sum(qty for price, qty in bids if price >= bid_cutoff)
            ask_depth = sum(qty for price, qty in asks if price <= ask_cutoff)
            total_depth = bid_depth + ask_depth
            imbalance = (bid_depth - ask_depth) / total_depth if total_depth else 0.0
            profile.append({
                'window_pct': pct,
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'total_depth': total_depth,
                'bid_depth_usd': bid_depth * mid_price,
                'ask_depth_usd': ask_depth * mid_price,
                'total_depth_usd': total_depth * mid_price,
                'imbalance': imbalance
            })
        return profile

    def _find_depth_entry(self, depth_profile: List[Dict], target_pct: float) -> Optional[Dict]:
        """Locate a depth profile entry for a given percentage band"""
        for entry in depth_profile:
            if abs(entry['window_pct'] - target_pct) < 1e-9:
                return entry
        return None

    def _format_pct_key(self, pct: float) -> str:
        """Convert a percentage value to a safe dict key suffix"""
        pct_str = f"{pct:.2f}".rstrip('0').rstrip('.')
        return f"{pct_str.replace('.', '_')}pct"

    def _format_usd_compact(self, value: float, show_sign: bool = False) -> str:
        """Format large USD values into compact human-readable form"""
        abs_value = abs(value)
        suffix = ''
        scaled = value
        if abs_value >= 1_000_000_000:
            scaled = value / 1_000_000_000
            suffix = 'B'
        elif abs_value >= 1_000_000:
            scaled = value / 1_000_000
            suffix = 'M'
        elif abs_value >= 1_000:
            scaled = value / 1_000
            suffix = 'K'
        else:
            formatted = f"{value:,.0f}"
            if show_sign and value > 0:
                return f"+{formatted}"
            return formatted

        formatted = f"{scaled:.1f}"
        if formatted.endswith('.0'):
            formatted = formatted[:-2]
        if show_sign and value > 0:
            formatted = f"+{formatted}"
        return f"{formatted}{suffix}"

    def calculate_market_pressure(self, bids: List[List[float]], asks: List[List[float]], mid_price: float) -> Dict:
        """Calculate various market pressure metrics"""
        pressure = {}
        depth_profile = self.calculate_depth_profile(bids, asks, mid_price)
        pressure['depth_profile'] = depth_profile

        for entry in depth_profile:
            pct_key = self._format_pct_key(entry['window_pct'])
            pressure[f'bid_depth_{pct_key}'] = entry['bid_depth']
            pressure[f'ask_depth_{pct_key}'] = entry['ask_depth']
            pressure[f'depth_imbalance_{pct_key}'] = entry['imbalance']

        immediate_entry = self._find_depth_entry(depth_profile, 0.1)
        if immediate_entry and immediate_entry['total_depth'] > 0:
            pressure['immediate_imbalance'] = immediate_entry['imbalance']

        short_entry = self._find_depth_entry(depth_profile, 1.0)
        if short_entry and short_entry['total_depth'] > 0:
            pressure['short_term_imbalance'] = short_entry['imbalance']

        medium_entry = self._find_depth_entry(depth_profile, 5.0)
        if medium_entry and medium_entry['total_depth'] > 0:
            pressure['medium_term_imbalance'] = medium_entry['imbalance']

        return pressure

    def _depth_bias_headline(self, avg_imbalance: float) -> str:
        """Translate average imbalance into a plain-language headline"""
        if avg_imbalance > 0.2:
            return "Buyers dominate the near book"
        if avg_imbalance > 0.08:
            return "Buyers hold a slight edge nearby"
        if avg_imbalance < -0.2:
            return "Sellers dominate the near book"
        if avg_imbalance < -0.08:
            return "Sellers hold a slight edge nearby"
        return "Depth looks balanced near price"

    def generate_depth_summary(self, depth_profile: List[Dict], mid_price: float) -> Optional[str]:
        """Create a short narrative summary of near-price depth"""
        key_windows = [0.1, 0.25, 0.5, 0.75]
        entries = []
        for pct in key_windows:
            entry = self._find_depth_entry(depth_profile, pct)
            if entry and entry['total_depth'] > 0:
                entries.append(entry)

        if not entries:
            return None

        avg_imbalance = sum(e['imbalance'] for e in entries) / len(entries)
        headline = self._depth_bias_headline(avg_imbalance)

        near_entry = self._find_depth_entry(depth_profile, 0.75)
        liquidity_note = ""
        if near_entry and near_entry['total_depth_usd'] > 0:
            liquidity_note = f" (~${near_entry['total_depth_usd']:,.0f} within 0.75%)"

        snapshots = []
        for entry in entries:
            imbalance_pct = entry['imbalance'] * 100
            if imbalance_pct > 5:
                direction = "buy"
            elif imbalance_pct < -5:
                direction = "sell"
            else:
                direction = "flat"
            pct_label = f"{entry['window_pct']:.2f}".rstrip('0').rstrip('.')
            snapshots.append(f"{pct_label}% {direction} {imbalance_pct:+.0f}%")

        snapshot_text = '; '.join(snapshots)
        return f"{headline}{liquidity_note}. {snapshot_text}."

    def detect_significant_levels(self, bids: List[List[float]], asks: List[List[float]], mid_price: float) -> Dict:
        """Detect significant support and resistance levels by aggregating orders in price zones for trading"""
        
        def get_dynamic_thresholds(bids, asks, mid_price):
            """Calculate dynamic thresholds based on market conditions"""
            # Calculate total volume in top 50 levels
            total_bid_volume = sum(price * qty for price, qty in bids[:50])
            total_ask_volume = sum(price * qty for price, qty in asks[:50])
            total_volume = total_bid_volume + total_ask_volume
            
            # Dynamic zone size based on price (higher prices need wider zones)
            if mid_price > 100000:  # BTC level
                zone_pct = 0.05  # 0.05% for high-value assets
            elif mid_price > 1000:  # ETH level  
                zone_pct = 0.1   # 0.1% for mid-value assets
            else:
                zone_pct = 0.2   # 0.2% for low-value assets
            
            # Dynamic thresholds based on volume
            if total_volume > 50000000:  # High volume day
                major_threshold = 500000   # 500K for major levels
                minor_threshold = 100000   # 100K for minor levels
            elif total_volume > 20000000:  # Medium volume
                major_threshold = 300000   # 300K for major levels
                minor_threshold = 75000    # 75K for minor levels
            else:  # Low volume
                major_threshold = 200000   # 200K for major levels
                minor_threshold = 50000    # 50K for minor levels
            
            return zone_pct, major_threshold, minor_threshold
        
        def aggregate_orders_by_zone(orders, zone_pct=0.1):
            """Aggregate orders within zone_pct price range"""
            if not orders:
                return []
            
            zones = []
            sorted_orders = sorted(orders, key=lambda x: x[0])  # Sort by price
            
            current_zone = {'min_price': sorted_orders[0][0], 'max_price': sorted_orders[0][0], 
                          'total_qty': sorted_orders[0][1], 'orders': [sorted_orders[0]]}
            
            for price, qty in sorted_orders[1:]:
                # Check if this order is within zone_pct of current zone
                zone_center = (current_zone['min_price'] + current_zone['max_price']) / 2
                
                if abs(price - zone_center) / zone_center * 100 <= zone_pct:
                    # Add to current zone
                    current_zone['min_price'] = min(current_zone['min_price'], price)
                    current_zone['max_price'] = max(current_zone['max_price'], price)
                    current_zone['total_qty'] += qty
                    current_zone['orders'].append([price, qty])
                else:
                    # Start new zone
                    zones.append(current_zone)
                    current_zone = {'min_price': price, 'max_price': price, 
                                  'total_qty': qty, 'orders': [[price, qty]]}
            
            zones.append(current_zone)  # Add last zone
            return zones

        # Get dynamic thresholds
        zone_pct, major_threshold, minor_threshold = get_dynamic_thresholds(bids, asks, mid_price)
        
        # Aggregate bids and asks into price zones
        bid_zones = aggregate_orders_by_zone(bids, zone_pct)
        ask_zones = aggregate_orders_by_zone(asks, zone_pct)

        major_bids = []
        minor_bids = []
        major_asks = []
        minor_asks = []
        
        # Analyze bid zones
        for zone in bid_zones:
            avg_price = (zone['min_price'] + zone['max_price']) / 2
            total_usdt = avg_price * zone['total_qty']
            distance_pct = ((mid_price - avg_price) / mid_price) * 100
            
            if total_usdt >= minor_threshold:
                level_data = {
                    'price': avg_price,
                    'quantity': zone['total_qty'],
                    'distance_pct': distance_pct,
                    'usdt_value': total_usdt,
                    'order_count': len(zone['orders']),
                    'strength': 'MAJOR' if total_usdt >= major_threshold else 'MINOR'
                }
                
                if total_usdt >= major_threshold:
                    major_bids.append(level_data)
                else:
                    minor_bids.append(level_data)
        
        # Analyze ask zones  
        for zone in ask_zones:
            avg_price = (zone['min_price'] + zone['max_price']) / 2
            total_usdt = avg_price * zone['total_qty']
            distance_pct = ((avg_price - mid_price) / mid_price) * 100
            
            if total_usdt >= minor_threshold:
                level_data = {
                    'price': avg_price,
                    'quantity': zone['total_qty'],
                    'distance_pct': distance_pct,
                    'usdt_value': total_usdt,
                    'order_count': len(zone['orders']),
                    'strength': 'MAJOR' if total_usdt >= major_threshold else 'MINOR'
                }
                
                if total_usdt >= major_threshold:
                    major_asks.append(level_data)
                else:
                    minor_asks.append(level_data)

        # Sort by distance from current price
        major_bids.sort(key=lambda x: x['distance_pct'])
        minor_bids.sort(key=lambda x: x['distance_pct'])
        major_asks.sort(key=lambda x: x['distance_pct'])
        minor_asks.sort(key=lambda x: x['distance_pct'])

        # Combine for backward compatibility
        significant_bids = major_bids + minor_bids
        significant_asks = major_asks + minor_asks

        return {
            'significant_bids': significant_bids[:10],
            'significant_asks': significant_asks[:10],
            'major_bids': major_bids[:5],
            'minor_bids': minor_bids[:5],
            'major_asks': major_asks[:5],
            'minor_asks': minor_asks[:5],
            'large_bid_walls': len(major_bids),
            'large_ask_walls': len(major_asks),
            'total_bid_walls': len(significant_bids),
            'total_ask_walls': len(significant_asks),
            'thresholds': {
                'major': major_threshold,
                'minor': minor_threshold,
                'zone_pct': zone_pct
            }
        }

    def identify_significant_levels(self, bids: List[List[float]], asks: List[List[float]], mid_price: float) -> Dict:
        """Identify significant support and resistance levels for entry analysis"""
        # Note: Using zone aggregation instead of individual order thresholds
        
        support_levels = []
        resistance_levels = []
        
        # Aggregate orders into zones first
        def aggregate_orders_by_zone(orders, zone_pct=0.1):
            if not orders:
                return []
            zones = []
            sorted_orders = sorted(orders, key=lambda x: x[0])
            current_zone = {'min_price': sorted_orders[0][0], 'max_price': sorted_orders[0][0], 
                          'total_qty': sorted_orders[0][1], 'orders': [sorted_orders[0]]}
            
            for price, qty in sorted_orders[1:]:
                zone_center = (current_zone['min_price'] + current_zone['max_price']) / 2
                if abs(price - zone_center) / zone_center * 100 <= zone_pct:
                    current_zone['min_price'] = min(current_zone['min_price'], price)
                    current_zone['max_price'] = max(current_zone['max_price'], price)
                    current_zone['total_qty'] += qty
                    current_zone['orders'].append([price, qty])
                else:
                    zones.append(current_zone)
                    current_zone = {'min_price': price, 'max_price': price, 
                                  'total_qty': qty, 'orders': [[price, qty]]}
            zones.append(current_zone)
            return zones

        bid_zones = aggregate_orders_by_zone(bids)
        ask_zones = aggregate_orders_by_zone(asks)

        # Find support levels from bid zones
        for zone in bid_zones:
            avg_price = (zone['min_price'] + zone['max_price']) / 2
            total_usdt = avg_price * zone['total_qty']
            distance_pct = ((mid_price - avg_price) / mid_price) * 100
            
            if total_usdt >= 50000 and distance_pct <= 5 and distance_pct >= 0:
                support_levels.append({
                    'price': avg_price,
                    'size': total_usdt,
                    'quantity': zone['total_qty'],
                    'distance_pct': distance_pct,
                    'order_count': len(zone['orders'])
                })
        
        # Find resistance levels from ask zones
        for zone in ask_zones:
            avg_price = (zone['min_price'] + zone['max_price']) / 2
            total_usdt = avg_price * zone['total_qty']
            distance_pct = ((avg_price - mid_price) / mid_price) * 100
            
            if total_usdt >= 50000 and distance_pct <= 5 and distance_pct >= 0:
                resistance_levels.append({
                    'price': avg_price,
                    'size': total_usdt,
                    'quantity': zone['total_qty'],
                    'distance_pct': distance_pct,
                    'order_count': len(zone['orders'])
                })
        
        # Sort by distance from current price
        support_levels.sort(key=lambda x: x['distance_pct'])
        resistance_levels.sort(key=lambda x: x['distance_pct'])
        
        # Remove levels too close to each other (within 0.5%)
        def filter_close_levels(levels):
            filtered = []
            for level in levels:
                if not any(abs(level['distance_pct'] - existing['distance_pct']) < 0.5 for existing in filtered):
                    filtered.append(level)
            return filtered
        
        return {
            'support_levels': filter_close_levels(support_levels)[:5],
            'resistance_levels': filter_close_levels(resistance_levels)[:5]
        }

    def get_deepest_walls(self, symbol: str = 'BTC/USDT', limit: int = 5) -> Optional[Dict]:
        """Fetch and return the deepest bid and ask walls sorted by size."""
        ob_data = self.get_order_book_data(symbol)
        if not ob_data:
            return None
        bids = ob_data['bids']
        asks = ob_data['asks']
        mid_price = ob_data['mid_price']
        levels = self.detect_significant_levels(bids, asks, mid_price)
        # Sort by USDT value (price * quantity) instead of just quantity
        deepest_bids = sorted(levels['significant_bids'], key=lambda x: x['price'] * x['quantity'], reverse=True)[:limit]
        deepest_asks = sorted(levels['significant_asks'], key=lambda x: x['price'] * x['quantity'], reverse=True)[:limit]
        return {
            'symbol': symbol,
            'current_price': mid_price,
            'bids': deepest_bids,
            'asks': deepest_asks
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
        pressure_metrics = self.calculate_market_pressure(bids, asks, mid_price)
        analysis.update(pressure_metrics)
        analysis.update(self.detect_significant_levels(bids, asks, mid_price))
        depth_profile = pressure_metrics.get('depth_profile')
        if depth_profile:
            depth_summary = self.generate_depth_summary(depth_profile, mid_price)
            if depth_summary:
                analysis['depth_summary'] = depth_summary
        # ... weighted prices, skew, sentiment, liquidity score ...
        analysis['market_sentiment'] = self.classify_market_sentiment(analysis)
        analysis['liquidity_score'] = self.calculate_liquidity_score(analysis, symbol)

        return analysis

    def generate_report(self, analysis: Dict) -> str:
        """Generate order book report"""
        # Always generate a full report regardless of sentiment
        sentiment = analysis.get('market_sentiment', 'NEUTRAL')
        
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

        depth_profile = analysis.get('depth_profile', [])
        if depth_profile:
            report += "\n📉 Depth Profile (distance from mid):\n"
            for pct in [0.1, 0.25, 0.5, 0.75]:
                entry = self._find_depth_entry(depth_profile, pct)
                if not entry or entry['total_depth'] == 0:
                    continue
                pct_label = f"{pct:.2f}".rstrip('0').rstrip('.')
                bias_pct = entry['imbalance'] * 100
                if bias_pct > 5:
                    bias_label = "BUY"
                    bias_emoji = "🟢"
                elif bias_pct < -5:
                    bias_label = "SELL"
                    bias_emoji = "🔴"
                else:
                    bias_label = "FLAT"
                    bias_emoji = "⚪️"
                total_text = self._format_usd_compact(entry['total_depth_usd'])
                bid_text = self._format_usd_compact(entry['bid_depth_usd'])
                ask_text = self._format_usd_compact(entry['ask_depth_usd'])
                net_text = self._format_usd_compact(entry['bid_depth_usd'] - entry['ask_depth_usd'], show_sign=True)
                report += (
                    f"  {pct_label}% band\n"
                    f"    {bias_emoji} Bias: {bias_label} ({bias_pct:+.0f}%)\n"
                    f"    💧 Depth: {total_text} USDT\n"
                    f"    🟢 Bids: {bid_text} | 🔴 Asks: {ask_text}\n"
                    f"    ⚖️ Net: {net_text}\n\n"
                )
            depth_summary = analysis.get('depth_summary')
            if depth_summary:
                report += f"🧭 Depth Read: {depth_summary}\n"
        
        # Significant levels with prices
        significant_bids = analysis.get('significant_bids', [])
        significant_asks = analysis.get('significant_asks', [])
        # Show minor levels only
        minor_bids = analysis.get('minor_bids', [])
        minor_asks = analysis.get('minor_asks', [])
        
        if minor_bids:
            report += f"\n🔵 <b>Minor Support Levels:</b>\n"
            for bid in minor_bids[:2]:  # Show top 2 minor
                decimals = 6 if bid['price'] < 1 else 4 if bid['price'] < 100 else 2
                usdt_value = bid['usdt_value']
                report += f"  ${bid['price']:.{decimals}f} ({bid['distance_pct']:.2f}% below) - ${usdt_value:,.0f} USDT [{bid['order_count']} orders]\n"
                
        if minor_asks:
            report += f"\n🟠 <b>Minor Resistance Levels:</b>\n"
            for ask in minor_asks[:2]:  # Show top 2 minor
                decimals = 6 if ask['price'] < 1 else 4 if ask['price'] < 100 else 2
                usdt_value = ask['usdt_value']
                report += f"  ${ask['price']:.{decimals}f} ({ask['distance_pct']:.2f}% above) - ${usdt_value:,.0f} USDT [{ask['order_count']} orders]\n"
        
        # Liquidity and sentiment
        report += f"\n💧 Liquidity Score: {analysis.get('liquidity_score', 0):.0f}/100\n"
        report += f"🎯 Sentiment: <b>{sentiment}</b>"
        
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


class EMAAnalyzer:
    """EMA touch/cross detection and alerting"""

    def __init__(self):
        data_dir = os.environ.get('DATA_DIR', '.')
        self.db_path = os.path.join(data_dir, "orderbook_data.db")
        self.init_database()
        self.ema_periods = [12, 20, 50, 100, 200]
        self.last_ema_states = {}
    
    def init_database(self):
        """Initialize EMA alerts database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ema_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                ema_period INTEGER,
                price REAL,
                ema_value REAL,
                alert_type TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def detect_ema_touches(self, df, symbol: str) -> List[Dict]:
        """Detect when price touches or crosses EMAs"""
        if len(df) < 2:
            return []
        
        alerts = []
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        for period in self.ema_periods:
            ema_col = f'ema_{period}'
            if ema_col not in df.columns:
                continue
                
            current_ema = current[ema_col]
            previous_ema = previous[ema_col]
            current_price = current['close']
            previous_price = previous['close']
            
            if pd.isna(current_ema) or pd.isna(previous_ema):
                continue
            
            # Check for touch (price within 0.2% of EMA) - increased threshold for crypto volatility
            touch_threshold = current_ema * 0.002  # 0.2%
            is_touching = abs(current_price - current_ema) <= touch_threshold
            
            # Check for cross - price crossing EMA line
            crossed_above = (previous_price < previous_ema and current_price > current_ema)
            crossed_below = (previous_price > previous_ema and current_price < current_ema)
            
            # Enhanced debug logging
            if crossed_above or crossed_below:
                logger.info(f"🚨 EMA CROSS DETECTED: {symbol} EMA-{period} - prev_price:{previous_price:.4f} curr_price:{current_price:.4f} prev_ema:{previous_ema:.4f} curr_ema:{current_ema:.4f}")
            elif is_touching:
                logger.debug(f"📍 EMA TOUCH: {symbol} EMA-{period} - price:{current_price:.4f} ema:{current_ema:.4f} diff:{abs(current_price - current_ema):.4f} threshold:{touch_threshold:.4f}")
            
            # Determine alert type
            alert_type = None
            if crossed_above:
                alert_type = "CROSS_ABOVE"
            elif crossed_below:
                alert_type = "CROSS_BELOW"
            elif is_touching and not self._was_touching_recently(symbol, period):
                if current_price > current_ema:
                    alert_type = "TOUCH_FROM_ABOVE"
                else:
                    alert_type = "TOUCH_FROM_BELOW"
            
            if alert_type:
                alerts.append({
                    'symbol': symbol,
                    'ema_period': period,
                    'price': current_price,
                    'ema_value': current_ema,
                    'alert_type': alert_type,
                    'rsi': current.get('rsi', 0),
                    'volume_ratio': current.get('volume_ratio', 1)
                })
                
                # Store alert in database
                self._store_ema_alert(symbol, period, current_price, current_ema, alert_type)
        
        return alerts
    
    def _was_touching_recently(self, symbol: str, period: int) -> bool:
        """Check if we already alerted for this EMA touch recently"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM ema_alerts 
                WHERE symbol = ? AND ema_period = ? 
                AND alert_type LIKE 'TOUCH%'
                AND timestamp > datetime('now', '-1 hour')
            ''', (symbol, period))
            result = cursor.fetchone()
            conn.close()
            return result[0] > 0
        except Exception:
            return False
    
    def _store_ema_alert(self, symbol: str, period: int, price: float, ema_value: float, alert_type: str):
        """Store EMA alert in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO ema_alerts (symbol, ema_period, price, ema_value, alert_type)
                VALUES (?, ?, ?, ?, ?)
            ''', (symbol, period, price, ema_value, alert_type))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to store EMA alert: {e}")
    
    def generate_ema_alert(self, alert: Dict) -> str:
        """Generate EMA alert message"""
        symbol = alert['symbol']
        period = alert['ema_period']
        price = alert['price']
        ema_value = alert['ema_value']
        alert_type = alert['alert_type']
        rsi = alert.get('rsi', 0)
        volume_ratio = alert.get('volume_ratio', 1)
        
        # Emoji mapping
        emoji_map = {
            'CROSS_ABOVE': '🚀',
            'CROSS_BELOW': '📉',
            'TOUCH_FROM_ABOVE': '🎯',
            'TOUCH_FROM_BELOW': '🎯'
        }
        
        emoji = emoji_map.get(alert_type, '📊')
        
        # Format message based on alert type
        if 'CROSS' in alert_type:
            action = "CROSSED ABOVE" if "ABOVE" in alert_type else "CROSSED BELOW"
            alert_msg = f"{emoji} <b>EMA CROSS ALERT</b>\n\n"
        else:
            action = "TOUCHING"
            alert_msg = f"{emoji} <b>EMA TOUCH ALERT</b>\n\n"
        
        alert_msg += f"💎 Symbol: <b>{symbol}</b>\n"
        alert_msg += f"📈 Price: ${price:.6f if price < 1 else price:.4f if price < 100 else price:.2f}\n"
        alert_msg += f"📊 EMA-{period}: ${ema_value:.6f if ema_value < 1 else ema_value:.4f if ema_value < 100 else ema_value:.2f}\n"
        alert_msg += f"🎯 Action: <b>{action} EMA-{period}</b>\n"
        
        # Add technical context
        if rsi > 0:
            alert_msg += f"📈 RSI: {rsi:.1f}\n"
        if volume_ratio > 1:
            alert_msg += f"📊 Volume: {volume_ratio:.1f}x average\n"
        
        # Add interpretation
        alert_msg += "\n💡 <b>Interpretation:</b>\n"
        
        if alert_type == "CROSS_ABOVE":
            if period in [12, 20]:
                alert_msg += "🚀 <i>Short-term bullish momentum</i>\n"
            elif period in [50, 100]:
                alert_msg += "📈 <i>Medium-term trend turning bullish</i>\n"
            else:  # 200
                alert_msg += "🔥 <i>Long-term trend reversal - major bullish signal</i>\n"
        elif alert_type == "CROSS_BELOW":
            if period in [12, 20]:
                alert_msg += "📉 <i>Short-term bearish momentum</i>\n"
            elif period in [50, 100]:
                alert_msg += "⚠️ <i>Medium-term trend turning bearish</i>\n"
            else:  # 200
                alert_msg += "🔴 <i>Long-term trend reversal - major bearish signal</i>\n"
        else:  # TOUCH
            if period in [50, 100, 200]:
                alert_msg += "⚖️ <i>Testing key support/resistance level</i>\n"
                alert_msg += "👀 <i>Watch for bounce or break</i>"
            else:
                alert_msg += "🎯 <i>Price interacting with dynamic level</i>"
        
        return alert_msg


class RSIAnalyzer:
    """RSI overbought/oversold detection and alerting"""

    def __init__(self):
        data_dir = os.environ.get('DATA_DIR', '.')
        self.db_path = os.path.join(data_dir, "orderbook_data.db")
        self.init_database()
        self.overbought_threshold = 70
        self.oversold_threshold = 30
        self.last_rsi_states = {}
    
    def init_database(self):
        """Initialize RSI alerts database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rsi_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                rsi_value REAL,
                price REAL,
                alert_type TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def detect_rsi_signals(self, df, symbol: str) -> List[Dict]:
        """Detect RSI overbought/oversold conditions"""
        if len(df) < 2 or 'rsi' not in df.columns:
            return []
        
        alerts = []
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        current_rsi = current['rsi']
        previous_rsi = previous['rsi']
        current_price = current['close']
        
        if pd.isna(current_rsi) or pd.isna(previous_rsi):
            return []
        
        # Check for RSI entering overbought territory
        if current_rsi >= self.overbought_threshold and previous_rsi < self.overbought_threshold:
            if not self._was_alerted_recently(symbol, 'OVERBOUGHT'):
                alerts.append({
                    'symbol': symbol,
                    'rsi_value': current_rsi,
                    'price': current_price,
                    'alert_type': 'OVERBOUGHT',
                    'volume_ratio': current.get('volume_ratio', 1)
                })
                self._store_rsi_alert(symbol, current_rsi, current_price, 'OVERBOUGHT')
        
        # Check for RSI entering oversold territory
        elif current_rsi <= self.oversold_threshold and previous_rsi > self.oversold_threshold:
            if not self._was_alerted_recently(symbol, 'OVERSOLD'):
                alerts.append({
                    'symbol': symbol,
                    'rsi_value': current_rsi,
                    'price': current_price,
                    'alert_type': 'OVERSOLD',
                    'volume_ratio': current.get('volume_ratio', 1)
                })
                self._store_rsi_alert(symbol, current_rsi, current_price, 'OVERSOLD')
        
        # Check for RSI exiting extreme territories (potential reversal signals)
        elif previous_rsi >= self.overbought_threshold and current_rsi < self.overbought_threshold:
            if not self._was_alerted_recently(symbol, 'EXIT_OVERBOUGHT'):
                alerts.append({
                    'symbol': symbol,
                    'rsi_value': current_rsi,
                    'price': current_price,
                    'alert_type': 'EXIT_OVERBOUGHT',
                    'volume_ratio': current.get('volume_ratio', 1)
                })
                self._store_rsi_alert(symbol, current_rsi, current_price, 'EXIT_OVERBOUGHT')
        
        elif previous_rsi <= self.oversold_threshold and current_rsi > self.oversold_threshold:
            if not self._was_alerted_recently(symbol, 'EXIT_OVERSOLD'):
                alerts.append({
                    'symbol': symbol,
                    'rsi_value': current_rsi,
                    'price': current_price,
                    'alert_type': 'EXIT_OVERSOLD',
                    'volume_ratio': current.get('volume_ratio', 1)
                })
                self._store_rsi_alert(symbol, current_rsi, current_price, 'EXIT_OVERSOLD')
        
        return alerts
    
    def _was_alerted_recently(self, symbol: str, alert_type: str) -> bool:
        """Check if we already alerted for this RSI condition recently"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM rsi_alerts 
                WHERE symbol = ? AND alert_type = ?
                AND timestamp > datetime('now', '-2 hours')
            ''', (symbol, alert_type))
            
            count = cursor.fetchone()[0]
            conn.close()
            return count > 0
        except Exception as e:
            logger.error(f"Error checking recent RSI alerts: {e}")
            return False
    
    def _store_rsi_alert(self, symbol: str, rsi_value: float, price: float, alert_type: str):
        """Store RSI alert in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO rsi_alerts (symbol, rsi_value, price, alert_type)
                VALUES (?, ?, ?, ?)
            ''', (symbol, rsi_value, price, alert_type))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error storing RSI alert: {e}")
    
    def format_rsi_alert(self, alert: Dict) -> str:
        """Format RSI alert message for Telegram"""
        symbol = alert['symbol']
        rsi_value = alert['rsi_value']
        price = alert['price']
        alert_type = alert['alert_type']
        volume_ratio = alert.get('volume_ratio', 1)
        
        # Base message
        if alert_type == 'OVERBOUGHT':
            alert_msg = f"🔴 <b>RSI OVERBOUGHT</b> 🔴\n"
            alert_msg += f"🪙 <b>{symbol}</b>\n"
            alert_msg += f"💰 Price: <b>${price:.4f}</b>\n"
            alert_msg += f"📊 RSI: <b>{rsi_value:.1f}</b> (>{self.overbought_threshold})\n"
            alert_msg += f"📈 Volume: <b>{volume_ratio:.1f}x</b>\n\n"
            alert_msg += "⚠️ <i>Potential selling pressure ahead</i>\n"
            alert_msg += "🎯 <i>Consider taking profits or wait for pullback</i>"
            
        elif alert_type == 'OVERSOLD':
            alert_msg = f"🟢 <b>RSI OVERSOLD</b> 🟢\n"
            alert_msg += f"🪙 <b>{symbol}</b>\n"
            alert_msg += f"💰 Price: <b>${price:.4f}</b>\n"
            alert_msg += f"📊 RSI: <b>{rsi_value:.1f}</b> (<{self.oversold_threshold})\n"
            alert_msg += f"📈 Volume: <b>{volume_ratio:.1f}x</b>\n\n"
            alert_msg += "💎 <i>Potential buying opportunity</i>\n"
            alert_msg += "🎯 <i>Consider entry or wait for reversal confirmation</i>"
            
        elif alert_type == 'EXIT_OVERBOUGHT':
            alert_msg = f"🟡 <b>RSI RECOVERY</b> 🟡\n"
            alert_msg += f"🪙 <b>{symbol}</b>\n"
            alert_msg += f"💰 Price: <b>${price:.4f}</b>\n"
            alert_msg += f"📊 RSI: <b>{rsi_value:.1f}</b> (cooling down)\n"
            alert_msg += f"📈 Volume: <b>{volume_ratio:.1f}x</b>\n\n"
            alert_msg += "🔄 <i>Exiting overbought territory</i>\n"
            alert_msg += "👀 <i>Potential continuation or reversal</i>"
            
        elif alert_type == 'EXIT_OVERSOLD':
            alert_msg = f"🟡 <b>RSI RECOVERY</b> 🟡\n"
            alert_msg += f"🪙 <b>{symbol}</b>\n"
            alert_msg += f"💰 Price: <b>${price:.4f}</b>\n"
            alert_msg += f"📊 RSI: <b>{rsi_value:.1f}</b> (recovering)\n"
            alert_msg += f"📈 Volume: <b>{volume_ratio:.1f}x</b>\n\n"
            alert_msg += "🔄 <i>Exiting oversold territory</i>\n"
            alert_msg += "🚀 <i>Potential bullish momentum building</i>"
        
        return alert_msg


class EnhancedTelegramBot:
    def __init__(self, bot_token, chat_id):
        self.bot = Bot(token=bot_token)
        self.chat_id = chat_id
        self.exchange = ccxt.binance()
        self.symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
            'HYPE/USDT', 'PENGU/USDT', 'XRP/USDT', 'MKR/USDT', 'AAVE/USDT',
            'DOGE/USDT', 'APT/USDT', 'XLM/USDT', 'QNT/USDT', 'SUI/USDT',
            'WIF/USDT', 'TON/USDT', 'KAITO/USDT', 'LINK/USDT', 'BCH/USDT'
        ]
        self.lookback = 20
        self.volume_threshold = 1.5
        self.running = False
        self.orderbook_running = False

        self.orderbook_analyzer = OrderBookAnalyzer()
        self.dominance_analyzer = USDTDominanceAnalyzer()
        self.volume_analyzer = VolumeSurgeAnalyzer()
        self.rotation_analyzer = CapitalRotationAnalyzer()
        self.setup_analyzer = OrderBookSetupAnalyzer()
        self.ema_analyzer = EMAAnalyzer()
        self.rsi_analyzer = RSIAnalyzer()
        self.last_macd_hist = None
        self.last_vwap_price = None
        self.dominance_running = False
        self.volume_surge_running = False
        self.rotation_running = False
        self.ema_running = False
        self.rsi_running = False
        self.setup_running = False
        self.last_dominance_sentiment = None

        application = Application.builder().token(bot_token).build()

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
        
        # Calculate EMAs
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_100'] = df['close'].ewm(span=100).mean()
        df['ema_200'] = df['close'].ewm(span=200).mean()
        
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
        rsi_allows_bullish = current['rsi'] < 75  # Not overbought
        rsi_allows_bearish = current['rsi'] > 25  # Not oversold
        
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
        trend_aligns_bullish = higher_tf_trend in ["bullish", "neutral"] or higher_tf_trend is None  # Allow neutral for sideways breakouts
        trend_aligns_bearish = higher_tf_trend in ["bearish", "neutral"] or higher_tf_trend is None
        
        # USDT dominance sentiment filter
        market_sentiment = self.get_current_market_sentiment()
        sentiment_allows_bullish = market_sentiment in ["BULLISH", "NEUTRAL", None]  # Don't trade bullish when USDT.D is spiking
        sentiment_allows_bearish = market_sentiment in ["BEARISH", "NEUTRAL", None]   # Don't trade bearish when USDT.D is falling
        
        # Main breakout logic - focus on price structure with multi-timeframe and sentiment confirmation
        bullish_breakout = (
            resistance_break and 
            rsi_allows_bullish and
            bullish_price_action and
            moderate_volume and
            trend_aligns_bullish and  # Higher timeframe must align or be neutral
            sentiment_allows_bullish  # Market sentiment must not be bearish (USDT.D spiking)
        )
        
        bearish_breakout = (
            support_break and
            rsi_allows_bearish and  
            bearish_price_action and
            moderate_volume and
            trend_aligns_bearish and  # Higher timeframe must align or be neutral
            sentiment_allows_bearish  # Market sentiment must not be bullish (USDT.D falling)
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
            if market_sentiment == "BULLISH":
                confirmations.append("USDT.D↘")
            
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
            if market_sentiment == "BEARISH":
                confirmations.append("USDT.D↗")
                
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

    async def dominance_monitor(self):
        """Monitor USDT dominance for sentiment shifts every hour"""
        while self.dominance_running:
            try:
                logger.info("Analyzing USDT dominance for market sentiment...")
                
                # Get dominance data
                df = self.dominance_analyzer.get_usdt_dominance_data()
                if df is not None and len(df) >= 20:
                    analysis = self.dominance_analyzer.analyze_dominance_sentiment(df)
                    
                    if analysis:
                        current_sentiment = analysis['sentiment']
                        signal_strength = analysis['signal_strength']
                        
                        # Only alert on sentiment changes (excluding WEAK) or strong signals
                        should_alert = (
                            (
                                self.last_dominance_sentiment != current_sentiment
                                and signal_strength != "WEAK"
                            )
                            or signal_strength == "STRONG"
                        )
                        
                        if should_alert:
                            alert = self.dominance_analyzer.generate_dominance_alert(analysis)
                            await self.send_alert(alert)
                            self.dominance_analyzer.store_analysis(analysis)
                            logger.info(f"USDT dominance alert sent: {current_sentiment}")
                            
                        self.last_dominance_sentiment = current_sentiment
                        
                else:
                    logger.warning("Failed to get USDT dominance data")
                    
            except Exception as e:
                logger.error(f"Dominance monitoring error: {e}")
                
            await asyncio.sleep(3600)  # 1 hour

    async def volume_surge_monitor(self):
        """Monitor all symbols for volume surges every 30 minutes"""
        while self.volume_surge_running:
            try:
                logger.info("Scanning all symbols for volume surges...")
                surge_count = 0
                
                for symbol in self.symbols:
                    try:
                        # Get OHLCV data for volume analysis
                        df = self.get_ohlcv(symbol, timeframe='1h', limit=50)
                        if df is None or len(df) < 21:
                            continue
                            
                        # Analyze for volume surge
                        surge_analysis = self.volume_analyzer.analyze_volume_surge(symbol, df)
                        
                        if surge_analysis:
                            # Generate and send alert
                            alert = self.volume_analyzer.generate_volume_alert(surge_analysis)
                            await self.send_alert(alert)
                            
                            # Store in database
                            self.volume_analyzer.store_surge(surge_analysis)
                            
                            surge_count += 1
                            logger.info(f"Volume surge detected for {symbol}: {surge_analysis['volume_ratio']:.1f}x")
                            
                    except Exception as e:
                        logger.error(f"Error analyzing volume for {symbol}: {e}")
                        
                if surge_count > 0:
                    logger.info(f"Volume surge monitoring completed: {surge_count} alerts sent")
                else:
                    logger.info("Volume surge monitoring completed: No significant surges detected")
                    
            except Exception as e:
                logger.error(f"Volume surge monitoring error: {e}")
                
            await asyncio.sleep(1800)  # 30 minutes

    async def rotation_monitor(self):
        """Monitor capital rotation across the watchlist every 30 minutes."""
        while self.rotation_running:
            try:
                logger.info("Analyzing capital rotation across watchlist...")
                analysis = self.rotation_analyzer.analyze_rotation(self.symbols, self.get_ohlcv, timeframe='1h')
                if analysis:
                    alert = self.rotation_analyzer.generate_rotation_alert(analysis, min_score=60)
                    if alert:
                        await self.send_alert(alert)
                        self.rotation_analyzer.store_rotation(analysis)
                        logger.info("Capital rotation alert sent")
                    else:
                        logger.info("No strong capital rotation detected")
                else:
                    logger.warning("Capital rotation analysis failed or insufficient data")
            except Exception as e:
                logger.error(f"Capital rotation monitoring error: {e}")
            await asyncio.sleep(1800)  # 30 minutes

    def get_current_market_sentiment(self) -> Optional[str]:
        """Get current market sentiment from USDT dominance"""
        try:
            df = self.dominance_analyzer.get_usdt_dominance_data()
            if df is not None and len(df) >= 20:
                analysis = self.dominance_analyzer.analyze_dominance_sentiment(df)
                return analysis.get('sentiment', 'NEUTRAL')
        except Exception as e:
            logger.error(f"Error getting market sentiment: {e}")
        return None

    async def setup_monitor(self):
        """Monitor for professional trading setups based on order book"""
        while self.setup_running:
            try:
                logger.info("Starting setup scanning cycle...")
                setup_count = 0
                
                for symbol in self.symbols:
                    try:
                        # Get current price and order book
                        ticker = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: self.exchange.fetch_ticker(symbol)
                        )
                        
                        orderbook = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: self.exchange.fetch_order_book(symbol, limit=2000)
                        )
                        
                        current_price = ticker['last']
                        
                        if not orderbook or not orderbook.get('bids') or not orderbook.get('asks'):
                            continue
                            
                        # Detect breakout setups
                        breakout_setups = self.setup_analyzer.detect_breakout_setups(
                            orderbook, current_price, symbol
                        )
                        
                        # Detect bounce setups  
                        bounce_setups = self.setup_analyzer.detect_bounce_setups(
                            orderbook, current_price, symbol
                        )
                        
                        # Process all setups
                        all_setups = breakout_setups + bounce_setups
                        
                        for setup in all_setups:
                            # Generate and send alert
                            alert = self.setup_analyzer.generate_setup_alert(setup)
                            await self.send_alert(alert)
                            
                            # Store setup in database
                            self.setup_analyzer.store_setup(setup)
                            setup_count += 1
                            
                            logger.info(f"Setup detected: {setup['symbol']} {setup['setup_type']} {setup['direction']}")
                            
                    except Exception as e:
                        logger.error(f"Setup analysis error for {symbol}: {e}")
                        continue
                        
                if setup_count > 0:
                    logger.info(f"Setup monitoring completed: {setup_count} setups detected")
                else:
                    logger.info("Setup monitoring completed: No setups detected")
                    
            except Exception as e:
                logger.error(f"Setup monitoring error: {e}")
                
            await asyncio.sleep(3600)  # Check every hour

    async def ema_monitor(self):
        """Monitor EMA touches and crosses"""
        while self.ema_running:
            try:
                logger.info("Starting EMA monitoring cycle...")
                alert_count = 0
                
                for symbol in self.symbols:
                    try:
                        # Get OHLCV data with enough history for EMA 200
                        df = self.get_ohlcv(symbol, timeframe='1h', limit=250)
                        if df is None or len(df) < 200:  # Need minimum data for EMA-200 calculations
                            logger.warning(f"Insufficient data for {symbol}: {len(df) if df is not None else 0} candles (need 200+)")
                            continue
                        
                        # Calculate levels including EMAs
                        df = self.calculate_levels(df)
                        
                        # Detect EMA touches/crosses
                        ema_alerts = self.ema_analyzer.detect_ema_touches(df, symbol)
                        
                        # Debug logging
                        if len(ema_alerts) == 0:
                            current_price = df.iloc[-1]['close']
                            logger.debug(f"EMA check: {symbol} price={current_price:.4f}, no alerts")
                        else:
                            logger.info(f"Found {len(ema_alerts)} EMA alerts for {symbol}")
                        
                        # Send alerts
                        for alert in ema_alerts:
                            try:
                                alert_message = self.ema_analyzer.generate_ema_alert(alert)
                                await self.send_alert(alert_message)
                                alert_count += 1
                                
                                logger.info(f"EMA alert sent: {symbol} {alert['alert_type']} EMA-{alert['ema_period']}")
                            except Exception as alert_error:
                                logger.error(f"Failed to send EMA alert for {symbol}: {alert_error}")
                            
                    except Exception as e:
                        logger.error(f"EMA analysis error for {symbol}: {e}", exc_info=True)
                        continue
                        
                if alert_count > 0:
                    logger.info(f"EMA monitoring completed: {alert_count} alerts sent")
                else:
                    logger.info("EMA monitoring completed: No EMA touches detected")
                    
            except Exception as e:
                logger.error(f"EMA monitoring error: {e}")
                
            await asyncio.sleep(60)  # Check every minute

    async def rsi_monitor(self):
        """Monitor RSI overbought/oversold conditions"""
        while self.rsi_running:
            try:
                logger.info("Starting RSI monitoring cycle...")
                alert_count = 0
                
                for symbol in self.symbols:
                    try:
                        # Get OHLCV data with enough history for RSI calculations
                        df = self.get_ohlcv(symbol, timeframe='1h', limit=50)
                        if df is None or len(df) < 15:  # Need minimum data for RSI-14 calculations
                            logger.warning(f"Insufficient data for {symbol}: {len(df) if df is not None else 0} candles (need 15+)")
                            continue
                        
                        # Calculate levels including RSI
                        df = self.calculate_levels(df)
                        
                        # Detect RSI signals
                        rsi_alerts = self.rsi_analyzer.detect_rsi_signals(df, symbol)
                        
                        # Debug logging
                        if len(rsi_alerts) == 0:
                            current_rsi = df.iloc[-1]['rsi']
                            logger.debug(f"RSI check: {symbol} RSI={current_rsi:.1f}, no alerts")
                        else:
                            logger.info(f"Found {len(rsi_alerts)} RSI alerts for {symbol}")
                        
                        # Send alerts
                        for alert in rsi_alerts:
                            try:
                                alert_message = self.rsi_analyzer.format_rsi_alert(alert)
                                await self.send_alert(alert_message)
                                alert_count += 1
                                
                                logger.info(f"RSI alert sent: {symbol} {alert['alert_type']} RSI={alert['rsi_value']:.1f}")
                            except Exception as alert_error:
                                logger.error(f"Failed to send RSI alert for {symbol}: {alert_error}")
                            
                    except Exception as e:
                        logger.error(f"RSI analysis error for {symbol}: {e}", exc_info=True)
                        continue
                        
                if alert_count > 0:
                    logger.info(f"RSI monitoring completed: {alert_count} alerts sent")
                else:
                    logger.info("RSI monitoring completed: No RSI signals detected")
                    
            except Exception as e:
                logger.error(f"RSI monitoring error: {e}")
                
            await asyncio.sleep(300)  # Check every 5 minutes

    async def start_monitoring(self):
        """Start both breakout and order book monitoring"""
        if self.running:
            logger.info("Monitoring already running")
            return
            
        self.running = True
        self.orderbook_running = True
        self.dominance_running = True
        self.volume_surge_running = True
        self.rotation_running = True
        self.ema_running = True
        self.rsi_running = True
        self.setup_running = True
        
        logger.info("Starting enhanced monitoring...")
        
        # Start breakout monitoring task
        breakout_task = asyncio.create_task(self.breakout_monitor())
        logger.info("Breakout monitoring started")
        
        # Start order book monitoring task
        orderbook_task = asyncio.create_task(self.order_book_monitor())
        logger.info("Order book monitoring started")
        
        # Start USDT dominance monitoring task
        dominance_task = asyncio.create_task(self.dominance_monitor())
        logger.info("USDT dominance monitoring started")
        
        # Start volume surge monitoring task
        volume_task = asyncio.create_task(self.volume_surge_monitor())
        logger.info("Volume surge monitoring started")
        
        # Start capital rotation monitoring task
        rotation_task = asyncio.create_task(self.rotation_monitor())
        logger.info("Capital rotation monitoring started")
        
        # Start setup monitoring task
        setup_task = asyncio.create_task(self.setup_monitor())
        logger.info("Setup monitoring started")
        
        # Start EMA monitoring task
        ema_task = asyncio.create_task(self.ema_monitor())
        logger.info("EMA monitoring started")
        
        # Start RSI monitoring task
        rsi_task = asyncio.create_task(self.rsi_monitor())
        logger.info("RSI monitoring started")
        
        # Send startup notification
        await self.send_alert(
            "🤖 <b>Enhanced Crypto Bot Started</b>\n\n"
            "🚀 Breakout monitoring: ACTIVE\n"
            "📊 Order book analysis: ACTIVE\n"
            "🎯 USDT.D sentiment: ACTIVE\n"
            "📈 Volume surge alerts: ACTIVE\n"
            "⚡ Setup detection: ACTIVE\n"
            "🔄 Capital rotation: ACTIVE\n"
            "🎯 EMA alerts: ACTIVE\n"
            "📊 RSI alerts: ACTIVE\n\n"
            f"Monitoring {len(self.symbols)} symbols"
        )
        
        return (
            breakout_task,
            orderbook_task,
            dominance_task,
            volume_task,
            rotation_task,
            setup_task,
            ema_task,
            rsi_task,
        )

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
        self.dominance_running = False
        self.volume_surge_running = False
        self.rotation_running = False
        self.ema_running = False
        self.rsi_running = False


# Bot command handlers

async def start_command(update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    await update.message.reply_text(
        "🤖 <b>Enhanced Crypto Bot</b>\n\n"
        "Features:\n"
        "🚀 Breakout alerts for 19 symbols\n"
        "📊 Order book analysis every 30min\n"
        "🎯 USDT.D macro sentiment monitoring\n"
        "📈 Volume surge early warning alerts\n"
        "🔄 Capital rotation detection\n"
        "\nCommands:\n"
        "/start - Show this message\n"
        "/status - Check bot status\n"
        "/symbols - Show monitored symbols\n"
        "/orderbook - Get instant BTC/USDT order book analysis\n"
        "/analyze SYMBOL - Analyze order book for specific symbol\n"
        "/analyze all - Analyze all 19 watchlist symbols\n"
        "/entry SYMBOL - Get optimal entry prices for symbol\n"
        "/entry all - Show best entry opportunities across watchlist\n"
        "/dominance - Check USDT dominance sentiment\n"
        "/volume - Scan for current volume surges\n"
        "/rotation - Analyze capital rotation across watchlist\n"
        "/setups - Scan for trading setups\n"
        "/stop - Stop monitoring",
        parse_mode='HTML'
    )

async def status_command(update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command"""
    bot_instance = context.bot_data.get('bot_instance')
    if bot_instance:
        breakout_status = "🟢 Running" if bot_instance.running else "🔴 Stopped"
        orderbook_status = "🟢 Running" if bot_instance.orderbook_running else "🔴 Stopped"
        dominance_status = "🟢 Running" if bot_instance.dominance_running else "🔴 Stopped"
        volume_status = "🟢 Running" if bot_instance.volume_surge_running else "🔴 Stopped"
        rotation_status = "🟢 Running" if bot_instance.rotation_running else "🔴 Stopped"
        ema_status = "🟢 Running" if bot_instance.ema_running else "🔴 Stopped"
        rsi_status = "🟢 Running" if bot_instance.rsi_running else "🔴 Stopped"
        await update.message.reply_text(
            f"<b>Bot Status:</b>\n"
            f"🚀 Breakouts: {breakout_status}\n"
            f"📊 Order Book: {orderbook_status}\n"
            f"🎯 USDT.D Sentiment: {dominance_status}\n"
            f"📈 Volume Surges: {volume_status}\n"
            f"🔄 Capital Rotation: {rotation_status}\n"
            f"🎯 EMA Alerts: {ema_status}\n"
            f"📊 RSI Alerts: {rsi_status}",
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
        f"<b>Additional Monitoring:</b>\n"
        f"• Order Book: BTC/USDT (every 30min)\n"
        f"• USDT.D Sentiment: Global (every 1h)\n"
        f"• Volume Surges: All symbols (every 30min)\n"
        f"• Capital Rotation: All symbols (every 30min)",
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

async def dominance_command(update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /dominance command - check USDT dominance sentiment"""
    bot_instance = context.bot_data.get('bot_instance')
    if not bot_instance:
        await update.message.reply_text("❌ Bot not initialized", parse_mode='HTML')
        return
        
    await update.message.reply_text("🎯 Analyzing USDT dominance sentiment...", parse_mode='HTML')
    
    try:
        df = bot_instance.dominance_analyzer.get_usdt_dominance_data()
        if df is not None and len(df) >= 20:
            analysis = bot_instance.dominance_analyzer.analyze_dominance_sentiment(df)
            if analysis:
                alert = bot_instance.dominance_analyzer.generate_dominance_alert(analysis)
                await update.message.reply_text(alert, parse_mode='HTML')
            else:
                await update.message.reply_text("❌ Failed to analyze dominance data", parse_mode='HTML')
        else:
            await update.message.reply_text("❌ Insufficient dominance data available", parse_mode='HTML')
    except Exception as e:
        await update.message.reply_text(f"❌ Error analyzing dominance: {str(e)[:100]}", parse_mode='HTML')

async def volume_command(update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /volume command - check volume surges across watchlist"""
    bot_instance = context.bot_data.get('bot_instance')
    if not bot_instance:
        await update.message.reply_text("❌ Bot not initialized", parse_mode='HTML')
        return
        
    await update.message.reply_text("📈 Scanning for volume surges...", parse_mode='HTML')
    
    try:
        surges_found = []
        
        for symbol in bot_instance.symbols:
            try:
                df = bot_instance.get_ohlcv(symbol, timeframe='1h', limit=50)
                if df is None or len(df) < 21:
                    continue
                    
                surge_analysis = bot_instance.volume_analyzer.analyze_volume_surge(symbol, df)
                
                if surge_analysis:
                    surges_found.append({
                        'symbol': symbol,
                        'volume_ratio': surge_analysis['volume_ratio'],
                        'surge_type': surge_analysis['surge_type'],
                        'price_change': surge_analysis['price_change_pct'],
                        'alignment': surge_analysis['volume_alignment'],
                        'risk': surge_analysis['risk_assessment']
                    })
                    
            except Exception as e:
                logger.error(f"Error checking volume for {symbol}: {e}")
                
        if surges_found:
            # Sort by volume ratio (highest first)
            surges_found.sort(key=lambda x: x['volume_ratio'], reverse=True)
            
            message = "📈 <b>Current Volume Surges</b>\n\n"
            for surge in surges_found[:10]:  # Show top 10
                emoji = "🚨" if surge['surge_type'] == "MEGA" else "⚠️" if surge['surge_type'] == "EXTREME" else "🟡"
                message += f"{emoji} <b>{surge['symbol']}</b>: {surge['volume_ratio']:.1f}x ({surge['surge_type']})\n"
                message += f"   Price: {surge['price_change']:+.2f}% | {surge['alignment'].replace('_', ' ')}\n\n"
                
            if len(surges_found) > 10:
                message += f"<i>...and {len(surges_found) - 10} more</i>"
                
        else:
            message = "📊 <b>Volume Scan Complete</b>\n\n✅ No significant volume surges detected across watchlist"
            
        await update.message.reply_text(message, parse_mode='HTML')
        
    except Exception as e:
        await update.message.reply_text(f"❌ Error scanning volumes: {str(e)[:100]}", parse_mode='HTML')

async def setups_command(update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /setups command - check for current trading setups"""
    bot_instance = context.bot_data.get('bot_instance')
    if not bot_instance:
        await update.message.reply_text("❌ Bot not initialized", parse_mode='HTML')
        return
        
    await update.message.reply_text("⚡ Scanning for trading setups...", parse_mode='HTML')
    
    try:
        setups_found = []
        
        for symbol in bot_instance.symbols:
            try:
                # Get current price and order book
                ticker = bot_instance.exchange.fetch_ticker(symbol)
                orderbook = bot_instance.exchange.fetch_order_book(symbol, limit=2000)
                current_price = ticker['last']
                
                if not orderbook or not orderbook.get('bids') or not orderbook.get('asks'):
                    continue
                    
                # Detect breakout setups
                breakout_setups = bot_instance.setup_analyzer.detect_breakout_setups(
                    orderbook, current_price, symbol
                )
                
                # Detect bounce setups  
                bounce_setups = bot_instance.setup_analyzer.detect_bounce_setups(
                    orderbook, current_price, symbol
                )
                
                # Combine all setups
                all_setups = breakout_setups + bounce_setups
                setups_found.extend(all_setups)
                
            except Exception as e:
                logger.error(f"Error checking setups for {symbol}: {e}")
                
        if setups_found:
            # Sort by confidence and risk/reward ratio
            setups_found.sort(key=lambda x: (x['confidence'] == 'HIGH', x['risk_reward_ratio']), reverse=True)
            
            if len(setups_found) == 1:
                # Single setup - send detailed alert
                setup = setups_found[0]
                alert = bot_instance.setup_analyzer.generate_setup_alert(setup)
                await update.message.reply_text(alert, parse_mode='HTML')
            else:
                # Multiple setups - send summary
                message = "⚡ <b>Trading Setups Found</b>\n\n"
                
                for i, setup in enumerate(setups_found[:8]):  # Show top 8
                    setup_emoji = "🚀" if setup['setup_type'] == 'BREAKOUT' else "🎯"
                    direction_emoji = "🟢" if setup['direction'] == 'LONG' else "🔴"
                    confidence_emoji = "🔥" if setup['confidence'] == 'HIGH' else "⭐"
                    
                    # Format price with appropriate decimals
                    price = setup['entry_price']
                    decimals = 6 if price < 1 else 4 if price < 100 else 2
                    
                    message += f"{setup_emoji}{direction_emoji} <b>{setup['symbol']}</b> {setup['setup_type']}\n"
                    message += f"   Entry: ${price:.{decimals}f} | RR: {setup['risk_reward_ratio']:.1f} {confidence_emoji}\n\n"
                    
                if len(setups_found) > 8:
                    message += f"<i>...and {len(setups_found) - 8} more setups</i>\n\n"
                    
                message += "💡 Use /setups to scan again or check individual symbols"
                await update.message.reply_text(message, parse_mode='HTML')
                
        else:
            message = "⚡ <b>Setup Scan Complete</b>\n\n📊 No high-quality trading setups found at current market conditions."
            await update.message.reply_text(message, parse_mode='HTML')
            
    except Exception as e:
        await update.message.reply_text(f"❌ Error scanning setups: {str(e)[:100]}", parse_mode='HTML')

async def rotation_command(update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /rotation command - analyze capital rotation across watchlist"""
    bot_instance = context.bot_data.get('bot_instance')
    if not bot_instance:
        await update.message.reply_text("❌ Bot not initialized", parse_mode='HTML')
        return
    
    # Optional timeframe arg (default 1h)
    tf = '1h'
    if context.args:
        candidate = context.args[0].lower()
        if candidate in ['15m', '30m', '1h', '4h']:
            tf = candidate
    
    await update.message.reply_text(f"🔄 Scanning for capital rotation ({tf})...", parse_mode='HTML')
    
    try:
        analysis = bot_instance.rotation_analyzer.analyze_rotation(bot_instance.symbols, bot_instance.get_ohlcv, timeframe=tf)
        if not analysis:
            await update.message.reply_text("❌ Unable to compute rotation (insufficient data)", parse_mode='HTML')
            return
        
        # Build alert; for manual command show even lower-score pairs
        alert = bot_instance.rotation_analyzer.generate_rotation_alert(analysis, min_score=0)
        if not alert:
            # Fallback summary
            inflows = analysis.get('inflows', [])
            outflows = analysis.get('outflows', [])
            lines = [f"🔄 <b>Capital Rotation Summary</b> ({tf})\n"]
            if inflows:
                top_in = ", ".join([f"{x['symbol']}({x['quote_vol_ratio']:.1f}x)" for x in inflows[:3]])
                lines.append(f"🏁 Inflows: {top_in}")
            if outflows:
                top_out = ", ".join([f"{x['symbol']}({x['quote_vol_ratio']:.1f}x)" for x in outflows[:3]])
                lines.append(f"💨 Outflows: {top_out}")
            alert = "\n".join(lines)
        
        await update.message.reply_text(alert, parse_mode='HTML')
    except Exception as e:
        await update.message.reply_text(f"❌ Error analyzing rotation: {str(e)[:100]}", parse_mode='HTML')

async def entry_command(update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /entry command - show deepest walls for entry analysis"""
    bot_instance = context.bot_data.get('bot_instance')
    if not bot_instance:
        await update.message.reply_text("❌ Bot not initialized", parse_mode='HTML')
        return

    args = context.args
    if not args:
        await update.message.reply_text(
            "Usage: /entry SYMBOL (e.g. BTC/USDT) or /entry all", parse_mode='HTML'
        )
        return

    # Handle /entry all command  
    if args[0].lower() == 'all':
        await update.message.reply_text("🔍 Analyzing deepest walls for all watchlist symbols...", parse_mode='HTML')
        
        messages = []
        for symbol in bot_instance.symbols:
            data = bot_instance.orderbook_analyzer.get_deepest_walls(symbol)
            if not data:
                continue
            price = data['current_price']
            text = [f"🔥 {symbol} (${price:.4f}):"]
            # Format top bids
            text.append("🟢 Top Bid Walls:")
            for wall in data['bids']:
                dist = (price - wall['price']) / price * 100
                usdt_value = wall['quantity'] * wall['price']
                text.append(f"  • ${wall['price']:.4f} – qty {wall['quantity']:.0f} – {dist:.2f}% below – ${usdt_value:,.0f} USDT")
            # Format top asks
            text.append("🔴 Top Ask Walls:")
            for wall in data['asks']:
                dist = (wall['price'] - price) / price * 100
                usdt_value = wall['quantity'] * wall['price']
                text.append(f"  • ${wall['price']:.4f} – qty {wall['quantity']:.0f} – {dist:.2f}% above – ${usdt_value:,.0f} USDT")
            messages.append("\n".join(text))
        
        if not messages:
            await update.message.reply_text("No deep walls found.", parse_mode='HTML')
        else:
            await update.message.reply_text("\n\n".join(messages), parse_mode='HTML')
        return

    # Single symbol analysis
    symbol = args[0].upper()
    if '/' not in symbol:
        symbol += '/USDT'
    
    await update.message.reply_text(f"🔍 Analyzing deepest walls for {symbol}...", parse_mode='HTML')
    
    try:
        data = bot_instance.orderbook_analyzer.get_deepest_walls(symbol)
        
        if not data:
            await update.message.reply_text(f"❌ Unable to fetch order book data for {symbol}", parse_mode='HTML')
            return
        
        price = data['current_price']
        message = f"🔥 <b>{symbol}</b> (${price:.4f})\n\n"
        
        # Format top bids
        message += "🟢 <b>Top Bid Walls:</b>\n"
        for wall in data['bids']:
            dist = (price - wall['price']) / price * 100
            usdt_value = wall['quantity'] * wall['price']
            message += f"  • ${wall['price']:.4f} – qty {wall['quantity']:.0f} – {dist:.2f}% below – ${usdt_value:,.0f} USDT\n"
        
        message += "\n"
        
        # Format top asks  
        message += "🔴 <b>Top Ask Walls:</b>\n"
        for wall in data['asks']:
            dist = (wall['price'] - price) / price * 100
            usdt_value = wall['quantity'] * wall['price']
            message += f"  • ${wall['price']:.4f} – qty {wall['quantity']:.0f} – {dist:.2f}% above – ${usdt_value:,.0f} USDT\n"
        
        await update.message.reply_text(message, parse_mode='HTML')
        
    except Exception as e:
        await update.message.reply_text(f"❌ Error analyzing deepest walls for {symbol}: {str(e)[:100]}", parse_mode='HTML')

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
    application.add_handler(CommandHandler("dominance", dominance_command))
    application.add_handler(CommandHandler("volume", volume_command))
    application.add_handler(CommandHandler("rotation", rotation_command))
    application.add_handler(CommandHandler("setups", setups_command))
    application.add_handler(CommandHandler("entry", entry_command))
    application.add_handler(CommandHandler("stop", stop_command))

    application.bot_data['bot_instance'] = bot

    logger.info("Starting enhanced Telegram bot...")
    application.run_polling()

if __name__ == "__main__":
    main()
