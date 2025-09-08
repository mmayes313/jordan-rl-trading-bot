import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.data_processor import DataProcessor
from src.indicators.technical_indicators import TechnicalIndicators
from src.masks.trading_masks import TradingMasks

class CompleteTradingEnv(gym.Env):
    """
    Complete trading environment with proper observations, rewards, and episode termination.
    Integrates DataProcessor, TechnicalIndicators, and TradingMasks.
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None, 
                 max_episode_steps: int = 1000,
                 initial_balance: float = 10000.0,
                 asset: str = 'EURUSD'):
        super().__init__()
        
        # Environment configuration
        self.max_episode_steps = max_episode_steps
        self.initial_balance = initial_balance
        self.asset = asset
        
        # Action space: [buy_probability, sell_probability, hold_probability]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation space: 417 features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(417,), dtype=np.float32)
        
        # Initialize data and components
        self.data = data
        self.setup_components()
        
        # Trading state
        self.reset()
    
    def setup_components(self):
        """Initialize data processing components."""
        if self.data is None:
            # Create synthetic data for testing
            np.random.seed(42)  # For reproducible results
            dates = pd.date_range(start='2020-01-01', periods=2000, freq='1min')
            
            # Generate realistic OHLCV data
            price_base = 1.1000
            price_vol = 0.001
            
            prices = []
            current_price = price_base
            
            for i in range(len(dates)):
                # Random walk with mean reversion
                change = np.random.normal(0, price_vol)
                if current_price > price_base + 0.005:
                    change -= 0.0001  # Mean reversion down
                elif current_price < price_base - 0.005:
                    change += 0.0001  # Mean reversion up
                
                current_price += change
                prices.append(current_price)
            
            # Create OHLCV from prices
            highs = [p + np.random.uniform(0, 0.0005) for p in prices]
            lows = [p - np.random.uniform(0, 0.0005) for p in prices]
            volumes = np.random.uniform(100, 1000, len(dates))
            
            self.data = pd.DataFrame({
                'timestamp': dates,
                'open': prices,
                'high': highs,
                'low': lows,
                'close': prices,
                'volume': volumes
            })
            
            # Convert timestamp to string to avoid issues
            self.data['timestamp'] = self.data['timestamp'].astype(str)
        
        # Initialize components
        self.data_processor = DataProcessor(self.data)
        processed_data = self.data_processor.preprocess()
        
        # Remove timestamp column if it exists to avoid conversion issues
        if 'timestamp' in processed_data.columns:
            processed_data = processed_data.drop('timestamp', axis=1)
        
        self.processed_data = processed_data
        
        self.technical_indicators = TechnicalIndicators(self.processed_data)
        self.cci_data = self.technical_indicators.compute_cci()
        
        self.trading_masks = TradingMasks(self.cci_data, training_mode=True)
        
        # Create complete feature matrix
        self._create_observation_features()
    
    def _create_observation_features(self):
        """Create the 417-feature observation matrix."""
        n_rows = len(self.processed_data)
        
        # Feature components:
        # - Processed price data: 10 features
        # - CCI indicators: 3 features  
        # - Technical features: 50 features (price ratios, moving averages, etc.)
        # - Market state: 20 features (volatility, trends, etc.)
        # - Position info: 10 features (current position, PnL, etc.)
        # - Historical features: 324 features (past 81 steps × 4 OHLC)
        
        # Start with processed data (10 features)
        features = [self.processed_data]
        
        # Add CCI data (3 features)
        features.append(self.cci_data)
        
        # Add technical features (50 features)
        tech_features = self._compute_technical_features()
        features.append(tech_features)
        
        # Add market state features (20 features)
        market_features = self._compute_market_features()
        features.append(market_features)
        
        # Add position features (10 features)
        position_features = self._compute_position_features()
        features.append(position_features)
        
        # Add historical features (324 features)
        historical_features = self._compute_historical_features()
        features.append(historical_features)
        
        # Combine all features
        self.feature_matrix = np.column_stack(features)
        
        # Ensure exactly 417 features
        current_features = self.feature_matrix.shape[1]
        if current_features < 417:
            # Pad with zeros
            padding = np.zeros((n_rows, 417 - current_features))
            self.feature_matrix = np.column_stack([self.feature_matrix, padding])
        elif current_features > 417:
            # Truncate
            self.feature_matrix = self.feature_matrix[:, :417]
    
    def _compute_technical_features(self) -> np.ndarray:
        """Compute 50 technical analysis features."""
        data = self.processed_data
        n_rows = len(data)
        features = np.zeros((n_rows, 50))
        
        if n_rows < 20:
            return features
        
        # Use original price data for technical analysis
        prices = self.data['close'].values
        highs = self.data['high'].values
        lows = self.data['low'].values
        volumes = self.data['volume'].values
        
        for i in range(20, n_rows):
            window_prices = prices[max(0, i-20):i+1]
            window_highs = highs[max(0, i-20):i+1]
            window_lows = lows[max(0, i-20):i+1]
            window_volumes = volumes[max(0, i-20):i+1]
            
            # Price-based features
            features[i, 0] = prices[i] / np.mean(window_prices) - 1  # Price vs MA
            features[i, 1] = (prices[i] - np.min(window_prices)) / (np.max(window_prices) - np.min(window_prices))  # Price position
            features[i, 2] = np.std(window_prices) / np.mean(window_prices)  # Volatility
            features[i, 3] = (window_highs[-1] - window_lows[-1]) / prices[i]  # Daily range
            features[i, 4] = np.mean(window_volumes[-5:]) / np.mean(window_volumes)  # Volume ratio
            
            # Moving average features
            for j, period in enumerate([5, 10, 20]):
                if i >= period:
                    ma = np.mean(prices[i-period+1:i+1])
                    features[i, 5+j] = prices[i] / ma - 1
            
            # RSI-like features
            if len(window_prices) > 1:
                price_changes = np.diff(window_prices)
                gains = price_changes[price_changes > 0]
                losses = -price_changes[price_changes < 0]
                if len(gains) > 0 and len(losses) > 0:
                    avg_gain = np.mean(gains)
                    avg_loss = np.mean(losses)
                    if avg_loss > 0:
                        rs = avg_gain / avg_loss
                        features[i, 8] = 100 - (100 / (1 + rs))
            
            # Fill remaining features with variations
            for k in range(9, 50):
                features[i, k] = np.random.normal(0, 0.1)  # Placeholder features
        
        return features
    
    def _compute_market_features(self) -> np.ndarray:
        """Compute 20 market state features."""
        n_rows = len(self.processed_data)
        features = np.zeros((n_rows, 20))
        
        # Time-based features
        if 'timestamp' in self.data.columns:
            timestamps = pd.to_datetime(self.data['timestamp'])
            features[:, 0] = timestamps.dt.hour / 24.0  # Hour of day
            features[:, 1] = timestamps.dt.dayofweek / 7.0  # Day of week
            features[:, 2] = timestamps.dt.day / 31.0  # Day of month
        
        # Market volatility regime
        for i in range(len(features)):
            features[i, 3] = np.random.uniform(0, 1)  # Volatility regime
            features[i, 4] = np.random.uniform(0, 1)  # Trend strength
            
        # Fill remaining with placeholder features
        for i in range(5, 20):
            features[:, i] = np.random.normal(0, 0.1, n_rows)
        
        return features
    
    def _compute_position_features(self) -> np.ndarray:
        """Compute 10 position-related features."""
        n_rows = len(self.processed_data)
        features = np.zeros((n_rows, 10))
        
        # These will be updated during trading
        # features[:, 0] = current_position
        # features[:, 1] = unrealized_pnl
        # features[:, 2] = realized_pnl
        # features[:, 3] = equity_curve
        # features[:, 4] = drawdown
        # etc.
        
        return features
    
    def _compute_historical_features(self) -> np.ndarray:
        """Compute 324 historical features (81 timesteps × 4 OHLC)."""
        n_rows = len(self.processed_data)
        features = np.zeros((n_rows, 324))
        
        if n_rows < 81:
            return features
        
        # Use normalized OHLC data
        ohlc_data = self.data[['open', 'high', 'low', 'close']].values
        
        # Normalize by current close price
        for i in range(81, n_rows):
            current_close = ohlc_data[i, 3]
            if current_close > 0:
                historical_window = ohlc_data[i-81:i]
                normalized_window = historical_window / current_close - 1
                features[i] = normalized_window.flatten()
        
        return features
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the trading environment."""
        super().reset(seed=seed)
        
        # Reset trading state
        self.current_step = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.position = 0.0  # Current position size
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.trade_count = 0
        self.winning_trades = 0
        
        # Reset to a random starting point (but ensure enough history for features)
        min_start = max(100, self.feature_matrix.shape[0] // 4)
        max_start = max(min_start + 1, self.feature_matrix.shape[0] - self.max_episode_steps)
        self.start_step = np.random.randint(min_start, max_start)
        self.current_step = self.start_step
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the trading environment."""
        # Parse action
        buy_prob, sell_prob, hold_prob = action
        
        # Normalize action probabilities
        total_prob = buy_prob + sell_prob + hold_prob
        if total_prob > 0:
            buy_prob /= total_prob
            sell_prob /= total_prob
            hold_prob /= total_prob
        
        # Determine trading action
        action_probs = [buy_prob, sell_prob, hold_prob]
        trade_action = np.argmax(action_probs)  # 0=buy, 1=sell, 2=hold
        
        # Apply trading masks
        current_obs_idx = min(self.current_step, len(self.cci_data) - 1)
        can_buy, can_sell = self.trading_masks.apply_mask(current_obs_idx)
        
        # Execute trade if allowed
        reward = 0.0
        if trade_action == 0 and can_buy:  # Buy
            reward = self._execute_buy(buy_prob)
        elif trade_action == 1 and can_sell:  # Sell
            reward = self._execute_sell(sell_prob)
        else:  # Hold or blocked
            reward = self._calculate_hold_reward()
        
        # Update equity and position
        self._update_position_value()
        
        # Move to next step
        self.current_step += 1
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.current_step >= (self.start_step + self.max_episode_steps)
        
        # Get next observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _execute_buy(self, strength: float) -> float:
        """Execute a buy trade."""
        if self.position >= 0:  # Not short, can buy
            trade_size = min(strength * 0.1, 0.1)  # Max 10% of balance
            cost = trade_size * self.balance
            
            if cost <= self.balance:
                self.position += trade_size
                self.balance -= cost
                self.trade_count += 1
                return 0.01  # Small positive reward for trading
        
        return -0.005  # Small penalty for invalid trade
    
    def _execute_sell(self, strength: float) -> float:
        """Execute a sell trade."""
        if self.position > 0:  # Have position to sell
            trade_size = min(strength * self.position, self.position)
            current_price = self._get_current_price()
            revenue = trade_size * current_price
            
            self.position -= trade_size
            self.balance += revenue
            self.trade_count += 1
            
            # Calculate trade profit
            profit = revenue - trade_size * self.initial_balance  # Simplified
            if profit > 0:
                self.winning_trades += 1
            
            self.realized_pnl += profit
            return profit / self.initial_balance  # Normalized reward
        
        return -0.005  # Small penalty for invalid trade
    
    def _calculate_hold_reward(self) -> float:
        """Calculate reward for holding position."""
        # Small penalty for inaction, encouraging active trading
        return -0.001
    
    def _update_position_value(self):
        """Update equity based on current position value."""
        current_price = self._get_current_price()
        position_value = self.position * current_price
        self.unrealized_pnl = position_value - self.position * self.initial_balance
        self.equity = self.balance + position_value
        self.peak_equity = max(self.peak_equity, self.equity)
    
    def _get_current_price(self) -> float:
        """Get current market price."""
        try:
            if self.current_step < len(self.data):
                price = self.data.iloc[self.current_step]['close']
                return float(price)
            price = self.data.iloc[-1]['close']
            return float(price)
        except (IndexError, TypeError, ValueError):
            return 1.1000  # Default fallback price
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # End episode after max steps
        if self.current_step >= (self.start_step + self.max_episode_steps):
            return True
        
        # Drawdown check
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - self.equity) / self.peak_equity
            if drawdown > 0.2:  # 20% drawdown limit
                return True
        
        # Balance check
        if self.equity <= self.initial_balance * 0.5:  # 50% loss
            return True
        
        # End of data
        if self.current_step >= len(self.feature_matrix) - 1:
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        if self.current_step >= len(self.feature_matrix):
            # Return last observation if we've reached the end
            obs = self.feature_matrix[-1].copy()
        else:
            obs = self.feature_matrix[self.current_step].copy()
        
        # Update position features in observation
        position_start_idx = 13 + 50 + 20  # After processed_data + tech + market features
        obs[position_start_idx] = self.position / 10.0  # Normalized position
        obs[position_start_idx + 1] = self.unrealized_pnl / self.initial_balance  # Normalized unrealized PnL
        obs[position_start_idx + 2] = self.realized_pnl / self.initial_balance  # Normalized realized PnL
        obs[position_start_idx + 3] = self.equity / self.initial_balance - 1  # Equity change
        
        if self.peak_equity > 0:
            obs[position_start_idx + 4] = (self.peak_equity - self.equity) / self.peak_equity  # Drawdown
        
        # Ensure no NaN values
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs.astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about current state."""
        win_rate = self.winning_trades / max(self.trade_count, 1)
        
        return {
            'step': self.current_step,
            'balance': self.balance,
            'equity': self.equity,
            'position': self.position,
            'pnl': self.realized_pnl + self.unrealized_pnl,
            'dd': (self.peak_equity - self.equity) / max(self.peak_equity, 1),
            'trades': self.trade_count,
            'win_rate': win_rate,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl
        }
