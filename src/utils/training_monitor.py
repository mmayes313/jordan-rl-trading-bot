"""
Training Monitor - Real-time training metrics display
Shows PnL, rewards, drawdown, daily goals, win/loss rate, and trade count during training
"""

import time
import json
import numpy as np
from datetime import datetime, timedelta
from collections import deque
from loguru import logger
from pathlib import Path

class TrainingMonitor:
    """
    Real-time training metrics monitor for PPO trading bot.
    Tracks and displays key performance indicators during training.
    """
    
    def __init__(self, update_frequency=100):
        """
        Initialize training monitor.
        
        Args:
            update_frequency: How often to update display (in timesteps)
        """
        self.update_frequency = update_frequency
        self.start_time = time.time()
        self.last_update = 0
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_pnls = deque(maxlen=100)
        self.episode_drawdowns = deque(maxlen=100)
        self.episode_trades = deque(maxlen=100)
        self.episode_wins = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        # Daily goal tracking
        self.daily_target_pct = 0.01  # 1% daily target
        self.max_dd_pct = 0.05  # 5% max drawdown
        self.daily_goals_met = 0
        self.risk_breaches = 0
        
        # Current episode tracking
        self.current_episode = 0
        self.current_pnl = 0.0
        self.current_drawdown = 0.0
        self.current_trades = 0
        self.current_wins = 0
        self.peak_equity = 1000.0  # Starting balance
        
        # Training metrics
        self.total_timesteps = 0
        self.last_policy_loss = 0.0
        self.last_value_loss = 0.0
        self.last_entropy_loss = 0.0
        self.last_kl_div = 0.0
        
        logger.info("Training Monitor initialized")
    
    def update_episode_metrics(self, episode_reward, episode_pnl, episode_trades, episode_wins, episode_length):
        """Update metrics at end of episode."""
        self.current_episode += 1
        
        # Calculate drawdown
        current_equity = self.peak_equity + episode_pnl
        self.peak_equity = max(self.peak_equity, current_equity)
        drawdown = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0
        
        # Store episode metrics
        self.episode_rewards.append(episode_reward)
        self.episode_pnls.append(episode_pnl)
        self.episode_drawdowns.append(drawdown)
        self.episode_trades.append(episode_trades)
        self.episode_wins.append(episode_wins)
        self.episode_lengths.append(episode_length)
        
        # Check daily goals
        daily_return = episode_pnl / self.peak_equity if self.peak_equity > 0 else 0
        if daily_return >= self.daily_target_pct:
            self.daily_goals_met += 1
        
        if drawdown >= self.max_dd_pct:
            self.risk_breaches += 1
        
        # Update current tracking
        self.current_pnl = episode_pnl
        self.current_drawdown = drawdown
        self.current_trades = episode_trades
        self.current_wins = episode_wins
    
    def update_training_metrics(self, timesteps, policy_loss=None, value_loss=None, entropy_loss=None, kl_div=None):
        """Update training-specific metrics."""
        self.total_timesteps = timesteps
        
        if policy_loss is not None:
            self.last_policy_loss = policy_loss
        if value_loss is not None:
            self.last_value_loss = value_loss
        if entropy_loss is not None:
            self.last_entropy_loss = entropy_loss
        if kl_div is not None:
            self.last_kl_div = kl_div
    
    def should_update_display(self, timesteps):
        """Check if it's time to update the display."""
        return timesteps - self.last_update >= self.update_frequency
    
    def print_training_status(self, timesteps=None):
        """Print comprehensive training status."""
        if timesteps:
            self.total_timesteps = timesteps
            self.last_update = timesteps
        
        # Calculate runtime
        runtime = time.time() - self.start_time
        runtime_str = str(timedelta(seconds=int(runtime)))
        
        # Calculate recent averages
        recent_reward = np.mean(list(self.episode_rewards)[-10:]) if self.episode_rewards else 0
        recent_pnl = np.mean(list(self.episode_pnls)[-10:]) if self.episode_pnls else 0
        recent_dd = np.mean(list(self.episode_drawdowns)[-10:]) if self.episode_drawdowns else 0
        recent_trades = np.mean(list(self.episode_trades)[-10:]) if self.episode_trades else 0
        
        # Calculate win rate
        total_trades = sum(self.episode_trades) if self.episode_trades else 0
        total_wins = sum(self.episode_wins) if self.episode_wins else 0
        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate goal success rate
        goal_success_rate = (self.daily_goals_met / max(1, self.current_episode) * 100)
        risk_breach_rate = (self.risk_breaches / max(1, self.current_episode) * 100)
        
        # Print status
        print("\n" + "="*80)
        print(f"🤖 JORDAN RL TRADING BOT - TRAINING STATUS")
        print("="*80)
        
        # Training progress
        print(f"📊 TRAINING PROGRESS:")
        print(f"   Timesteps: {self.total_timesteps:,} | Episodes: {self.current_episode:,} | Runtime: {runtime_str}")
        print(f"   Policy Loss: {self.last_policy_loss:.6f} | Value Loss: {self.last_value_loss:.6f}")
        print(f"   Entropy: {self.last_entropy_loss:.6f} | KL Div: {self.last_kl_div:.6f}")
        
        # Performance metrics
        print(f"\n💰 PERFORMANCE METRICS (Last 10 Episodes):")
        print(f"   Avg Reward: {recent_reward:+.4f} | Avg PnL: ${recent_pnl:+.2f}")
        print(f"   Avg Drawdown: {recent_dd:.2%} | Avg Trades/Episode: {recent_trades:.1f}")
        print(f"   Win Rate: {win_rate:.1f}% ({total_wins}/{total_trades} trades)")
        
        # Daily goals and risk
        print(f"\n🎯 DAILY GOALS & RISK MANAGEMENT:")
        print(f"   Daily Target: {self.daily_target_pct:.1%} | Max Drawdown: {self.max_dd_pct:.1%}")
        print(f"   Goals Met: {self.daily_goals_met}/{self.current_episode} ({goal_success_rate:.1f}%)")
        print(f"   Risk Breaches: {self.risk_breaches}/{self.current_episode} ({risk_breach_rate:.1f}%)")
        
        # Current episode status
        status_emoji = "🟢" if self.current_pnl >= 0 else "🔴"
        dd_emoji = "🟢" if self.current_drawdown < 0.02 else "🟡" if self.current_drawdown < 0.05 else "🔴"
        
        print(f"\n📈 CURRENT EPISODE STATUS:")
        print(f"   {status_emoji} PnL: ${self.current_pnl:+.2f} | {dd_emoji} Drawdown: {self.current_drawdown:.2%}")
        print(f"   Trades: {self.current_trades} | Wins: {self.current_wins}")
        print(f"   Peak Equity: ${self.peak_equity:.2f}")
        
        # Health indicators
        print(f"\n🏥 TRAINING HEALTH:")
        learning_status = "🟢 Learning" if abs(self.last_policy_loss) > 1e-6 else "🔴 Stagnant"
        exploration_status = "🟢 Exploring" if self.last_entropy_loss > 0.1 else "🟡 Low Exploration"
        
        print(f"   {learning_status} | {exploration_status}")
        print(f"   Trades per Episode: {'🟢 Active' if recent_trades > 1 else '🔴 Inactive'}")
        
        print("="*80 + "\n")
        
        # Log to file as well
        logger.info(f"Training Status - Episode {self.current_episode}, Timesteps {self.total_timesteps}, "
                   f"Avg Reward: {recent_reward:.4f}, Avg PnL: ${recent_pnl:.2f}, Win Rate: {win_rate:.1f}%")
    
    def save_metrics(self, filepath="data/logs/training_metrics.json"):
        """Save current metrics to file."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "total_timesteps": self.total_timesteps,
            "current_episode": self.current_episode,
            "runtime_seconds": time.time() - self.start_time,
            "episode_rewards": list(self.episode_rewards),
            "episode_pnls": list(self.episode_pnls),
            "episode_drawdowns": list(self.episode_drawdowns),
            "episode_trades": list(self.episode_trades),
            "episode_wins": list(self.episode_wins),
            "daily_goals_met": self.daily_goals_met,
            "risk_breaches": self.risk_breaches,
            "current_pnl": self.current_pnl,
            "current_drawdown": self.current_drawdown,
            "peak_equity": self.peak_equity,
            "last_policy_loss": self.last_policy_loss,
            "last_value_loss": self.last_value_loss,
            "last_entropy_loss": self.last_entropy_loss,
            "last_kl_div": self.last_kl_div
        }
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Training metrics saved to {filepath}")
    
    def get_summary_stats(self):
        """Get summary statistics for display."""
        if not self.episode_rewards:
            return {}
        
        return {
            "total_episodes": self.current_episode,
            "total_timesteps": self.total_timesteps,
            "avg_reward": np.mean(self.episode_rewards),
            "avg_pnl": np.mean(self.episode_pnls),
            "avg_drawdown": np.mean(self.episode_drawdowns),
            "win_rate": (sum(self.episode_wins) / max(1, sum(self.episode_trades))) * 100,
            "goal_success_rate": (self.daily_goals_met / max(1, self.current_episode)) * 100,
            "risk_breach_rate": (self.risk_breaches / max(1, self.current_episode)) * 100,
            "peak_equity": self.peak_equity,
            "current_pnl": self.current_pnl
        }


# Callback for Stable-Baselines3 integration
class TrainingMonitorCallback:
    """Callback to integrate TrainingMonitor with SB3 training."""
    
    def __init__(self, monitor_frequency=100):
        self.monitor = TrainingMonitor(monitor_frequency)
        self.episode_start_time = time.time()
        self.episode_trades = 0
        self.episode_wins = 0
        self.episode_pnl = 0.0
        self.episode_reward = 0.0
    
    def on_step(self, timestep, reward, pnl_change=0, trade_outcome=None):
        """Called on each environment step."""
        self.episode_reward += reward
        self.episode_pnl += pnl_change
        
        if trade_outcome is not None:
            self.episode_trades += 1
            if trade_outcome > 0:  # Winning trade
                self.episode_wins += 1
        
        # Update training metrics
        self.monitor.update_training_metrics(timestep)
        
        # Print status periodically
        if self.monitor.should_update_display(timestep):
            self.monitor.print_training_status(timestep)
    
    def on_episode_end(self, episode_length):
        """Called at end of episode."""
        self.monitor.update_episode_metrics(
            self.episode_reward,
            self.episode_pnl,
            self.episode_trades,
            self.episode_wins,
            episode_length
        )
        
        # Reset episode tracking
        self.episode_trades = 0
        self.episode_wins = 0
        self.episode_pnl = 0.0
        self.episode_reward = 0.0
        self.episode_start_time = time.time()
    
    def on_training_end(self):
        """Called when training is complete."""
        self.monitor.print_training_status()
        self.monitor.save_metrics()
        logger.info("Training completed - final metrics saved")
