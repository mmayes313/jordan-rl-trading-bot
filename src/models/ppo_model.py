from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch
import numpy as np
from loguru import logger
from pathlib import Path

# Import our training monitor
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.training_monitor import TrainingMonitor

class PPOTrainingCallback(BaseCallback):
    """
    Custom callback to integrate with our TrainingMonitor.
    Captures training metrics and episode data for real-time display.
    """
    
    def __init__(self, monitor_frequency=100):
        super().__init__()
        self.monitor = TrainingMonitor(monitor_frequency)
        self.episode_rewards = []
        self.episode_pnls = []
        self.episode_trades = []
        self.episode_wins = []
        self.current_episode_reward = 0
        self.current_episode_pnl = 0
        self.current_episode_trades = 0
        self.current_episode_wins = 0
        
    def _on_step(self) -> bool:
        """Called after each environment step."""
        # Get current environment info
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            
            # Extract metrics from environment info
            reward = self.locals.get('rewards', [0])[0]
            pnl_change = info.get('pnl', 0)
            trade_outcome = info.get('trade_outcome', None)
            
            # Update episode tracking
            self.current_episode_reward += reward
            self.current_episode_pnl += pnl_change
            
            if trade_outcome is not None:
                self.current_episode_trades += 1
                if trade_outcome > 0:
                    self.current_episode_wins += 1
        
        # Update training metrics from logger
        if hasattr(self.model, 'logger') and self.model.logger:
            self.monitor.update_training_metrics(
                timesteps=self.num_timesteps,
                policy_loss=self.model.logger.name_to_value.get('train/policy_gradient_loss', 0),
                value_loss=self.model.logger.name_to_value.get('train/value_loss', 0),
                entropy_loss=self.model.logger.name_to_value.get('train/entropy_loss', 0),
                kl_div=self.model.logger.name_to_value.get('train/approx_kl', 0)
            )
        
        # Print status periodically
        if self.monitor.should_update_display(self.num_timesteps):
            self.monitor.print_training_status(self.num_timesteps)
        
        return True
    
    def _on_episode_end(self):
        """Called at the end of each episode."""
        # Calculate episode length
        episode_length = len(self.locals.get('rewards', []))
        
        # Update monitor with episode data
        self.monitor.update_episode_metrics(
            self.current_episode_reward,
            self.current_episode_pnl,
            self.current_episode_trades,
            self.current_episode_wins,
            episode_length
        )
        
        # Reset episode counters
        self.current_episode_reward = 0
        self.current_episode_pnl = 0
        self.current_episode_trades = 0
        self.current_episode_wins = 0
    
    def _on_training_end(self):
        """Called when training ends."""
        self.monitor.print_training_status()
        self.monitor.save_metrics()
        logger.info("🎉 Training completed! Final metrics saved.")

def create_ppo_model(env, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
                     gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
                     device='cuda' if torch.cuda.is_available() else 'cpu', 
                     monitor_training=True, monitor_frequency=100):
    """
    Create a PPO model with specified hyperparameters and optional training monitoring.

    Args:
        env: Gym environment
        learning_rate: Learning rate for optimizer
        n_steps: Number of steps to collect before updating
        batch_size: Minibatch size for optimization
        n_epochs: Number of epochs for optimization
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        device: Device to run on ('cuda' or 'cpu')
        monitor_training: Whether to enable real-time training monitoring
        monitor_frequency: How often to display training stats (in timesteps)

    Returns:
        Tuple: (PPO model ready for training, training callback if monitoring enabled)
    """
    # Wrap environment if needed
    if not hasattr(env, 'num_envs'):
        env = DummyVecEnv([lambda: env])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=1,
        device=device
    )
    
    # Create training callback for monitoring
    callback = None
    if monitor_training:
        callback = PPOTrainingCallback(monitor_frequency)
        logger.info(f"Training monitoring enabled - updates every {monitor_frequency} timesteps")
    
    return model, callback

def train_model_with_monitoring(env, total_timesteps=1000000, 
                               device='cuda' if torch.cuda.is_available() else 'cpu',
                               monitor_frequency=100, save_path="data/models/best_model"):
    """
    Train a PPO model with comprehensive monitoring and real-time metrics.
    
    Args:
        env: Trading environment
        total_timesteps: Total training timesteps
        device: Device to use for training
        monitor_frequency: How often to display training stats
        save_path: Path to save the trained model
        
    Returns:
        Trained PPO model
    """
    logger.info("🚀 Starting PPO training with real-time monitoring...")
    
    # Create model with monitoring
    model, callback = create_ppo_model(
        env, 
        device=device, 
        monitor_training=True, 
        monitor_frequency=monitor_frequency
    )
    
    # Ensure save directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Start training with callback
    logger.info(f"Training for {total_timesteps:,} timesteps...")
    logger.info("🎯 Monitoring: PnL, Rewards, Drawdown, Daily Goals, Win/Loss Rate, Trade Count")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # Save the trained model
    model.save(save_path)
    logger.info(f"💾 Model saved to {save_path}")
    
    # Final training summary
    if callback:
        stats = callback.monitor.get_summary_stats()
        logger.info("🏁 Training Summary:")
        logger.info(f"   Episodes: {stats.get('total_episodes', 0)}")
        logger.info(f"   Avg Reward: {stats.get('avg_reward', 0):.4f}")
        logger.info(f"   Avg PnL: ${stats.get('avg_pnl', 0):.2f}")
        logger.info(f"   Win Rate: {stats.get('win_rate', 0):.1f}%")
        logger.info(f"   Goal Success: {stats.get('goal_success_rate', 0):.1f}%")
        logger.info(f"   Final Equity: ${stats.get('peak_equity', 1000):.2f}")
    
    return model

def train_model(env, total_timesteps=1000000, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Legacy training function - use train_model_with_monitoring for enhanced monitoring."""
    model, _ = create_ppo_model(env, device=device, monitor_training=False)
    model.learn(total_timesteps=total_timesteps)
    model.save("data/models/best_model")
    return model

# Load: model = PPO.load("data/models/best_model")
