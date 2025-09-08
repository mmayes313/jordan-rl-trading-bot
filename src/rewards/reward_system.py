from config.trading_config import DAILY_TARGET_PCT, TRAILING_DD_PCT

class RewardSystem:
    def __init__(self):
        self.episode_start_balance = 0
        self.trades_today = []
        self.consecutive_wins = 0
        self.hour_no_trade = 0
        self.last_three_pnls = []
        self.last_three_dds = []
        self.all_time_high_pnl = float('-inf')
        self.all_time_low_dd = float('inf')
        self.episode_count = 0
        self.successful_episodes = 0

    def get_reward(self, state, action, pnl_change, is_end_of_episode=False):
        reward = 0.0
        current_time = state['time_of_day']  # Normalized

        # Per-step rewards
        if pnl_change > 0:
            self.consecutive_wins += 1
            reward += 1.0  # Rule 2
            if action == 'add_to_winner':  # Rule 8
                reward += 0.01
        else:
            self.consecutive_wins = 0
            reward -= 2  # Rule 2

        # Rule 12: -2 per hour no trade
        if 'no_trade' in state:
            self.hour_no_trade += 1
            reward -= 2 * self.hour_no_trade

        # Rule 13/14: CCI timing
        if 'cci_entry_before_100' in state and pnl_change > 0 and state.get('cci_15m_30', 0) > 160:
            reward += 0.1

        # Rule 15: Cross-asset
        if 'new_asset' in state:
            reward += 1

        # Rule 18: Close before 50% pullback
        if 'closed_before_pullback' in state:
            reward += 0.01

        # Rule 19: Close before SMA cross
        if 'closed_before_sma_cross' in state:
            reward += 0.1

        # End-of-episode rewards
        if is_end_of_episode:
            daily_pnl_pct = (state['equity'] - self.episode_start_balance) / self.episode_start_balance
            dd_breached = state['max_dd'] > TRAILING_DD_PCT

            if daily_pnl_pct >= DAILY_TARGET_PCT and not dd_breached:
                reward += 100  # Rule 1
                hours_taken = current_time * 24
                if hours_taken <= 1: reward += 100
                elif hours_taken <= 3: reward += 100
                elif hours_taken <= 6: reward += 100
                elif hours_taken <= 12: reward += 100
                if daily_pnl_pct >= 0.50 and not dd_breached: reward += 500  # Rule 5
                if daily_pnl_pct >= 1.00 and not dd_breached: reward += 2000  # Rule 6
                if daily_pnl_pct > max(self.last_three_pnls or [0]): reward += 10  # Rule 16
                if state['max_dd'] < min(self.last_three_dds or [float('inf')]): reward += 10  # Rule 17
                self.successful_episodes += 1
            else:
                reward -= 100  # Rule 1 penalty
                # Rule 21: Consistent failure (could add more logic)

            # Rule 4/7: Profit rate
            if self.trades_today:
                prof_rate = sum(1 for t in self.trades_today if t['pnl'] > 0) / len(self.trades_today)
                if prof_rate > 0.60: reward += 50  # Rule 4
                if prof_rate > 0.50: reward += 50  # Rule 7

            # Rule 9: <10 pairs
            unique_assets = len(set(t['asset'] for t in self.trades_today))
            if unique_assets < 10: reward -= 50

            # Rule 3: Trade count
            trade_count = len(self.trades_today)
            if 500 <= trade_count <= 700: reward += 50
            elif trade_count > 900: reward -= 50

            # Update records for Rule 10/11
            self.update_at_records(state['pnl'], state['max_dd'])

            # Rule 20: Consistent success
            if self.episode_count >= 3 and self.successful_episodes / self.episode_count > 0.8:
                reward += 100

            self.last_three_pnls.append(daily_pnl_pct)
            self.last_three_dds.append(state['max_dd'])
            if len(self.last_three_pnls) > 3: self.last_three_pnls.pop(0)
            if len(self.last_three_dds) > 3: self.last_three_dds.pop(0)
            self.episode_count += 1

        return reward

    def update_at_records(self, pnl, dd):
        if pnl > self.all_time_high_pnl:
            self.all_time_high_pnl = pnl
            # Rule 10: + bonus for new high
        if dd < self.all_time_low_dd:
            self.all_time_low_dd = dd
            # Rule 11: + bonus for new low DD

    def reset_episode(self, start_balance):
        self.episode_start_balance = start_balance
        self.trades_today = []
        self.consecutive_wins = 0
        self.hour_no_trade = 0
