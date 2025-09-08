class RewardSystem:
    def __init__(self, target):
        self.target = target
        self.episode_start = 0
        self.last_pnl = 0
        self.last_drawdown = 0

    def compute_reward(self, state, action):
        reward = 0
        pnl = state['pnl']
        drawdown = state['drawdown']
        trades = state['trades']
        hours = (state.get('time', 0) - self.episode_start) / 3600

        # Primary Rewards (1-3)
        if pnl >= self.target and drawdown < 0.05: reward += 100  # 1: Daily goal
        if hours <= 1: reward += 100  # 2: Time bonus 1h
        elif hours <= 3: reward += 100  # 3h
        elif hours <= 6: reward += 100  # 6h
        elif hours <= 12: reward += 100  # 12h

        # Trade Performance (4-8)
        if pnl > self.last_pnl: reward += 1  # 4: Consecutive profitable
        else: reward -= 2
        if 500 <= trades <= 700: reward += 50  # 5: Frequency
        elif trades > 900: reward -= 50
        if pnl / trades > 0.6: reward += 50  # 6: Profit rate >60%
        if len(state.get('assets', [])) < 10: reward -= 50  # 7: Multi-asset <10 pairs
        if state.get('add_to_winner', False) and pnl > 0: reward += 0.01  # 8: Position mgmt

        # Advanced (9-13)
        if state.get('cci_entry', 0) < 100 and state.get('cci_exit', 0) > 160: reward += 1  # 9: CCI timing
        reward += 1  # 10: Cross-asset (mock)
        if pnl > self.last_pnl * 1.1: reward += 10  # 11: Consistency P&L
        if drawdown < self.last_drawdown: reward += 10  # 12: Risk mgmt
        if state.get('pullback', 1) < 0.5: reward += 0.01  # 13: Exit timing 50% pullback
        if state.get('sma_cross', 1) == 0: reward += 0.1  # 20 SMA cross

        # Penalties (14-16)
        if pnl < self.target or drawdown > 0.05: reward -= 100  # 14: Daily failure
        if hours > 1 and trades == 0: reward -= 2  # 15: Inactivity per hour
        if state.get('consec_fails', 0) > 3: reward -= 30  # 16: Consistency failure

        self.last_pnl = pnl
        self.last_drawdown = drawdown
        return reward
        self.consecutive_wins = 0
        self.hour_no_trade = 0
