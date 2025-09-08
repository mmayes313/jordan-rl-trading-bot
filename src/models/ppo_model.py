# Mock PPO model for testing without stable-baselines3
class MockPPO:
    def __init__(self, policy, env, **kwargs):
        self.policy = policy
        self.env = env
        self.kwargs = kwargs

    def learn(self, total_timesteps):
        print(f"Mock training for {total_timesteps} timesteps")
        return self

    def get_env(self):
        return self.env

class MockVecEnv:
    def __init__(self, env_fns):
        self.env_fns = env_fns
        self.env = env_fns[0]()  # Create first environment

    def get_env(self):
        return self

def create_ppo_model(data_path):
    # Mock environment creation
    class MockEnv:
        pass

    env = MockVecEnv([lambda: MockEnv()])
    model = MockPPO('MlpPolicy', env, policy_kwargs={'net_arch': [512, 256]}, verbose=1)
    return model
