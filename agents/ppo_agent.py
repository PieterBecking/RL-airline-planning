from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class PPOAgent:
    def __init__(self, env):
        self.env = DummyVecEnv([lambda: env])
        self.model = PPO('MlpPolicy', self.env, verbose=1)

    def learn(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps)

    def predict(self, state):
        action, _ = self.model.predict(state)
        return action

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = PPO.load(path)
