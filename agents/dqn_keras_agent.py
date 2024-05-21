import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

class DQNKerasAgent:
    def __init__(self, env):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.memory_size = 50000
        self.learning_rate = 0.001
        self.batch_size = 32
        self.model = self._build_model()
        self.policy = EpsGreedyQPolicy()
        self.memory = SequentialMemory(limit=self.memory_size, window_length=1)
        self.dqn = DQNAgent(model=self.model, nb_actions=self.action_size, memory=self.memory, nb_steps_warmup=1000,
                            target_model_update=1e-2, policy=self.policy)
        self.dqn.compile(Adam(lr=self.learning_rate), metrics=['mae'])

    def _build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1, self.state_size)))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        return model

    def learn(self, total_timesteps):
        self.dqn.fit(self.env, nb_steps=total_timesteps, visualize=False, verbose=2)

    def predict(self, state):
        action = self.dqn.forward(state)
        return action

    def save(self, path):
        self.dqn.save_weights(path, overwrite=True)

    def load(self, path):
        self.dqn.load_weights(path)
