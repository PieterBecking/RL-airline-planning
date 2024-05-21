import numpy as np

class QLearningAgent:
    def __init__(self, env):
        self.env = env
        self.learning_rate = 0.01
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.q_table = {}  # Use a dictionary to store Q-values

    def choose_action(self, state):
        state = tuple(state)  # Ensure the state is hashable and consistent
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        self.ensure_state_exists(state)
        return np.argmax(self.q_table[state])  # Exploit best known action

    def update(self, state, action, reward, next_state, done):
        state = tuple(state)
        next_state = tuple(next_state)
        self.ensure_state_exists(state)
        self.ensure_state_exists(next_state)

        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])

        # Update rule for Q-learning
        new_value = old_value + self.learning_rate * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action] = new_value

        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def ensure_state_exists(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.env.action_space.n)

    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path, allow_pickle=True).item()
