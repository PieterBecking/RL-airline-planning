import gym
from gym import spaces
import numpy as np
import pandas as pd
from utils.config import config

class AirlineEnv(gym.Env):
    """
    A reinforcement learning environment for airline scheduling, loading data from CSV files.
    """
    def __init__(self):
        super(AirlineEnv, self).__init__()
        self.num_airports = config['num_airports']
        self.time_slots = config['time_slots']
        self.aircraft_capacity = config['aircraft_capacity']
        self.cost_parameter = config['cost_parameter']
        self.max_aircraft = config['max_aircraft']  # Max number of aircraft available
        
        # Load data from CSV files
        self.load_data()

        # Define action and state spaces:
        # Actions are choosing to schedule or not schedule each possible flight in each time slot
        self.action_space = spaces.Discrete(self.num_airports * self.num_airports * self.time_slots)
        self.observation_space = spaces.MultiBinary(self.num_airports * self.num_airports * self.time_slots)

        self.state = np.zeros((self.num_airports, self.num_airports, self.time_slots))
        self.current_time_slot = 0  # Current time slot in the day
        self.available_aircraft = self.max_aircraft  # Track available aircraft

    def load_data(self):
        # Load matrices from CSV files
        self.demand_matrix = pd.read_csv('data/demand_matrix.csv').values
        self.price_matrix = pd.read_csv('data/price_matrix.csv').values
        self.distance_matrix = pd.read_csv('data/distance_matrix.csv').values

    def step(self, action):
        # Decode the action into specific flight and time slot
        airport_pair_index = action // self.time_slots
        time_slot_index = action % self.time_slots
        i, j = divmod(airport_pair_index, self.num_airports)

        reward = 0
        if self.state[i, j, time_slot_index] == 0 and self.available_aircraft > 0:
            self.state[i, j, time_slot_index] = 1  # Schedule the flight
            self.available_aircraft -= 1  # Use one aircraft

            passengers = min(self.demand_matrix[i, j], self.aircraft_capacity)
            revenue = passengers * self.price_matrix[i, j]
            cost = self.distance_matrix[i, j] * self.cost_parameter
            reward = revenue - cost

            # Update demand
            self.demand_matrix[i, j] -= passengers
            if self.demand_matrix[i, j] < 0:
                self.demand_matrix[i, j] = 0  # Ensure demand doesn't go negative

        else:
            reward = -10  # Penalize redundant or invalid actions

        self.current_time_slot += 1

        # Check if the day is over
        done = self.current_time_slot >= self.time_slots
        if done:
            self.reset()

        return self.state.flatten(), reward, done, {}

    def reset(self):
        self.state = np.zeros((self.num_airports, self.num_airports, self.time_slots))
        self.current_time_slot = 0
        self.available_aircraft = self.max_aircraft  # Reset available aircraft

        # Reload data
        self.load_data()

        return self.state.flatten()

    def render(self, mode='human'):
        print(f"Current state at time slot {self.current_time_slot}:")
        print(self.state)
        print(f"Available aircraft: {self.available_aircraft}")
