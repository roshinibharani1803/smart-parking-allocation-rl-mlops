import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ParkingEnv(gym.Env):

    def __init__(self):
        super(ParkingEnv, self).__init__()

        self.num_slots = 5

        self.action_space = spaces.Discrete(self.num_slots)

        self.observation_space = spaces.MultiBinary(self.num_slots)

        self.state = np.zeros(self.num_slots, dtype=int)

    def reset(self, seed=None, options=None):

        self.state = np.zeros(self.num_slots, dtype=int)

        return self.state, {}

    def step(self, action):

        reward = 0

        if self.state[action] == 0:
            self.state[action] = 1
            reward = 10
        else:
            reward = -10

        done = np.all(self.state == 1)

        return self.state, reward, done, False, {}