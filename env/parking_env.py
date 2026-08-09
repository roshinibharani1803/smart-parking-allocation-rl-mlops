import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class ParkingEnv(gym.Env):

    def __init__(self):

        super(ParkingEnv, self).__init__()

        # Total parking slots
        self.total_slots = 5
        self.slot_info = {

            0: {
                "distance": 80,
                "charger": False,
                "vip": False
            },

            1: {
                "distance": 20,
                "charger": True,
                "vip": False
            },

            2: {
                "distance": 50,
                "charger": False,
                "vip": True
            },

            3: {
                "distance": 10,
                "charger": False,
                "vip": False
            },

            4: {
                "distance": 100,
                "charger": False,
                "vip": False
            }

        }

        # Action space:
        # Agent can choose slot 0 to 4
        self.action_space = spaces.Discrete(self.total_slots)

        # Observation space:
        # 0 = free
        # 1 = occupied
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.total_slots,),
            dtype=np.int32
        )

        # Initial parking state
        self.state = np.zeros(self.total_slots, dtype=np.int32)

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        # Random parking occupancy
        self.state = np.random.randint(0, 2, size=(self.total_slots,))

        return self.state, {}

    def step(self, action):

        reward = 0
        done = False

        # If chosen slot is free
        if self.state[action] == 0:

            reward = 50

            # Reward for closer slots
            reward += (100 - self.slot_info[action]["distance"]) * 0.5

            # Bonus for EV charging
            if self.slot_info[action]["charger"]:
                reward += 20

            # Penalty for VIP slots
            if self.slot_info[action]["vip"]:
                reward -= 40
            self.state[action] = 1

        else:

            reward = -50
        

        # End episode if all slots occupied
        if np.all(self.state == 1):
            done = True

        return self.state, reward, done, False, {}

    def render(self):

        print("\n========== Parking Lot Status ==========\n")

        for i in range(self.total_slots):

            status = "Occupied" if self.state[i] else "Free"

            print(
                f"Slot {i} | "
                f"{status} | "
                f"Distance: {self.slot_info[i]['distance']} m | "
                f"EV Charger: {self.slot_info[i]['charger']} | "
                f"VIP: {self.slot_info[i]['vip']}"
            )

        print()