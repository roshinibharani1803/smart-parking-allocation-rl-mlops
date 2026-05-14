import numpy as np
import random
import time
import mlflow
import matplotlib.pyplot as plt

from env.parking_env import ParkingEnv

env = ParkingEnv()

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

episodes = 100

# Q-table
q_table = np.zeros((32, env.num_slots))

reward_history = []

mlflow.set_experiment("SmartParkingRL")

with mlflow.start_run():

    mlflow.log_param("alpha", alpha)
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("epsilon_decay", epsilon_decay)

    for episode in range(episodes):

        state, _ = env.reset()

        state_index = int("".join(map(str, state)), 2)

        total_reward = 0
        done = False

        while not done:

            if random.uniform(0,1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state_index])

            next_state, reward, done, _, _ = env.step(action)

            next_state_index = int("".join(map(str, next_state)), 2)

            q_table[state_index][action] = q_table[state_index][action] + alpha * (
                reward + gamma * np.max(q_table[next_state_index]) - q_table[state_index][action]
            )

            state_index = next_state_index

            total_reward += reward

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        reward_history.append(total_reward)

        mlflow.log_metric("reward", total_reward, step=episode)

        print(f"Episode {episode+1} Reward: {total_reward}")

    timestamp = int(time.time())

    model_path = f"models/q_table_{timestamp}.npy"

    np.save(model_path, q_table)

    mlflow.log_artifact(model_path)

    plt.plot(reward_history)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Training Rewards")

    graph_path = "logs/reward_plot.png"

    plt.savefig(graph_path)

    mlflow.log_artifact(graph_path)

print("Training Complete")