import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from utils.config import config

# Import agents
from agents.q_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.dqn_keras_agent import DQNKerasAgent

# Import environment
from envs.airline_env import AirlineEnv

def main():
    env = AirlineEnv()
    agent_type = config['agent']
    if agent_type == "q_learning":
        agent = QLearningAgent(env)
    elif agent_type == "dqn":
        agent = DQNAgent(env)
    elif agent_type == "ppo":
        agent = PPOAgent(env)
    elif agent_type == "dqn_keras":
        agent = DQNKerasAgent(env)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    num_episodes = 1000  # Number of episodes for Q-learning and DQN
    total_timesteps = 50000  # Total timesteps for PPO and DQN with Keras
    results = []  # List to store results

    if agent_type in ["q_learning", "dqn"]:
        for episode in range(num_episodes):
            state = env.reset()  # Reset environment and get initial state
            total_reward = 0
            done = False

            while not done:
                if agent_type == "q_learning":
                    action = agent.choose_action(state)
                elif agent_type == "dqn":
                    action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward

                if agent_type == "q_learning":
                    agent.update(state, action, reward, next_state, done)
                elif agent_type == "dqn":
                    agent.remember(state, action, reward, next_state, done)
                    if len(agent.memory) > agent.batch_size:
                        agent.replay()

                state = next_state

            print(f"Episode: {episode + 1}, Total reward: {total_reward}, Epsilon: {agent.epsilon}")
            results.append((episode + 1, total_reward, agent.epsilon))

            # Optional: Save the Q-table or DQN model every 100 episodes
            if (episode + 1) % 100 == 0:
                if agent_type == "q_learning":
                    agent.save(f'q_table_{episode + 1}.npy')
                elif agent_type == "dqn":
                    agent.save(f'dqn_model_{episode + 1}.h5')

    elif agent_type in ["ppo", "dqn_keras"]:
        agent.learn(total_timesteps=total_timesteps)
        # Since PPO and DQN with Keras don't run in episodes, we'll save the model at the end
        agent.save(f'{agent_type}_model')

    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)

    # Save results to CSV file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'results/training_results_{timestamp}.csv'
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Total Reward', 'Epsilon'])
        writer.writerows(results)

    # Plotting the results
    if results:
        episodes, total_rewards, epsilons = zip(*results)
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward', color=color)
        ax1.plot(episodes, total_rewards, label='Total Reward', color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Epsilon', color=color)
        ax2.plot(episodes, epsilons, label='Epsilon', color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.title('Total Rewards and Epsilon per Episode')
        plt.savefig(f'results/training_results_{timestamp}.png')
        plt.show()

if __name__ == '__main__':
    main()
