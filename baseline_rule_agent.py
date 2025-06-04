from dynamic_pricing_env import DynamicPricingEnv
import numpy as np

env = DynamicPricingEnv()
num_episodes = 500
rewards_per_episode = []
rule_based_action = 2

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        next_state, reward, done, _, _ = env.step(rule_based_action)
        total_reward += reward
    rewards_per_episode.append(total_reward)

average_revenue = np.mean(rewards_per_episode)
print(f"Rule-based agent average revenue over {num_episodes} episodes: {average_revenue:.2f}")
