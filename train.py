import numpy as np
import torch
from dynamic_pricing_env import DynamicPricingEnv
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt

# Initialize environment and agent
env = DynamicPricingEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim)

max_q_values = []  # Add this at the top

num_episodes = 1500
#target_update_freq = 10  # how often to update target network
target_update_freq = 5  
rewards_per_episode = []

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    max_q_episode = -float('inf')
    
    while not done:
        action = agent.select_action(state)
        with torch.no_grad():
            q_values = agent.q_network(torch.FloatTensor(state).unsqueeze(0))
        max_q_episode = max(max_q_episode, q_values.max().item())
        
        next_state, reward, done, _, _ = env.step(action)
        agent.store_experience(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        total_reward += reward
    
    if episode % target_update_freq == 0:
        agent.update_target_network()
    
    rewards_per_episode.append(total_reward)
    max_q_values.append(max_q_episode)
    
    if (episode + 1) % 50 == 0:
        avg_reward = np.mean(rewards_per_episode[-50:])
        print(f"Episode {episode + 1}/{num_episodes}, Average Revenue: {avg_reward:.2f}")


average_dqn_revenue = np.mean(rewards_per_episode)
print(f"DQN agent average revenue over {num_episodes} episodes: {average_dqn_revenue:.2f}")

baseline_avg_revenue = 6000.00  # updated for 10-step episodes
improvement = ((average_dqn_revenue - baseline_avg_revenue) / baseline_avg_revenue) * 100
print(f"Revenue improved by: {improvement:.2f}%")

# Plotting revenue trends
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Total Revenue')
plt.title('Revenue over Episodes')

# Plotting Q-value curves
plt.subplot(1, 2, 2)
plt.plot(max_q_values)
plt.xlabel('Episode')
plt.ylabel('Max Q-Value')
plt.title('Max Q-Value over Episodes')

plt.tight_layout()
plt.show()
plt.savefig("plots/revenue_qvalue.png")
