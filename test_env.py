from dynamic_pricing_env import DynamicPricingEnv

env = DynamicPricingEnv()
obs, _ = env.reset()
print("Initial Observation:", obs)

action = env.action_space.sample()
obs, reward, done, _, _ = env.step(action)
print("Action:", action, "New Observation:", obs, "Reward:", reward)
