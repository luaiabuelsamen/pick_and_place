from ppo_train import PickPlaceEnvImage
from stable_baselines3 import PPO

env = PickPlaceEnvImage(add_proprioception=True, use_viewer= True)
print("Loading trained model...")
model = PPO.load("best_model/best_model")
obs, _ = env.reset()
for _ in range(1000):  # Run for a few steps to visualize
    action, _ = model.predict(obs, deterministic=True)
    print(action)
    obs, reward, terminated, truncated, info = env.step(action)
    env.env.render()  # Render the environment for visualization
    if terminated or truncated:
        obs, _ = env.reset()
print("Visualization complete.")
