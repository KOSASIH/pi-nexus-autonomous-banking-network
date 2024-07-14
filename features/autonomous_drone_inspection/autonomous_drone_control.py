# File name: autonomous_drone_control.py
import gym
import numpy as np
from stable_baselines3 import PPO

env = gym.make('DroneEnv-v0')
model = PPO('MlpPolicy', env, verbose=1)

def train_model():
    model.learn(total_timesteps=10000)
    model.save("autonomous_drone_control")

def test_model():
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(reward)

train_model()
test_model()
