import gym
import numpy as np
from stable_baselines3 import PPO

class ReinforcementLearningAgent:
    def __init__(self, env, num_episodes):
        self.env = env
        self.num_episodes = num_episodes
        self.model = PPO('MlpPolicy', self.env, verbose=1)

    def train(self):
        self.model.learn(total_timesteps=self.num_episodes)

    def make_decision(self, state):
        action, _ = self.model.predict(state)
        return action

class ReinforcementLearningSystem:
    def __init__(self, reinforcement_learning_agent):
        self.reinforcement_learning_agent = reinforcement_learning_agent

    def make_autonomous_decision(self, state):
        action = self.reinforcement_learning_agent.make_decision(state)
        return action
