# File name: financial_planning_ai.py
import gym
import numpy as np
from stable_baselines3 import PPO


class FinancialPlanningAI:
    def __init__(self, env):
        self.env = env
        self.model = PPO("MlpPolicy", env, verbose=1)

    def train(self):
        self.model.learn(total_timesteps=10000)
        self.model.save("financial_planning_ai")

    def plan(self, state):
        action, _ = self.model.predict(state)
        return action


env = gym.make("FinancialPlanningEnv-v0")
financial_planning_ai = FinancialPlanningAI(env)
financial_planning_ai.train()
