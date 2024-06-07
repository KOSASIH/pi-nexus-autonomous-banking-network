import torch
import torch.nn as nn
from explainable_rl import ExplainableRL

class ExplainableRLAgent:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions
        self.explainable_rl = ExplainableRL(num_states, num_actions)

    def learn(self, experiences):
        self.explainable_rl.learn(experiences)

    def act(self, state):
        action, explanation = self.explainable_rl.act(state)
        return action, explanation

class AutonomousSystem:
    def __init__(self, explainable_rl_agent):
        self.explainable_rl_agent = explainable_rl_agent

    def make_autonomous_decision(self, state):
        action, explanation = self.explainable_rl_agent.act(state)
        return action, explanation
