# sidra_chain_reinforcement_learning.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class SidraChainReinforcementLearning:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.gamma = 0.99
        self.epsilon = 0.1

    def train(self, episodes=1000):
        rewards = []
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                self.agent.update(state, action, next_state, reward, done)
                state = next_state
            rewards.append(episode_reward)
            print(f"Episode {episode+1}, Reward: {episode_reward}")
        return rewards

    def test(self, episodes=10):
        rewards = []
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
            rewards.append(episode_reward)
            print(f"Episode {episode+1}, Reward: {episode_reward}")
        return rewards

class SidraChainAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SidraChainAgent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action = self.forward(state)
        action = torch.argmax(action)
        return action.item()

    def update(self, state, action, next_state, reward, done):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.bool)

        # Calculate TD error
        td_error = reward + self.gamma * self.forward(next_state) - self.forward(state)

        # Update policy
        self.optimizer.zero_grad()
        loss = td_error ** 2
        loss.backward()
        self.optimizer.step()
