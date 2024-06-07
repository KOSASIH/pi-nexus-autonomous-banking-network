import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        return samples

class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, learning_rate, epsilon, epsilon_decay):
        self.dqn = DQN(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(10000)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, action_dim)
        else:
            return self.dqn(state).argmax().item()

    def learn(self, batch_size):
        samples = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float)

        q_values = self.dqn(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        target_q_values = rewards + 0.99 * self.dqn(next_states).max(1)[0] * (1 - dones)
        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon *= self.epsilon_decay
