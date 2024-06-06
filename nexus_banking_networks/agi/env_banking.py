import gym
import numpy as np

class BankingEnvironment(gym.Env):
    def __init__(self, num_accounts, num_transactions):
        self.num_accounts = num_accounts
        self.num_transactions = num_transactions
        self.state_dim = num_accounts * num_transactions
        self.action_dim = num_accounts

    def reset(self):
        self.accounts = np.random.rand(self.num_accounts)
        self.transactions = np.random.rand(self.num_transactions, self.num_accounts)
        return self.encode_state()

    def step(self, action):
        # Update accounts and transactions based on action
        self.accounts += np.random.rand(self.num_accounts)
        self.transactions = np.roll(self.transactions, 1, axis=0)
        reward = np.sum(self.accounts)
        done = False
        return self.encode_state(), reward, done, {}

    def encode_state(self):
        state = np.concatenate((self.accounts, self.transactions.flatten()))
        return state

    def encode_attention_mask(self, state):
        attention_mask = np.ones((self.num_transactions, self.num_accounts))
        return attention_mask
