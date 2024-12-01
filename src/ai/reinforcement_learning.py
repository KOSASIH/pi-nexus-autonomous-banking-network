import numpy as np

class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.99):
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.q_table = {}

    def get_q_value(self, state, action):
        """Get the Q-value for a given state and action."""
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state, action, reward, next_state):
        """Update the Q-value based on the action taken."""
        best_next_q = max([self.get_q_value(next_state, a) for a in self.actions], default=0.0)
        current_q = self.get_q_value(state, action)
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * best_next_q)
        self.q_table[(state, action)] = new_q

    def choose_action(self, state):
        """Choose an action based on the exploration-exploitation trade-off."""
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.actions)  # Explore
        else:
            return max(self.actions, key=lambda a: self.get_q_value(state, a))  # Exploit

    def decay_exploration(self):
        """Decay the exploration rate."""
        self.exploration_rate *= self.exploration_decay
