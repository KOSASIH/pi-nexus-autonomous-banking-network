# Function to learn
def learn(self, state, action, reward, next_state):
    q_value = self.q_table[state][action]
    next_q_value = max(self.q_table[next_state])
    q_value += self.alpha * (reward + self.gamma * next_q_value - q_value)
    self.q_table[state][action] = q_value

# Function to train the agent
def train(self, episodes):
    for episode in range(episodes):
        state = self.env.reset()
        done = False
        rewards = 0
        while not done:
            action = self.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)
            rewards += reward
            self.learn(state, action, reward, next_state)
            state = next_state
        print(f"Episode {episode+1}, Reward: {rewards}")

# Function to test the agent
def test(self, episodes):
    for episode in range(episodes):
        state = self.env.reset()
        done = False
        rewards = 0
        while not done:
            action = np.argmax(self.q_table[state])
            next_state, reward, done, _ = self.env.step(action)
            rewards += reward
            state = next_state
        print(f"Episode {episode+1}, Reward: {rewards}")

# Example usage
env = gym.make('CartPole-v1')
agent = PiNetworkReinforcementLearningAgent(env, alpha=0.1, epsilon=0.1, gamma=0.9)
agent.train(episodes=1000)
agent.test(episodes=10)
