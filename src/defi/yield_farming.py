import time
from collections import defaultdict

class Token:
    def __init__(self, name, symbol, total_supply):
        self.name = name
        self.symbol = symbol
        self.total_supply = total_supply
        self.balances = defaultdict(int)

    def transfer(self, from_address, to_address, amount):
        if self.balances[from_address] < amount:
            raise ValueError("Insufficient balance")
        self.balances[from_address] -= amount
        self.balances[to_address] += amount

    def mint(self, to_address, amount):
        self.balances[to_address] += amount
        self.total_supply += amount

class YieldFarming:
    def __init__(self, token, reward_token):
        self.token = token
        self.reward_token = reward_token
        self.stakers = {}
        self.total_staked = 0
        self.reward_rate = 0.01  # 1% reward per block
        self.last_update_time = time.time()

    def stake(self, amount, user):
        if amount <= 0:
            raise ValueError("Amount must be greater than 0")
        self.token.transfer(user, self, amount)
        if user not in self.stakers:
            self.stakers[user] = {'amount': 0, 'reward_debt': 0}
        self.stakers[user]['amount'] += amount
        self.total_staked += amount
        print(f"{user} has staked {amount} {self.token.symbol}.")

    def withdraw(self, amount, user):
        if user not in self.stakers or self.stakers[user]['amount'] < amount:
            raise ValueError("Insufficient staked amount")
        self.update_rewards(user)
        self.stakers[user]['amount'] -= amount
        self.total_staked -= amount
        self.token.transfer(self, user, amount)
        print(f"{user} has withdrawn {amount} {self.token.symbol}.")

    def claim_rewards(self, user):
        self.update_rewards(user)
        reward = self.stakers[user]['reward_debt']
        if reward > 0:
            self.reward_token.mint(user, reward)
            self.stakers[user]['reward_debt'] = 0
            print(f"{user} has claimed {reward} {self.reward_token.symbol} as rewards.")

    def update_rewards(self, user):
        if user in self.stakers:
            time_passed = time.time() - self.last_update_time
            reward = (self.stakers[user]['amount'] * self.reward_rate * time_passed)
            self.stakers[user]['reward_debt'] += reward
            self.last_update_time = time.time()

# Example usage
if __name__ == "__main__":
    # Create tokens
    token_a = Token("TokenA", "TKA", 1000000)
    reward_token = Token("RewardToken", "RWT", 1000000)

    # Mint some tokens for the user
    token_a.mint("user_address", 10000)
    reward_token.mint("farming_contract", 100000)  # Mint rewards for the contract

    # Create a yield farming contract
    yield_farming = YieldFarming(token_a, reward_token)

    # User stakes tokens
    yield_farming.stake(1000, "user_address")

    # Simulate time passing
    time.sleep(2)  # Simulate 2 seconds passing

    # User claims rewards
    yield_farming.claim_rewards("user_address")

    # User withdraws staked tokens
    yield_farming.withdraw(500, "user_address")
