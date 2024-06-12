import random

class Gamification:
    def __init__(self, user_data):
        self.user_data = user_data

    def reward_user(self, user_id, reward_amount):
        self.user_data[user_id]['balance'] += reward_amount
        return self.user_data[user_id]['balance']

    def leaderboard(self):
        sorted_users = sorted(self.user_data.items(), key=lambda x: x[1]['balance'], reverse=True)
        return sorted_users

# Example usage:
user_data = {'user1': {'balance': 100}, 'user2': {'balance': 50}, 'user3': {'balance': 200}}
gamification = Gamification(user_data)
user_id = 'user1'
reward_amount = 50
new_balance = gamification.reward_user(user_id, reward_amount)
print(new_balance)
leaderboard = gamification.leaderboard()
print(leaderboard)
