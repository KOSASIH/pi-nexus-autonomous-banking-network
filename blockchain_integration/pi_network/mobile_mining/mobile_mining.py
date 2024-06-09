import hashlib

class MobileMining:
    def __init__(self):
        self.mining_pool = []

    def add_miner(self, miner):
        self.mining_pool.append(miner)

    def mine_pi_coins(self):
        # Mine Pi coins using mobile device's processing power
        for miner in self.mining_pool:
            # Simulate mining process
            nonce = 0
            while True:
                hash = hashlib.sha256(str(nonce).encode()).hexdigest()
                if hash[:4] == '0000':
                    break
                nonce += 1
            # Reward miner with Pi coins
            miner.reward_pi_coins(10)

if __name__ == '__main__':
    mm = MobileMining()
    miner1 = {'id': 1, 'pi_coins': 0}
    miner2 = {'id': 2, 'pi_coins': 0}
    mm.add_miner(miner1)
    mm.add_miner(miner2)
    mm.mine_pi_coins()
    print(miner1['pi_coins'])  # 10
    print(miner2['pi_coins'])  # 10
