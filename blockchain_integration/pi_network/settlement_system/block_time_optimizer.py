import time

class BlockTimeOptimizer:
    def __init__(self):
        self.block_time = 10  # seconds

    def optimize_block_time(self):
        # Optimize block time based on network conditions
        current_time = time.time()
        if current_time - self.block_time > 10:
            self.block_time -= 1
        elif current_time - self.block_time < 5:
            self.block_time += 1
        return self.block_time

if __name__ == '__main__':
    bto = BlockTimeOptimizer()
    print(bto.optimize_block_time())
