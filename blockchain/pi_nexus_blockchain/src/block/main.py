from blockchain import Blockchain

def main():
    blockchain = Blockchain()

    # Create the first block
    blockchain.add_block("First block")

    # Add a few more blocks
    for i in range(2, 6):
        blockchain.add_block(f"Block {i}")

    # Print the blockchain
    for block in blockchain.get_blockchain():
        print(block)

if __name__ == "__main__":
    main()
