# app.py
from wallet import Wallet
from blockchain import Blockchain
from node import Node
from exchange import Exchange

# Initialize the wallet, blockchain, node, and exchange
wallet = Wallet()
blockchain = Blockchain(wallet)
node = Node(wallet, blockchain)
exchange = Exchange({'coins_file': 'coins.json'})
exchange.load_coins()

# Mine thegenesis block
miner = Miner(wallet, blockchain)
miner.mine_new_block([])

# Start the node
node.join_network('localhost', 8080)

# Start the exchange
exchange.start()

# Run the application
while True:
    command = input('> ')
    if command == 'ine':
        miner.mine_new_block([])
    elif command == 'list_coins':
        print(exchange.list_coins())
    elif command == 'et_coin_value':
        coin = input('Enter the coin symbol: ')
        value = float(input('Enter the coin value: '))
        exchange.set_coin_value(coin, value)
    elif command == 'trade_coins':
        coin1 = input('Enter the coin to trade: ')
        coin2 = input('Enter the coin to receive: ')
        amount = float(input('Enter the amount to trade: '))
        exchange.trade_coins(coin1, coin2, amount)
    elif command == 'get_balance':
        coin = input('Enter the coin symbol: ')
        print(exchange.get_balance(coin))
    elif command == 'quit':
        break
    else:
        print('Invalid command')
