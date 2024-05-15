import web3

# Connect to Ethereum node
w3 = web3.Web3(web3.HTTPProvider('http://localhost:8545'))

# Get the latest block number
latest_block_number = w3.eth.blockNumber

# Get the balance of an Ethereum address
eth_address = '0x5409ed021d9299bf6814279a6a1411a7e866a631'
balance = w3.eth.getBalance(eth_address)

# Convert the balance from wei to ether
balance_ether = w3.fromWei(balance, 'ether')

# Print the balance
print(f'The balance of {eth_address} is {balance_ether} ether.')

# Check if the address has any transactions
transactions = w3.eth.getTransactionCount(eth_address)

# Print the number of transactions
print(f'The number of transactions of {eth_address} is {transactions}.')
