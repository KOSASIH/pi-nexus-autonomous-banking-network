# Overview
This multi-signature wallet allows multiple owners to manage funds securely. Transactions require a predefined number of approvals before execution, enhancing security against unauthorized access.

## Smart Contract: multi_sig_wallet.sol
The smart contract implements a multi-signature wallet with the following features:

- **Deposit Funds**: Owners can deposit Ether into the wallet.
- **Submit Transactions**: Any owner can propose a transaction.
- **Approve Transactions**: Owners can approve proposed transactions.
- **Execute Transactions**: Once the required number of approvals is reached, the transaction can be executed.

## Key Functions

- submitTransaction(address to, uint value): Propose a transaction to send Ether.
- approveTransaction(uint txIndex): Approve a proposed transaction.
- executeTransaction(uint txIndex): Execute a transaction if it has enough approvals.
- getTransaction(uint txIndex): Retrieve details of a transaction.

## Python Service: multi_sig_service.py
This Python script provides an interface to interact with the multi-signature wallet smart contract using Web3.py.

## Key Methods
- submit_transaction(owner_address, to, value): Submit a transaction proposal.
- approve_transaction(owner_address, tx_index): Approve a transaction.
- execute_transaction(owner_address, tx_index): Execute a transaction.
- get_transaction(tx_index): Get details of a specific transaction.
- get_transaction_count(): Get the total number of transactions.

## Usage

- Deploy the Smart Contract: Deploy multi_sig_wallet.sol to an Ethereum network.
- Initialize the Service: Use the contract address and ABI to initialize MultiSigWalletService.
- Interact with the Wallet:
- Use the service methods to submit, approve, and execute transactions as needed.

## Example

```python
1 from multi_sig_service import MultiSigWalletService
2 
3 # Initialize the service
4 wallet_service = MultiSigWalletService('YOUR_CONTRACT_ADDRESS', 'YOUR_ABI')
5 
6 # Submit a transaction
7 tx_hash = wallet_service.submit_transaction('OWNER_ADDRESS', 'RECIPIENT_ADDRESS', 1_000_000_000_000_000_000)  # 1 Ether in Wei
8 print(f'Transaction submitted: {tx_hash}')
9 
10 # Approve a transaction
11 tx_hash = wallet_service.approve_transaction('OWNER_ADDRESS', 0)  # Approving the first transaction
12 print(f'Transaction approved: {tx_hash}')
13 
14 # Execute a transaction
15 tx_hash = wallet_service.execute_transaction('OWNER_ADDRESS', 0)  # Executing the first transaction
16 print(f'Transaction executed: {tx_hash}')
```

## Conclusion
This multi-signature wallet implementation provides a robust solution for managing funds with enhanced security through multiple approvals. The accompanying Python service allows for easy interaction with the smart contract, making it user-friendly for developers and users alike.
