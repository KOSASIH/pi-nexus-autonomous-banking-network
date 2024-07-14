// File name: secure_transactions.sol
pragma solidity ^0.8.0;

contract SecureTransactions {
    address private owner;
    mapping (address => uint256) public balances;

    constructor() public {
        owner = msg.sender;
    }

    function transfer(address recipient, uint256 amount) public {
        require(msg.sender == owner, "Only the owner can transfer funds");
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        balances[recipient] += amount;
    }

    function getBalance(address account) public view returns (uint256) {
        return balances[account];
    }
}
