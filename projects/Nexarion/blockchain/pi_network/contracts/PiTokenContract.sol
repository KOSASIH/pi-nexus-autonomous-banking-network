pragma solidity ^0.8.0;

contract PiTokenContract {
    // Mapping of user addresses to their PI token balances
    mapping (address => uint256) public piTokenBalances;

    // Event emitted when a user's PI token balance is updated
    event PiTokenBalanceUpdated(address indexed userAddress, uint256 newBalance);

    // Constructor function
    constructor() public {
        // Initialize PI token balances
        piTokenBalances[msg.sender] = 1000000; // 1 million PI tokens for the contract creator
    }

    // Function to transfer PI tokens
    function transfer(address recipient, uint256 amount) public {
        // Check if the sender has enough PI tokens
        require(piTokenBalances[msg.sender] >= amount, "Insufficient PI tokens");

        // Update the sender's PI token balance
        piTokenBalances[msg.sender] -= amount;

        // Update the recipient's PI token balance
        piTokenBalances[recipient] += amount;

        // Emit event to notify the network of the PI token transfer
        emit PiTokenBalanceUpdated(recipient, piTokenBalances[recipient]);
    }
}
