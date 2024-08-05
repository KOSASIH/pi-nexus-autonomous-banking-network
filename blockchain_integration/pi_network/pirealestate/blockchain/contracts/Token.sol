pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract Token {
    // Mapping of token balances
    mapping (address => uint256) public balances;

    // Mapping of token allowances
    mapping (address => mapping (address => uint256)) public allowances;

    // Event emitted when tokens are transferred
    event Transfer(address from, address to, uint256 amount);

    // Event emitted when tokens are approved
    event Approval(address owner, address spender, uint256 amount);

    // Function to transfer tokens
    function transfer(address to, uint256 amount) public {
        // Check if the sender has enough tokens
        require(balances[msg.sender] >= amount, "Sender does not have enough tokens");

        // Transfer the tokens
        balances[msg.sender] -= amount;
        balances[to] += amount;

        // Emit an event to notify the transfer of tokens
        emit Transfer(msg.sender, to, amount);
    }

    // Function to approve tokens
    function approve(address spender, uint256 amount) public {
        // Check if the sender has enough tokens
        require(balances[msg.sender] >= amount, "Sender does not have enough tokens");

        // Approve the tokens
        allowances[msg.sender][spender] = amount;

        // Emit an event to notify the approval of tokens
        emit Approval(msg.sender, spender, amount);
    }

    // Function to get the token balance
    function balanceOf(address owner) public view returns (uint256) {
        // Return the token balance
        return balances[owner];
    }

    // Function to get the token allowance
    function allowance(address owner, address spender) public view returns (uint256) {
        // Return the token allowance
        return allowances[owner][spender];
    }
}
