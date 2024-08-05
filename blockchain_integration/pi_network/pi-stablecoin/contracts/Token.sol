pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract Token {
    using SafeERC20 for address;

    // Mapping of token balances
    mapping (address => uint256) public balances;

    // Event emitted when tokens are transferred
    event Transfer(address indexed from, address indexed to, uint256 value);

    // Event emitted when tokens are minted
    event Mint(address indexed to, uint256 value);

    // Event emitted when tokens are burned
    event Burn(address indexed from, uint256 value);

    // Function to mint new tokens
    function mint(address to, uint256 value) public {
        // Only allow minting by authorized addresses
        require(msg.sender == governanceContract, "Only governance contract can mint tokens");

        // Mint new tokens
        balances[to] += value;
        emit Mint(to, value);
    }

    // Function to transfer tokens
    function transfer(address to, uint256 value) public {
        // Check if sender has sufficient balance
        require(balances[msg.sender] >= value, "Insufficient balance");

        // Transfer tokens
        balances[msg.sender] -= value;
        balances[to] += value;
        emit Transfer(msg.sender, to, value);
    }

    // Function to burn tokens
    function burn(uint256 value) public {
        // Check if sender has sufficient balance
        require(balances[msg.sender] >= value, "Insufficient balance");

        // Burn tokens
        balances[msg.sender] -= value;
        emit Burn(msg.sender, value);
    }
}
