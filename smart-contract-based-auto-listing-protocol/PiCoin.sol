pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/ERC20.sol";

contract PiCoin is ERC20 {
    // Mapping of Pi Coin balances
    mapping (address => uint256) public balances;

    // Event emitted when a Pi Coin is transferred
    event Transfer(address indexed from, address indexed to, uint256 value);

    // Function to transfer Pi Coins
    function transfer(address to, uint256 value) public {
        require(to != address(0), "Recipient address cannot be zero");
        require(value > 0, "Value must be greater than zero");
        balances[msg.sender] -= value;
        balances[to] += value;
        emit Transfer(msg.sender, to, value);
    }

    // Function to get the balance of a Pi Coin holder
    function balanceOf(address holder) public view returns (uint256) {
        return balances[holder];
    }
}
