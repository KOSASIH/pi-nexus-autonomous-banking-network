pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract PiNexusToken is SafeERC20 {
    // Token properties
    string public name;
    string public symbol;
    uint256 public totalSupply;

    // Token constructor
    constructor() public {
        name = "Pi Nexus Token";
        symbol = "PNT";
        totalSupply = 1000000000;
    }

    // Token functions
    function transfer(address to, uint256 amount) public {
        // Transfer tokens to another address
        _transfer(msg.sender, to, amount);
    }

    function approve(address spender, uint256 amount) public {
        // Approve another address to spend tokens
        _approve(msg.sender, spender, amount);
    }

    function transferFrom(address from, address to, uint256 amount) public {
        // Transfer tokens from one address to another
        _transfer(from, to, amount);
    }
}
