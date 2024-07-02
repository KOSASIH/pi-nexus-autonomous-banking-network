pragma solidity ^0.8.0;

contract Token {
    string public name;
    string public symbol;
    uint256 public totalSupply;

    mapping (address => uint256) public balances;

    constructor() {
        name = "Pi Nexus Token";
        symbol = "PNT";
        totalSupply = 1000000;
        balances[msg.sender] = totalSupply;
    }

    function transfer(address recipient, uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        balances[recipient] += amount;
    }

    function balanceOf(address account) public view returns (uint256) {
        return balances[account];
    }
}
