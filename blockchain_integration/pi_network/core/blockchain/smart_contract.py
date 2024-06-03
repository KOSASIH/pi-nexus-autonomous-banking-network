pragma solidity ^0.8.0;

contract AutonomousBankingContract {
    address private owner;
    mapping (address => uint256) public balances;

    constructor() public {
        owner = msg.sender;
    }

    function deposit(uint256 amount) public {
        balances[msg.sender] += amount;
    }

    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
    }

    function getBalance(address account) public view returns (uint256) {
        return balances[account];
    }
}
