pragma solidity ^0.8.0;

contract LendingPool {
    address public owner;
    mapping (address => uint256) public deposits;
    mapping (address => uint256) public borrows;

    constructor() {
        owner = msg.sender;
    }

    function deposit(uint256 amount) public {
        deposits[msg.sender] += amount;
    }

    function borrow(uint256 amount) public {
        require(deposits[msg.sender] >= amount, "Insufficient deposit");
        borrows[msg.sender] += amount;
    }

    function repay(uint256 amount) public {
        require(borrows[msg.sender] >= amount, "Insufficient borrow");
        borrows[msg.sender] -= amount;
    }

    function getDeposit(address account) public view returns (uint256) {
        return deposits[account];
    }

    function getBorrow(address account) public view returns (uint256) {
        return borrows[account];
    }
}
