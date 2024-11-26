// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DeFiContract {
    mapping(address => uint256) public deposits;
    mapping(address => uint256) public loans;

    event Deposited(address indexed user, uint256 amount);
    event Loaned(address indexed user, uint256 amount);

    function deposit() external payable {
        require(msg.value > 0, "Deposit amount must be greater than zero");
        deposits[msg.sender] += msg.value;
        emit Deposited(msg.sender, msg.value);
    }

    function takeLoan(uint256 amount) external {
        require(deposits[msg.sender] > 0, "No deposits to secure a loan");
        loans[msg.sender] += amount;
        payable(msg.sender).transfer(amount);
        emit Loaned(msg.sender, amount);
    }
}
