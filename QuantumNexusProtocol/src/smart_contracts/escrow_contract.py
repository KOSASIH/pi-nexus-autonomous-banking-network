// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract EscrowContract {
    address public buyer;
    address public seller;
    address public arbiter;
    uint256 public amount;
    bool public isCompleted;

    event EscrowCreated(address indexed buyer, address indexed seller, uint256 amount);
    event EscrowCompleted(address indexed buyer, address indexed seller);
    event EscrowRefunded(address indexed buyer);

    constructor(address _seller, address _arbiter) {
        buyer = msg.sender;
        seller = _seller;
        arbiter = _arbiter;
    }

    function deposit() external payable {
        require(msg.sender == buyer, "Only buyer can deposit");
        require(amount == 0, "Already deposited");
        amount = msg.value;
        emit EscrowCreated(buyer, seller, amount);
    }

    function complete() external {
        require(msg.sender == arbiter, "Only arbiter can complete");
        require(!isCompleted, "Escrow already completed");
        isCompleted = true;
        payable(seller).transfer(amount);
        emit EscrowCompleted(buyer, seller);
    }

    function refund() external {
        require(msg.sender == arbiter , "Only arbiter can refund");
        require(!isCompleted, "Escrow already completed");
        isCompleted = true;
        payable(buyer).transfer(amount);
        emit EscrowRefunded(buyer);
    }
}
