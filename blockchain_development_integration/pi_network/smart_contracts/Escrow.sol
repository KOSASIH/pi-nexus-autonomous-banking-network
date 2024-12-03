// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Escrow {
    address public buyer;
    address public seller;
    address public arbiter;
    uint256 public amount;
    bool public isCompleted;

    constructor(address _seller, address _arbiter) {
        buyer = msg.sender;
        seller = _seller;
        arbiter = _arbiter;
        isCompleted = false;
    }

    function deposit() external payable {
        require(msg.sender == buyer, "Only buyer can deposit");
        require(amount == 0, "Already deposited");
        amount = msg.value;
    }

    function releaseFunds() external {
        require(msg.sender == arbiter, "Only arbiter can release funds");
        require(!isCompleted, "Escrow already completed");
        payable(seller).transfer(amount);
        isCompleted = true;
    }

    function refund() external {
        require(msg.sender == arbiter, "Only arbiter can refund");
        require(!isCompleted, "Escrow already completed");
        payable(buyer).transfer(amount);
        isCompleted = true;
    }
}
