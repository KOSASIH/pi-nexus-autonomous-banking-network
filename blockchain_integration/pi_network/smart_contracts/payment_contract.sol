// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract PaymentContract {
    event PaymentReceived(address indexed payer, uint256 amount);
    event PaymentSent(address indexed payee, uint256 amount);

    // Receive payment
    receive() external payable {
        emit PaymentReceived(msg.sender, msg.value);
    }

    // Send payment to a specified address
    function sendPayment(address payable _payee) public payable {
        require(msg.value > 0, "Payment amount must be greater than zero.");
        _payee.transfer(msg.value);
        emit PaymentSent(_payee, msg.value);
    }

    // Get contract balance
    function getBalance() public view returns (uint256) {
        return address(this).balance;
    }
}
