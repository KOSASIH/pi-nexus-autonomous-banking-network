// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TransactionContract {
    event TransactionExecuted(address indexed from, address indexed to, uint256 amount);

    function executeTransaction(address payable to) external payable {
        require(msg.value > 0, "Transaction value must be greater than zero");
        to.transfer(msg.value);
        emit TransactionExecuted(msg.sender, to, msg.value);
    }
}
