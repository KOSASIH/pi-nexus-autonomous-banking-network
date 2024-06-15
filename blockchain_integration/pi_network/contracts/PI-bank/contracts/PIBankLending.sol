pragma solidity ^0.8.0;

import "./PIBank.sol";

contract PIBankLending {
    // Mapping of lending balances
    mapping(address => mapping(address => uint256)) public lendingBalances;

    // Event
    event NewLending(address indexed lender, address indexed borrower, uint256 amount);

    // Function
    function lend(address lender, address borrower, uint256 amount) public {
        // Update lending balances
        lendingBalances[lender][borrower] = amount;
        emit NewLending(lender, borrower, amount);
    }
}
