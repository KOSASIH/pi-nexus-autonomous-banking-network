pragma solidity ^0.8.0;

import "./PIBank.sol";

contract PIBankFiatGateway {
    // Mapping of fiat deposits
    mapping(address => uint256) public fiatDeposits;

    // Event
    event NewFiatDeposit(address indexed user, uint256 amount);

    // Function
    function depositFiat(address user, uint256 amount) public {
        // Update fiat deposits
        fiatDeposits[user] = amount;
        emit NewFiatDeposit(user, amount);
    }
}
