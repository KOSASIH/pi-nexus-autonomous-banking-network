pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract DecentralizedExchange {
    // Mapping of token addresses to token balances
    mapping (address => mapping (address => uint256)) public tokenBalances;

    // Event emitted when a token transfer is executed
    event TokenTransferExecuted(address sender, address recipient, address token, uint256 amount);

    // Function to execute a token transfer
    function executeTokenTransfer(address _token, address _recipient, uint256 _amount) public {
        // Get token balance for sender
        uint256 senderBalance = tokenBalances[msg.sender][_token];

        // Check if sender has enough balance
        require(senderBalance >= _amount, "Insufficient balance");

        // Transfer tokens
        tokenBalances[msg.sender][_token] = senderBalance.sub(_amount);
        tokenBalances[_recipient][_token] = tokenBalances[_recipient][_token].add(_amount);

        // Emit token transfer executed event
        emit TokenTransferExecuted(msg.sender, _recipient, _token, _amount);
    }
}
