pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";

contract RiskManagement {
    // Mapping of user positions
    mapping (address => Position) public userPositions;

    // Function to update user position
    function updatePosition(address user, uint256 amount, uint256 leverage) public {
        // Update user position
        Position storage position = userPositions[user];
        position.amount = amount;
        position.leverage = leverage;
    }
}
