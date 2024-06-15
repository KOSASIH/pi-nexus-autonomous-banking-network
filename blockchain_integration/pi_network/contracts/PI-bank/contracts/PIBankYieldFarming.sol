pragma solidity ^0.8.0;

import "./PIBank.sol";

contract PIBankYieldFarming {
    // Mapping of yield farming balances
    mapping(address => uint256) public yieldFarmingBalances;

    // Event
    event NewYieldFarming(address indexed user, uint256 amount);

    // Function
    function yieldFarm(address user, uint256 amount) public {
        // Update yield farming balances
        yieldFarmingBalances[user] = amount;
        emit NewYieldFarming(user, amount);
    }
}
