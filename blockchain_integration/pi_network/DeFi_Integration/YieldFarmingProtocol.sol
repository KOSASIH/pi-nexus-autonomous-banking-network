pragma solidity ^0.8.0;

import {DeFiIntegration} from "./DeFiIntegration.sol";
import {ERC20} from "./ERC20.sol";

contract YieldFarmingProtocol {
    // Mapping of yield farmers to their corresponding yield farming balances
    mapping (address => uint256) public yieldFarmingBalances;

    // Event emitted when a yield farmer deposits Pi Coin into the yield farming protocol
    event YieldFarmerDeposit(address indexed yieldFarmer, uint256 amount);

    // Event emitted when a yield farmer withdraws Pi Coin from the yield farming protocol
    event YieldFarmerWithdrawal(address indexed yieldFarmer, uint256 amount);

    // Event emitted when a yield farmer earns interest on their Pi Coin deposit
    event YieldFarmerInterest(address indexed yieldFarmer, uint256 amount);

    // Function to deposit Pi Coin into the yield farming protocol
    function deposit(uint256 _amount) public {
        // Transfer Pi Coin from yield farmer to yield farming protocol
        DeFiIntegration.transferFrom(msg.sender, address(this), _amount);

        // Update yield farmer's yield farming balance
        yieldFarmingBalances[msg.sender] += _amount;

        // Emit yield farmer deposit event
        emit YieldFarmerDeposit(msg.sender, _amount);
    }

    // Function to withdraw Pi Coin from the yield farming protocol
    function withdraw(uint256 _amount) public {
        // Check if yield farmer has sufficient yield farming balance
        require(yieldFarmingBalances[msg.sender] >= _amount, "Insufficient yield farming balance");

        // Transfer Pi Coin from yield farming protocol to yield farmer
        DeFiIntegration.transfer(msg.sender, _amount);

        // Update yield farmer's yield farming balance
        yieldFarmingBalances[msg.sender] -= _amount;

        // Emit yield farmer withdrawal event
        emit YieldFarmerWithdrawal(msg.sender, _amount);
    }

    // Function to earn interest on Pi Coin deposit
    function earnInterest() public {
        // Calculate interest earned by yield farmer
        uint256 interest = yieldFarmingBalances[msg.sender] * 10 / 100;

        // Update yield farmer's yield farming balance
        yieldFarmingBalances[msg.sender] += interest;

        // Emit yield farmer interest event
        emit YieldFarmerInterest(msg.sender, interest);
    }
}
