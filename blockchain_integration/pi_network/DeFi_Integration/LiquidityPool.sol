pragma solidity ^0.8.0;

import {ERC20} from "./ERC20.sol";

contract LiquidityPool {
    // Mapping of liquidity providers to their corresponding liquidity balances
    mapping (address => uint256) public liquidityBalances;

    // Event emitted when a liquidity provider deposits Pi Coin into the liquidity pool
    event LiquidityProviderDeposit(address indexed liquidityProvider, uint256 amount);

    // Event emitted when a liquidity provider withdraws Pi Coin from the liquidity pool
    event LiquidityProviderWithdrawal(address indexed liquidityProvider, uint256 amount);

    // Function to deposit Pi Coin into the liquidity pool
    function deposit(uint256 _amount) public {
        // Transfer Pi Coin from liquidity provider to liquidity pool
        ERC20.transferFrom(msg.sender, address(this), _amount);

        // Update liquidity provider's liquidity balance
        liquidityBalances[msg.sender] += _amount;

        // Emit liquidity provider deposit event
        emit LiquidityProviderDeposit(msg.sender, _amount);
    }

    // Function to withdraw Pi Coin from the liquidity pool
    function withdraw(uint256 _amount) public {
        // Check if liquidity provider has sufficient liquidity balance
        require(liquidityBalances[msg.sender] >= _amount, "Insufficient liquidity balance");

        // Transfer Pi Coin from liquidity pool to liquidity provider
        ERC20.transfer(msg.sender, _amount);

        // Update liquidity provider's liquidity balance
        liquidityBalances[msg.sender] -= _amount;

        // Emit liquidity provider withdrawal event
        emit LiquidityProviderWithdrawal(msg.sender, _amount);
    }
}
