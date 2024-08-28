pragma solidity ^0.8.0;

import "./LiquidityProvider.sol";

contract LiquidityPool {
    using LiquidityProvider for address;

    // Mapping of liquidity providers
    mapping (address => LiquidityProvider) public liquidityProviders;

    // Event emitted when a new liquidity provider joins the pool
    event NewLiquidityProvider(address indexed provider);

    // Function to join the liquidity pool
    function joinLiquidityPool() public {
        LiquidityProvider storage provider = liquidityProviders[msg.sender];
        provider.provideLiquidity(1000); // Initial liquidity provision
        emit NewLiquidityProvider(msg.sender);
    }

    // Function to get the total liquidity level
    function getTotalLiquidityLevel() public view returns (uint256) {
        uint256 totalLiquidity = 0;
        for (address provider in liquidityProviders) {
            totalLiquidity += liquidityProviders[provider].getLiquidityLevel();
        }
        return totalLiquidity;
    }
}
