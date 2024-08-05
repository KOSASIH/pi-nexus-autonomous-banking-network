pragma solidity ^0.8.0;

contract LiquidityPoolContract {
    mapping (address => uint256) public liquidityProviders;
    uint256 public totalLiquidity;

    function provideLiquidity(uint256 _amount) public {
        require(_amount > 0, "Amount must be greater than 0");
        liquidityProviders[msg.sender] += _amount;
        totalLiquidity += _amount;
    }

    function withdrawLiquidity(uint256 _amount) public {
        require(_amount > 0, "Amount must be greater than 0");
        require(liquidityProviders[msg.sender] >= _amount, "Insufficient liquidity");
        liquidityProviders[msg.sender] -= _amount;
        totalLiquidity -= _amount;
    }
}
