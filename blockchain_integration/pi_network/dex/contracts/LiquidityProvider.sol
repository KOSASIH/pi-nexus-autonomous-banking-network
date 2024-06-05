pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "./PiDex.sol";

contract LiquidityProvider {
    using SafeMath for uint256;

    // Mapping of LPs
    mapping (address => uint256) public lpBalances;

    // Mapping of LP shares
    mapping (address => uint256) public lpShares;

    // Event
    event Deposit(address indexed lp, uint256 amount);
    event Withdraw(address indexed lp, uint256 amount);

    // Deposit liquidity
    function deposit(uint256 amount) public {
        require(amount > 0, "LiquidityProvider: invalid amount");
        require(msg.sender!= address(0), "LiquidityProvider: invalid sender");

        // Update LP balance and shares
        lpBalances[msg.sender] = lpBalances[msg.sender].add(amount);
        lpShares[msg.sender] = lpShares[msg.sender].add(amount);

        // Emit event
        emit Deposit(msg.sender, amount);
    }

    // Withdraw liquidity
    function withdraw(uint256 amount) public {
        require(amount > 0, "LiquidityProvider: invalid amount");
        require(lpBalances[msg.sender] >= amount, "LiquidityProvider: insufficient balance");

        // Update LP balance and shares
        lpBalances[msg.sender] = lpBalances[msg.sender].sub(amount);
        lpShares[msg.sender] = lpShares[msg.sender].sub(amount);

        // Emit event
        emit Withdraw(msg.sender, amount);
    }

    // Get LP balance
    function getBalance(address lp) public view returns (uint256) {
        return lpBalances[lp];
    }

    // Get LP shares
    function getShares(address lp) public view returns (uint256) {
        return lpShares[lp];
    }
}
