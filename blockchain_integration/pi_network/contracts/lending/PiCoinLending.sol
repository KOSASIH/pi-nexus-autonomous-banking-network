// PiCoinLending.sol
pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract PiCoinLending {
    using SafeERC20 for IERC20;

    // Mapping of lenders to their lent Pi Coin balances
    mapping (address => uint256) public lentBalances;

    // Event emitted when Pi Coins are lent
    event PiCoinsLent(address indexed lender, uint256 amount);

    // Function to lend Pi Coins
    function lendPiCoins(uint256 amount) public {
        require(amount > 0, "Invalid lending amount");
        lentBalances[msg.sender] += amount;
        emit PiCoinsLent(msg.sender, amount);
    }

    // Function to borrow Pi Coins
    function borrowPiCoins(uint256 amount) public {
        require(lentBalances[msg.sender] >= amount, "Insufficient lent balance");
        lentBalances[msg.sender] -= amount;
        emit PiCoinsBorrowed(msg.sender, amount);
    }
}
