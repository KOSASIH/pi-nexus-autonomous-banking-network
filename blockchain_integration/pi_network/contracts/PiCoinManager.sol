// PiCoinManager.sol
pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract PiCoinManager {
    using SafeERC20 for IERC20;

    // Mapping of user addresses to their Pi Coin balances
    mapping (address => uint256) public piCoinBalances;

    // Event emitted when Pi Coins are transferred
    event PiCoinTransfer(address indexed from, address indexed to, uint256 amount);

    // Function to transfer Pi Coins between users
    function transferPiCoins(address from, address to, uint256 amount) public {
        require(piCoinBalances[from] >= amount, "Insufficient Pi Coin balance");
        piCoinBalances[from] -= amount;
        piCoinBalances[to] += amount;
        emit PiCoinTransfer(from, to, amount);
    }

    // Function to mint new Pi Coins
    function mintPiCoins(address to, uint256 amount) public {
        piCoinBalances[to] += amount;
        emit PiCoinTransfer(address(0), to, amount);
    }
}
