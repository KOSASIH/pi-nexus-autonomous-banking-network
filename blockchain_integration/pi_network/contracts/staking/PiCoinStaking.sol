// PiCoinStaking.sol
pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract PiCoinStaking {
    using SafeERC20 for IERC20;

    // Mapping of user addresses to their staked Pi Coin balances
    mapping (address => uint256) public stakedBalances;

    // Event emitted when Pi Coins are staked
    event PiCoinsStaked(address indexed user, uint256 amount);

    // Function to stake Pi Coins
    function stakePiCoins(uint256 amount) public {
        require(amount > 0, "Invalid staking amount");
        stakedBalances[msg.sender] += amount;
        emit PiCoinsStaked(msg.sender, amount);
    }

    // Function to unstake Pi Coins
    function unstakePiCoins(uint256 amount) public {
        require(stakedBalances[msg.sender] >= amount, "Insufficient staked balance");
        stakedBalances[msg.sender] -= amount;
        emit PiCoinsUnstaked(msg.sender, amount);
    }
}
