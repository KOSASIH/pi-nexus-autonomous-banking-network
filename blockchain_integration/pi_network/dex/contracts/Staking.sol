pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/SafeERC20.sol";

contract Staking {
    mapping(address => uint256) public stakes;

    function stake(uint256 amount) external {
        //...
    }

    function unstake(uint256 amount) external {
        //...
    }

    function getStake(address user) public view returns (uint256) {
        //...
    }
}
