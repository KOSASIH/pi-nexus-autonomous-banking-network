pragma solidity ^0.8.4;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Ownable.sol";

contract PIBankRewards is Ownable {
    address public owner;
    mapping (address => uint256) public rewards;

event RewardDistributed(address indexed recipient, uint256 rewardAmount);

    constructor() public {
        owner = msg.sender;
    }

    function distributeReward(address recipient, uint256 rewardAmount) public {
        rewards[recipient] += rewardAmount;
        emit RewardDistributed(recipient, rewardAmount);
    }

    function getReward(address recipient) public view returns (uint256) {
        return rewards[recipient];
    }
}
