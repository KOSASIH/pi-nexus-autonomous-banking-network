pragma solidity ^0.8.0;

contract Staking {
    address public owner;
    mapping (address => uint256) public stakes;

    constructor() {
        owner = msg.sender;
    }

    function stake(uint256 amount) public {
        stakes[msg.sender] += amount;
    }

    function unstake(uint256 amount) public {
        require(stakes[msg.sender] >= amount, "Insufficient stake");
        stakes[msg.sender] -= amount;
    }

    function getStake(address account) public view returns (uint256) {
        return stakes[account];
    }
}
