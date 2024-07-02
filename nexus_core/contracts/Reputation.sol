pragma solidity ^0.8.0;

contract Reputation {
    mapping (address => uint256) public reputations;

    constructor() {
        // Initialize reputation mapping
    }

    function updateReputation(address account, uint256 score) public {
        reputations[account] = score;
    }

    function getReputation(address account) public view returns (uint256) {
        return reputations[account];
    }
}
