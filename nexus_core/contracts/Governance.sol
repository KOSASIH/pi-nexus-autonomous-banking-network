pragma solidity ^0.8.0;

contract Governance {
    address public owner;
    mapping (address => uint256) public votes;

    constructor() {
        owner = msg.sender;
    }

    function vote(address proposal, uint256 amount) public {
        votes[proposal] += amount;
    }

    function getVote(address proposal) public view returns (uint256) {
        return votes[proposal];
    }
}
